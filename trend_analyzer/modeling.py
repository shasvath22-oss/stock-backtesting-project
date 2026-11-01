from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import complete, read_json, require, write_json


def splits(run):
    run = Path(run)
    cfg = require(run, "features")
    schema = read_json(run / "feature_schema.json")
    df = pd.read_parquet(run / "features.parquet").dropna(subset=schema["columns"])
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    periods = cfg.get("research_periods")
    if periods:
        from .periods import period_windows
        windows = period_windows(dates,periods)
        dates = windows["old"].union(windows["new"])
        cut = len(windows["old"])
    else:
        cut = int(len(dates)*(1-cfg["test_fraction"]))
    if cut <= cfg["initial_train_sessions"]:
        raise ValueError("Not enough post-warmup dates for requested train/development/test windows")
    if cut >= len(dates):
        raise ValueError("No eligible final-test dates; collect prices covering the selected new period")
    folds = []
    for phase, begin, end in [("development", cfg["initial_train_sessions"], cut), ("test", cut, len(dates))]:
        for i in range(begin, end, cfg["fold_sessions"]):
            stop = min(i+cfg["fold_sessions"], end)
            # Train at previous close; labels exiting on the next evaluation open are unknown.
            folds.append({"fold": len(folds), "phase": phase, "fit_cutoff": str(dates[i-1].date()),
                          "start": str(dates[i].date()), "end": str(dates[stop-1].date()),
                          "train_start":str(dates[0].date()),"phase_end":str(dates[end-1].date())})
            if periods:
                folds[-1]["old_end"] = periods["old_end"]
                folds[-1]["new_start"] = periods["new_start"]
    write_json(run / "splits.json", {"test_start": str(dates[cut].date()), "folds": folds,
        "partitions":{"initial_training":{"start":str(dates[0].date()),"end":str(dates[cfg['initial_train_sessions']-1].date())},
            "validation":{"start":str(dates[cfg['initial_train_sessions']].date()),"end":str(dates[cut-1].date())},
            "test":{"start":str(dates[cut].date()),"end":str(dates[-1].date())}},
        "policy": "train dates and target exits <= fit_cutoff; fixed parameters, no test-driven tuning"})
    complete(run, "splits", {"folds": len(folds)})


def fold_masks(df, fold):
    cutoff = pd.Timestamp(fold["fit_cutoff"])
    # Exclude training rows whose outcomes were not known by the fitting cutoff.
    train = df.date.le(cutoff) & df.exit_date.le(cutoff) & df.target.notna()
    if "train_start" in fold:
        train &= df.date.ge(fold["train_start"])
    if "old_end" in fold:
        train &= df.date.le(fold["old_end"]) | df.date.ge(fold["new_start"])
    evaluation = df.date.between(fold["start"], fold["end"])
    return train, evaluation


def estimators(cfg):
    from xgboost import XGBClassifier
    return {
        "always_up": DummyClassifier(strategy="constant", constant=1),
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1)),
        "xgboost": XGBClassifier(**cfg["xgboost"], objective="binary:logistic", eval_metric="logloss",
            random_state=cfg["seed"], n_jobs=2, tree_method="hist"),
    }


def train(run, horizon=None):
    run = Path(run)
    cfg = require(run, "splits")
    columns = read_json(run / "feature_schema.json")["columns"]
    data = pd.read_parquet(run / "features.parquet").dropna(subset=columns)
    folds = read_json(run / "splits.json")["folds"]
    root = run / "models"
    root.mkdir(exist_ok=True)
    horizons = [horizon] if horizon else cfg["horizons"]
    if not set(horizons).issubset(cfg["horizons"]):
        raise ValueError("Requested horizon is not configured")
    for h in horizons:
        for stale in root.glob(f"h{h}_f*.joblib"):
            stale.unlink()
        df = data[data.horizon.eq(h)]
        for fold in folds:
            mask, _ = fold_masks(df, fold)
            fit = df[mask]
            if fit.target.nunique() < 2:
                raise ValueError(f"Insufficient classes for horizon {h}, fold {fold['fold']}")
            models = estimators(cfg)
            for model in models.values():
                model.fit(fit[columns], fit.target.astype(int))
            bundle = {"models": models, "columns": columns, "horizon": h, "fold": fold,
                "max_train_date": fit.date.max(), "max_label_exit": fit.exit_date.max(), "rows": len(fit)}
            joblib.dump(bundle, root / f"h{h}_f{fold['fold']}.joblib")
        complete(run, f"train_h{h}", {"folds": len(folds)})
    complete(run, "train", {"horizons": [h for h in cfg["horizons"] if f"train_h{h}" in read_json(run/"manifest.json")["stages"]]})


def predict(run):
    run = Path(run)
    cfg = require(run, "train")
    columns = read_json(run / "feature_schema.json")["columns"]
    data = pd.read_parquet(run/"features.parquet").dropna(subset=columns)
    out = []
    for path in sorted((run/"models").glob("*.joblib")):
        trained_h = int(path.stem.split("_")[0][1:])
        if f"train_h{trained_h}" not in read_json(run/"manifest.json")["stages"]:
            continue
        bundle = joblib.load(path)  # Only load model files created in this trusted local run.
        df = data[data.horizon.eq(bundle["horizon"])]
        fold = bundle["fold"]
        _, mask = fold_masks(df, fold)
        evaluation = df[mask]
        if evaluation.empty:
            continue
        for name, model in bundle["models"].items():
            p = evaluation[["date","ticker","horizon","entry_date","exit_date","target","forward_return"]].copy()
            probs = model.predict_proba(evaluation[columns])
            p["probability"] = probs[:, list(model.classes_).index(1)] if 1 in model.classes_ else 0.
            p["model"], p["fold"], p["phase"] = name, fold["fold"], fold["phase"]
            p["fit_cutoff"] = fold["fit_cutoff"]
            p["max_label_exit"] = bundle["max_label_exit"]
            p["phase_end"] = fold.get("phase_end",fold["end"])
            # Final validation outcomes must not extend into final test.
            outside = p.exit_date.gt(pd.Timestamp(p.phase_end.iloc[0]))
            p.loc[outside,["target","forward_return"]] = np.nan
            out.append(p)
        p = out[-1].copy()
        p["model"], p["probability"] = "momentum", evaluation.return_5.gt(0).astype(float)
        out.append(p)
    if not out:
        raise ValueError("No trained model bundles found")
    predictions = pd.concat(out, ignore_index=True)
    predictions["signal"] = predictions.probability.ge(cfg["threshold"])
    predictions.to_parquet(run/"predictions.parquet", index=False)
    records = []
    valid = predictions.dropna(subset=["target"])
    for keys, block in valid.groupby(["horizon", "model", "phase"]):
        records.append(dict(zip(["horizon","model","phase"], keys)) | scores(block))
    pd.DataFrame(records).to_csv(run/"accuracy.csv", index=False)
    details = []
    for keys, block in valid.groupby(["horizon", "model", "phase", "fold", "ticker"]):
        details.append(dict(zip(["horizon","model","phase","fold","ticker"], keys)) | scores(block))
    pd.DataFrame(details).to_csv(run/"accuracy_detail.csv", index=False)
    complete(run, "predict", {"rows": len(predictions)})


def scores(block):
    y, p = block.target.astype(int), block.probability.clip(1e-7, 1-1e-7)
    label = block.probability.ge(.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, label, labels=[0,1]).ravel()
    recalls = ([tn/(tn+fp)] if tn+fp else []) + ([tp/(tp+fn)] if tp+fn else [])
    return {"samples": len(y), "up_fraction": float(y.mean()), "accuracy": accuracy_score(y,label),
        "balanced_accuracy": float(np.mean(recalls)), "precision": precision_score(y,label,zero_division=0),
        "recall": recall_score(y,label,zero_division=0), "log_loss": log_loss(y,p,labels=[0,1]),
        "brier": brier_score_loss(y,p), "auc": roc_auc_score(y,p) if y.nunique()==2 else np.nan,
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}
