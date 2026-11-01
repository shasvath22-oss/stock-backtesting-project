from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_run(config_path="config/research.json", demo=False, full=False, learning=False, overrides=None):
    cfg = read_json(config_path)
    if overrides:
        cfg.update(overrides)
    if "research_periods" not in cfg:
        from .periods import period_settings
        cfg["research_periods"] = period_settings()
    if not cfg.get("end"):
        cfg["end"] = str(pd.Timestamp.now(tz="America/New_York").date())
    symbols = pd.read_csv(cfg["universe"])["ticker"].tolist()
    cfg["tickers"] = symbols if full else cfg.get("symbols",symbols[:cfg["ticker_limit"]])
    if demo:
        cfg.update(start="2018-01-01", end="2024-01-01", minimum_sessions=500,
                   initial_train_sessions=504, fold_sessions=126)
        cfg["tickers"] = symbols[:3]
        cfg["xgboost"]["n_estimators"] = 40
        if "research_periods" in cfg:
            cfg["research_periods"] = {"old_start":"2018-01-01","old_end":"2022-12-31","new_start":"2023-01-01","new_end":None}
    if learning:
        cfg["tickers"] = ["AAPL"]
        cfg["horizons"] = [5]
    if pd.Timestamp(cfg["start"]) >= pd.Timestamp(cfg["end"]):
        raise ValueError("Start must be before the exclusive end date")
    if pd.Timestamp(cfg["end"]) > pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize():
        raise ValueError("End must exclude the incomplete current session")
    if len(set(cfg["tickers"])) != len(cfg["tickers"]):
        raise ValueError("Universe has duplicate ticker symbols")
    if not 0 < cfg["test_fraction"] < 0.5 or not 0.5 <= cfg["threshold"] < 1:
        raise ValueError("Invalid holdout fraction or trading threshold")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path("artifacts") / run_id
    run.mkdir(parents=True)
    versions = {}
    for name in ["numpy", "pandas", "yfinance", "scikit-learn", "xgboost", "streamlit"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "project-local or unavailable"
    write_json(run / "manifest.json", {"run_id": run_id, "synthetic": demo, "learning": learning,
               "workflow":"multi_horizon_research",
               "config": cfg, "versions": versions, "stages": {},
               "price_convention": "Yahoo split-normalized OHLC; dividends credited separately; no split applied twice"})
    return run


def manifest(run):
    return read_json(Path(run) / "manifest.json")


def complete(run, stage, details=None):
    m = manifest(run)
    m["stages"][stage] = {"completed_utc": datetime.now(timezone.utc).isoformat(), "details": details}
    write_json(Path(run) / "manifest.json", m)


def begin_stage(run, stage, horizon=None):
    """Invalidate downstream artifacts before work, including if the stage fails."""
    order = ["collect", "features", "splits", "train", "predict", "backtest", "report"]
    m = manifest(run)
    index = order.index(stage)
    for key in list(m["stages"]):
        base = "train" if key.startswith("train_h") else key
        if base in order and order.index(base) >= index:
            if stage=="train" and key.startswith("train_h") and horizon is not None and key!=f"train_h{horizon}":
                continue
            del m["stages"][key]
    write_json(Path(run)/"manifest.json",m)
    if index <= order.index("predict"):
        for name in ["performance.csv", "report.html"]:
            stale = Path(run)/name
            if stale.exists():
                stale.unlink()


def require(run, *stages):
    m = manifest(run)
    for stage in stages:
        if stage not in m["stages"]:
            raise ValueError(f"Run {stage} first for {Path(run).name}")
    return m["config"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
