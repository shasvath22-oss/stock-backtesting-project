"""Small, fixed XGBoost experiment for confirmation-time HH/HL signals."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from .swing_backtest import rule_signals, simulate_rule
from .swings import swings_asof


FEATURES = ["return_5","return_20","volatility_20","distance_to_stop","distance_to_target",
            "range_fraction","swing_sessions","relative_volume_20"]
PARAMETERS = dict(n_estimators=40,max_depth=2,learning_rate=.05,subsample=.8,
    colsample_bytree=.8,objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=2)


def signal_features(bars,events):
    data = bars.sort_values("date").set_index("date")
    returns = data.close.pct_change(fill_method=None)
    records = []
    for signal in rule_signals(bars,events).itertuples():
        i = data.index.get_loc(signal.confirmed_at)
        past = data.iloc[:i+1]
        close = float(past.close.iloc[-1])
        points = swings_asof(events,signal.confirmed_at)
        current = points.iloc[-1]
        segment = points[points.segment.eq(current.segment)]
        previous = segment.iloc[-2]
        records.append({"confirmed_at":signal.confirmed_at,"stop":signal.stop,"target":signal.target,
            "combination":signal.combination,
            "return_5":close/data.close.iloc[i-5]-1 if i>=5 else np.nan,
            "return_20":close/data.close.iloc[i-20]-1 if i>=20 else np.nan,
            "volatility_20":returns.iloc[:i+1].tail(20).std() if i>=20 else np.nan,
            "distance_to_stop":(close-signal.stop)/close,
            "distance_to_target":(signal.target-close)/close,
            "range_fraction":(signal.target-signal.stop)/close,
            "swing_sessions":int(data.index.searchsorted(current.pivot_date)-data.index.searchsorted(previous.pivot_date)),
            "relative_volume_20":past.volume.iloc[-1]/past.volume.tail(20).mean() if len(past)>=20 and past.volume.tail(20).mean()>0 else np.nan})
    return pd.DataFrame(records,columns=["confirmed_at","stop","target","combination"]+FEATURES).replace([np.inf,-np.inf],np.nan)


def label_old_signals(bars,features,old_dates,max_hold=20):
    old_bars = bars[bars.date.isin(old_dates)]
    old = features[features.confirmed_at.isin(old_dates)].dropna(subset=FEATURES)
    records = []
    for row in old.itertuples(index=False):
        single = pd.DataFrame([row._asdict()])
        _,_,trades,_ = simulate_rule(old_bars,single,1000,10,max_hold)
        if trades.empty:
            continue # No executable next open inside levels.
        trade = trades.iloc[0]
        if trade.reason == "end_of_data_close":
            continue # Outcome not matured by the old-period boundary.
        records.append(row._asdict() | {"target_profit":int(trade.pnl>0),"label_exit":trade.exit_date,
            "label_return":trade.pnl/1000})
    return pd.DataFrame(records,columns=list(features.columns)+["target_profit","label_exit","label_return"])


def fit_filter(examples):
    data = examples.sort_values("confirmed_at").reset_index(drop=True)
    if len(data)<20:
        raise ValueError(f"Only {len(data)} matured examples; this experiment requires at least 20. Use a longer old period or another saved swing analysis.")
    split = int(len(data)*.75)
    validation = data.iloc[split:].copy()
    train = data.iloc[:split]
    train = train[train.label_exit < validation.confirmed_at.min()]
    if len(train)<10 or train.target_profit.nunique()<2:
        raise ValueError("Chronological training needs at least 10 matured examples and both positive/negative outcomes.")
    def estimator():
        return XGBClassifier(**PARAMETERS)
    model = estimator()
    model.fit(train[FEATURES],train.target_profit.astype(int))
    validation["probability"] = model.predict_proba(validation[FEATURES])[:,list(model.classes_).index(1)]
    prior = float(train.target_profit.mean())
    scores = []
    for name, probabilities in [("xgboost",validation.probability), ("training_up_fraction",np.full(len(validation),prior))]:
        scores.append({"model":name,"samples":len(validation),
            "accuracy":accuracy_score(validation.target_profit,np.asarray(probabilities)>=.5),
            "brier":brier_score_loss(validation.target_profit,probabilities)})
    final_model = estimator()
    final_model.fit(data[FEATURES],data.target_profit.astype(int))
    audit = {"old_examples":len(data),"training_examples":len(train),"validation_examples":len(validation),
        "purged_examples":split-len(train),"training_max_label_exit":str(train.label_exit.max()),
        "validation_start":str(validation.confirmed_at.min()),"final_max_label_exit":str(data.label_exit.max())}
    return final_model,validation,pd.DataFrame(scores),audit
