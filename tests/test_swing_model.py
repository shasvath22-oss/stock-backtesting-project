import numpy as np
import pandas as pd
import pytest

from trend_analyzer.swing_model import FEATURES,signal_features,label_old_signals,fit_filter


def test_signal_inputs_do_not_change_with_future_prices():
    dates = pd.bdate_range("2020-01-01",periods=65)
    close = 100+np.sin(np.arange(65))
    bars = pd.DataFrame({"date":dates,"open":close,"high":close+1,"low":close-1,"close":close,"volume":1000.,"dividend":0.})
    events = pd.DataFrame([
        {"pivot_date":dates[30],"confirmed_at":dates[32],"segment":0,"kind":"high","price":120.,"label":"HH","status":"accepted","swing_id":0},
        {"pivot_date":dates[38],"confirmed_at":dates[40],"segment":0,"kind":"low","price":90.,"label":"HL","status":"accepted","swing_id":1}])
    before = signal_features(bars,events)
    altered = bars.copy()
    altered.loc[41:,["open","high","low","close","volume"]] *= 3
    pd.testing.assert_frame_equal(before,signal_features(altered,events))
    assert len(before) == 1
    assert "target_profit" not in FEATURES


def test_training_label_uses_costs_and_excludes_unfinished_boundary():
    dates = pd.bdate_range("2020-01-01",periods=6)
    bars = pd.DataFrame({"date":dates,"open":100.,"high":101.,"low":99.,"close":100.,"dividend":0.})
    bars.loc[2,"high"] = 111
    features = pd.DataFrame([{**{k:.1 for k in FEATURES},"confirmed_at":d,"stop":95.,"target":110.,"combination":"HH → HL"} for d in [dates[0],dates[4]]])
    labeled = label_old_signals(bars,features,dates,20)
    assert len(labeled) == 1
    assert labeled.iloc[0].target_profit == 1
    assert labeled.iloc[0].label_exit == dates[2]
    assert labeled.iloc[0].label_return == pytest.approx(110*.999/(100*1.001)-1)


def test_chronological_validation_purges_unmatured_training_answers():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01",periods=50)
    examples = pd.DataFrame(rng.normal(size=(40,len(FEATURES))),columns=FEATURES)
    examples["confirmed_at"] = dates[:40]
    examples["label_exit"] = dates[5:45]
    examples["target_profit"] = [i%2 for i in range(40)]
    model,validation,scores,audit = fit_filter(examples)
    assert audit["purged_examples"] == 5
    assert pd.Timestamp(audit["training_max_label_exit"]) < validation.confirmed_at.min()
    assert len(validation) == 10
    assert set(scores.model) == {"xgboost","training_up_fraction"}
    assert validation.probability.between(0,1).all()
    assert list(model.feature_names_in_) == FEATURES
    with pytest.raises(ValueError,match="at least 20"):
        fit_filter(examples.head(10))
