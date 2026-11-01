import numpy as np
import pandas as pd

from trend_analyzer.model_audit import audit_candidates
from trend_analyzer.swing_backtest import simulate_rule


def fixture():
    dates = pd.bdate_range("2025-01-02",periods=8)
    bars = pd.DataFrame({"date":dates,"open":100.,"high":101.,"low":99.,"close":100.,"dividend":0.})
    bars.loc[2,"high"] = 111
    bars.loc[5,"low"] = 94
    candidates = pd.DataFrame({"confirmed_at":[dates[0],dates[3]],"stop":[95.,95.],"target":[110.,110.]})
    predictions = pd.DataFrame({"confirmed_at":[dates[0],dates[3]],"probability":[.4,.7],"selected":[False,True]})
    _,_,model,entries = simulate_rule(bars,candidates.iloc[1:],1000,10)
    _,_,baseline,_ = simulate_rule(bars,candidates,1000,10)
    return bars,candidates,predictions,model,baseline,entries


def test_rejected_profit_and_selected_loss_do_not_change_predictions():
    bars,candidates,predictions,model,baseline,entries = fixture()
    before = predictions.copy(deep=True)
    result = audit_candidates(bars,candidates,predictions,model,baseline,entries,1000,10)
    assert list(result.decision) == ["rejected","selected"]
    assert list(result.outcome) == ["profit","loss"]
    assert pd.isna(result.iloc[0].actual_model_pnl)
    assert result.iloc[1].actual_model_pnl == model.iloc[0].pnl
    assert result.iloc[1].model_execution == "executed"
    pd.testing.assert_frame_equal(predictions,before)


def test_missing_inputs_and_unmatured_outcomes_remain_separate():
    bars,candidates,predictions,model,baseline,entries = fixture()
    candidates = pd.DataFrame({"confirmed_at":[bars.date.iloc[5]],"stop":[90.],"target":[120.]})
    result = audit_candidates(bars,candidates,predictions.iloc[:0],model,baseline,entries,1000,10)
    assert result.iloc[0].decision == "missing_inputs"
    assert result.iloc[0].outcome == "incomplete_history"
    assert np.isnan(result.iloc[0].hypothetical_pnl)
