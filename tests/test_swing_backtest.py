import pandas as pd
import pytest

from trend_analyzer.swing_backtest import simulate_rule


def case():
    dates = pd.bdate_range("2024-01-02",periods=5)
    bars = pd.DataFrame({"date":dates,"open":100.,"close":100.,"high":101.,"low":99.,"dividend":0.})
    signals = pd.DataFrame({"confirmed_at":[dates[0]],"stop":[95.],"target":[110.]})
    return bars,signals,dates


def test_both_touch_stop_first_on_entry_day_and_costs():
    bars,signals,dates = case()
    bars.loc[1,["high","low"]] = [111,94]
    metrics,curve,trades,audit = simulate_rule(bars,signals,1000,10)
    trade = trades.iloc[0]
    assert trade.entry_date == trade.exit_date == dates[1]
    assert trade.reason == "stop_first_if_both"
    assert trade.exit_price == pytest.approx(95*.999)
    expected = 1000/(100*1.001)*95*.999
    assert metrics["final_equity"] == pytest.approx(expected)
    assert trade.pnl == pytest.approx(expected-1000)
    assert curve.equity.iloc[0] == 1000


def test_gap_exit_and_dividend_entitlement():
    bars,signals,dates = case()
    bars.loc[1,"dividend"] = 9
    bars.loc[2,["open","high","low","close","dividend"]] = [90,91,89,90,1]
    _,_,trades,_ = simulate_rule(bars,signals,1000,0)
    trade = trades.iloc[0]
    assert trade.reason == "gap_stop"
    assert trade.exit_price == 90
    assert trade.dividends == 10
    assert trade.pnl == -90


def test_cost_cases_use_identical_timing_and_target():
    bars,signals,dates = case()
    bars.loc[2,"high"] = 111
    finals = []
    for bps in [5,10,20]:
        metrics,_,trades,_ = simulate_rule(bars,signals,1000,bps)
        assert trades.iloc[0].entry_date == dates[1]
        assert trades.iloc[0].exit_date == dates[2]
        assert trades.iloc[0].reason == "target"
        finals.append(metrics["final_equity"])
    assert finals[0] > finals[1] > finals[2]


def test_open_outside_levels_skips_and_hold_limit():
    bars,signals,dates = case()
    bars.loc[1,"open"] = 111
    _,_,trades,audit = simulate_rule(bars,signals)
    assert trades.empty
    assert audit.iloc[0].status == "skipped_open_outside_levels"
    bars,signals,dates = case()
    _,_,trades,_ = simulate_rule(bars,signals,max_hold=1)
    assert trades.iloc[0].exit_date == dates[1]
    assert trades.iloc[0].reason == "max_hold_close"
