import pandas as pd
import pytest
from pathlib import Path
from datetime import date
from streamlit.testing.v1 import AppTest

from trend_analyzer.periods import period_settings, period_windows, validate_periods
from trend_analyzer.structure_stats import structure_outcomes
from trend_analyzer.swing_backtest import simulate_rule


def test_project_windows_are_disjoint_and_have_requested_boundaries():
    dates = pd.to_datetime(["2019-12-31","2020-01-01","2024-12-31","2025-01-02","2026-10-23"])
    windows = period_windows(dates)
    assert list(windows["old"]) == list(dates[1:3])
    assert list(windows["new"]) == list(dates[3:])
    assert windows["old"].intersection(windows["new"]).empty
    assert period_settings()["new_start"] == "2025-01-01"


def test_custom_starts_ends_and_gap():
    cfg = {"old_start":"2016-01-01","old_end":"2018-12-31","new_start":"2020-01-01","new_end":"2021-12-31"}
    dates = pd.to_datetime(["2015-01-01","2016-01-01","2018-12-31","2019-01-01","2020-01-01","2021-12-31","2022-01-01"])
    windows = period_windows(dates,cfg)
    assert list(windows["old"]) == list(dates[1:3])
    assert list(windows["new"]) == list(dates[4:6])
    with pytest.raises(ValueError,match="overlap"):
        validate_periods(cfg | {"new_start":"2018-01-01"})


def test_sidebar_period_changes_apply_without_running_pipeline(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"streamlit_app.py")).run()
    for label,value in [("Old start",date(2016,1,1)),("Old end",date(2020,12,31)),("New start",date(2021,1,1)),("New end",date(2024,12,31))]:
        next(w for w in app.sidebar.date_input if w.label.startswith(label)).set_value(value)
    app.sidebar.checkbox[0].uncheck()
    next(w for w in app.sidebar.button if w.label.startswith("Apply periods")).click().run()
    assert not app.exception
    assert app.session_state["active_periods"]["new_end"] == "2024-12-31"
    assert app.session_state["active_periods"]["old_start"] == "2016-01-01"
    assert not (tmp_path/"data").exists()
    next(w for w in app.sidebar.date_input if w.label.startswith("Old end")).set_value(date(2022,12,31))
    next(w for w in app.sidebar.button if w.label.startswith("Apply periods")).click().run()
    assert any("overlap" in w.value for w in app.error)
    assert app.session_state["active_periods"]["old_end"] == "2020-12-31"


def test_old_observation_cannot_use_new_year_break():
    dates = pd.to_datetime(["2024-12-27","2024-12-30","2024-12-31","2025-01-02"])
    bars = pd.DataFrame({"date":dates,"high":[120,115,115,125],"low":[115,110,111,112]})
    events = pd.DataFrame([
        {"pivot_date":dates[0],"confirmed_at":dates[0],"kind":"high","price":120.,"label":"H","segment":0,"status":"accepted","swing_id":0},
        {"pivot_date":dates[1],"confirmed_at":dates[2],"kind":"low","price":110.,"label":"HL","segment":0,"status":"accepted","swing_id":1}])
    assert structure_outcomes(bars,events,dates,20).iloc[0].outcome == "upper_first"
    old = period_windows(dates)["old"]
    assert structure_outcomes(bars,events,old,20).iloc[0].outcome == "incomplete_history"


def test_backtest_closes_at_old_boundary_and_new_starts_in_cash():
    dates = pd.to_datetime(["2024-12-27","2024-12-30","2024-12-31","2025-01-02","2025-01-03"])
    bars = pd.DataFrame({"date":dates,"open":100.,"high":101.,"low":99.,"close":100.,"dividend":0.})
    signals = pd.DataFrame({"confirmed_at":[dates[0]],"stop":[90.],"target":[120.]})
    windows = period_windows(dates)
    old_bars = bars[bars.date.isin(windows["old"])]
    _,_,trades,_ = simulate_rule(old_bars,signals)
    assert trades.iloc[0].exit_date == dates[2]
    new_bars = bars[bars.date.isin(windows["new"])]
    new_signals = signals[signals.confirmed_at.ge("2025-01-01")]
    metrics,_,trades,_ = simulate_rule(new_bars,new_signals)
    assert trades.empty
    assert metrics["final_equity"] == 10000
