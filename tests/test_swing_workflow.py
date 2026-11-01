from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from trend_analyzer.common import read_json
from trend_analyzer.swing_data import collect_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_is_one_stock_and_prices_only(tmp_path):
    run = collect_snapshot("aapl", "2023-01-01", "2024-01-01", demo=True, root=tmp_path)
    bars = pd.read_parquet(run / "bars.parquet")
    assert set(bars.ticker) == {"AAPL"}
    assert bars.date.max() < pd.Timestamp("2024-01-01")
    m = read_json(run / "manifest.json")
    assert m["completed_steps"] == ["prices"]
    assert m["sensitivity_bps_per_side"] == [5,10,20]
    assert not (run / "models").exists()
    assert not read_json(run / "quality.json")["missing_requested_sessions"]


def test_invalid_date_range_does_not_create_run(tmp_path):
    with pytest.raises(ValueError, match="Start must"):
        collect_snapshot("AAPL", "2024-01-01", "2023-01-01", root=tmp_path)
    assert not list(tmp_path.iterdir())


def test_step_one_ui_is_explicit_and_saved(tmp_path, monkeypatch, fake_yfinance):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    assert not Path("data/swing_runs").exists()
    # Every step remains visible even before its prerequisites exist.
    assert len(app.sidebar.radio[0].options) == 7
    app.sidebar.radio[0].set_value(4).run()
    assert not app.exception
    assert any("Step 1" in w.value for w in app.markdown)
    app.sidebar.radio[0].set_value(1).run()
    assert not any(w.label.startswith("Data source") for w in app.selectbox)
    next(w for w in app.button if w.label.startswith("Run step 1:")).click().run(timeout=30)
    assert not app.exception
    runs = list(Path("data/swing_runs").glob("*/manifest.json"))
    assert len(runs) == 1
    app.slider[0].set_value(30).run()
    assert not app.exception
    assert len(list(Path("data/swing_runs").glob("*/manifest.json"))) == 1
    app.sidebar.radio[0].set_value(2).run()
    next(w for w in app.button if w.label.startswith("Run step 2:")).click().run(timeout=30)
    assert not app.exception
    assert len(list(Path("data/swing_runs").glob("*/swings/*/events.parquet"))) == 1
    # Trading is a sibling of statistics, visible before statistics have run.
    assert not list(Path("data/swing_runs").glob("*/swings/*/statistics/*/outcomes.parquet"))
    app.sidebar.radio[0].set_value(4).run()
    assert not app.exception
    assert any(w.label.startswith("Run step 4:") for w in app.button)
    assert not any(w.label.startswith("Run step 3:") for w in app.button)
    app.sidebar.radio[0].set_value(2).run()
    cutoff = next(w for w in app.selectbox if w.label.startswith("Known by this close"))
    cutoff.set_value(pd.Timestamp("2020-01-02")).run()
    assert not app.exception
    app.sidebar.radio[0].set_value(3).run()
    next(w for w in app.button if w.label.startswith("Run step 3:")).click().run(timeout=30)
    assert not app.exception
    assert len(list(Path("data/swing_runs").glob("*/swings/*/statistics/*/outcomes.parquet"))) == 1
    next(w for w in app.selectbox if w.label.startswith("Example grouping")).set_value("Combination").run()
    assert not app.exception
    assert len(list(Path("data/swing_runs").glob("*/swings/*/statistics/*/outcomes.parquet"))) == 1
    app.sidebar.radio[0].set_value(4).run()
    next(w for w in app.button if w.label.startswith("Run step 4:")).click().run(timeout=30)
    assert not app.exception
    paths = list(Path("data/swing_runs").glob("*/swings/*/rule_backtests/*/performance.csv"))
    assert len(paths) == 1
    performance = pd.read_csv(paths[0])
    assert set(performance.bps) == {5,10,20}
    assert set(performance.model) == {"HH_HL","buy_hold"}
    assert any(w.value.startswith("Step 4A: 2020-01-01 to 2024-12-31") for w in app.subheader)
    costs = [w for w in app.selectbox if w.label.startswith("Execution cost per side")]
    assert len(costs) == 1
    assert [w.value for w in costs] == [10]
    app.sidebar.radio[0].set_value("4b").run()
    assert not app.exception
    assert any(w.value.startswith("Step 4B: 2025-01-01") for w in app.subheader)
    costs = [w for w in app.selectbox if w.label.startswith("Execution cost per side")]
    costs[0].set_value(20).run()
    assert not app.exception
    assert [w.value for w in app.selectbox if w.label.startswith("Execution cost per side")] == [20]
    assert len(list(Path("data/swing_runs").glob("*/swings/*/rule_backtests/*/performance.csv"))) == 1
    app.sidebar.radio[0].set_value(5).run()
    assert not app.exception
    assert any(w.label.startswith("Run step 5:") for w in app.button)
    assert not list(Path("data/swing_runs").glob("*/swings/*/model_runs/*/model.joblib"))
    app.sidebar.radio[0].set_value(6).run()
    assert not app.exception
    assert any("Run Step 5 first" in w.value for w in app.info)
