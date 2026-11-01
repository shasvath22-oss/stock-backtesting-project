from pathlib import Path
import sqlite3

import pandas as pd
from streamlit.testing.v1 import AppTest

from trend_analyzer.__main__ import execute
from trend_analyzer.common import create_run,manifest
from trend_analyzer.data import collect,sessions,synthetic_bars


ROOT = Path(__file__).resolve().parents[1]


def prepare(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"config").mkdir()
    for name in ["research.json","universe.csv"]:
        (tmp_path/"config"/name).write_bytes((ROOT/"config"/name).read_bytes())


def test_50_requested_symbols_sql_keys_and_vendor_actions(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    import yfinance as yf
    run = create_run(full=True,overrides={"start":"2016-01-01","end":"2022-01-01"})
    calendar = sessions("2016-01-01","2021-12-31")
    raw = synthetic_bars(["fixture"],calendar,42).set_index("date")
    raw.loc[calendar[500],"split"] = 2.
    raw.loc[calendar[501],"dividend"] = 1.25
    vendor = raw.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","adj_close":"Adj Close","volume":"Volume","split":"Stock Splits","dividend":"Dividends"})
    called = []
    class Ticker:
        def __init__(self,ticker):
            called.append(ticker)
        def history(self,**kwargs):
            assert kwargs["auto_adjust"] is False
            assert kwargs["actions"] is True
            return vendor.copy()
    monkeypatch.setattr(yf,"Ticker",Ticker)
    monkeypatch.setattr(yf,"set_tz_cache_location",lambda path:None)
    collect(run)
    assert len(manifest(run)["config"]["tickers"]) == 50
    assert len(set(called)) == 51 # 50 requested instruments + SPY
    with sqlite3.connect("data/market.sqlite") as connection:
        count = connection.execute("SELECT COUNT(*) FROM price_bars WHERE snapshot_id=?",[run.name]).fetchone()[0]
        assert count == 51*len(calendar)
        keys = connection.execute("PRAGMA table_info(price_bars)").fetchall()
        assert [r[1] for r in keys if r[5]] == ["snapshot_id","ticker","date"]
        assert connection.execute("SELECT split,dividend FROM price_bars WHERE snapshot_id=? AND ticker='AAPL' AND date=?",[run.name,str(calendar[500].date())]).fetchone()[0] == 2
    coverage = pd.read_csv(run/"coverage.csv")
    assert coverage.eligible.all()
    assert (run/"raw_bars.parquet").exists()
    # A retry replaces this snapshot only rather than duplicating ticker/date rows.
    collect(run)
    with sqlite3.connect("data/market.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM price_bars").fetchone()[0] == count


def test_complete_multi_horizon_demo_and_all_research_pages(tmp_path,monkeypatch,fake_yfinance):
    prepare(tmp_path,monkeypatch)
    run = create_run(demo=False,overrides={"ticker_limit":3,"start":"2018-01-01","end":"2024-01-01",
        "minimum_sessions":500,"initial_train_sessions":504,"fold_sessions":126,
        "xgboost":{"n_estimators":40,"max_depth":3,"learning_rate":.05,"subsample":.8,"colsample_bytree":.8},
        "research_periods":{"old_start":"2018-01-01","old_end":"2022-12-31","new_start":"2023-01-01","new_end":None}})
    for stage in ["collect","features","splits","train","predict","backtest","report"]:
        execute(stage,run)
    from trend_analyzer.common import read_json
    splits = read_json(run/"splits.json")
    assert pd.Timestamp(splits["partitions"]["initial_training"]["end"]) < pd.Timestamp(splits["partitions"]["validation"]["start"])
    assert pd.Timestamp(splits["partitions"]["validation"]["end"]) < pd.Timestamp(splits["partitions"]["test"]["start"])
    predictions = pd.read_parquet(run/"predictions.parquet")
    assert set(predictions.horizon) == {1,5,21}
    assert (pd.to_datetime(predictions.max_label_exit) <= pd.to_datetime(predictions.fit_cutoff)).all()
    labeled = predictions.dropna(subset=["target"])
    assert (labeled.exit_date <= pd.to_datetime(labeled.phase_end)).all()
    features = read_json(run/"feature_schema.json")["columns"]
    assert {"rsi","macd","doji","hammer","bull_engulfing","bear_engulfing","return_1","return_5","return_21","completed_week","completed_month"}.issubset(features)
    assert not set(features)&{"target","forward_return","exit_date","entry_date"}
    performance = pd.read_csv(run/"performance.csv")
    assert set(performance.bps) == {0,5,10,20}
    assert {"SPY","equal_weight","xgboost"}.issubset(set(performance.model))
    assert {"cagr","sharpe","max_drawdown"}.issubset(performance.columns)
    assert (run/"report.html").exists()
    from trend_analyzer.verification import verify_run
    checks = verify_run(run)
    assert checks["passed"], checks
    app = AppTest.from_file(str(ROOT/"research_app.py")).run(timeout=30)
    for page in range(1,7):
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, page


def test_research_ui_create_does_not_download_or_train(tmp_path,monkeypatch):
    prepare(tmp_path,monkeypatch)
    app = AppTest.from_file(str(ROOT/"research_app.py")).run()
    assert not app.exception
    assert len(app.sidebar.radio[0].options) == 6
    assert not any("synthetic" in w.label.lower() for w in app.checkbox)
    next(w for w in app.button if w.label=="Create research experiment").click().run()
    assert not app.exception
    assert not Path("data/market.sqlite").exists()
    run = next(Path("artifacts").glob("*"))
    assert not manifest(run)["stages"]
    assert manifest(run)["synthetic"] is False
    assert manifest(run)["config"]["tickers"] == ["SPY","QQQ"]
    assert len(app.sidebar.radio[0].options) == 6


def test_fold_training_excludes_date_gaps_and_unmatured_labels():
    from trend_analyzer.modeling import fold_masks
    frame = pd.DataFrame({"date":pd.to_datetime(["2019-01-01","2020-01-01","2024-01-01","2025-02-01","2025-03-01"]),
        "exit_date":pd.to_datetime(["2019-02-01","2020-02-01","2024-02-01","2025-02-10","2025-04-01"]),"target":1})
    fold = {"train_start":"2020-01-01","old_end":"2023-12-31","new_start":"2025-01-01",
        "fit_cutoff":"2025-03-10","start":"2025-03-11","end":"2025-04-01"}
    training,_ = fold_masks(frame,fold)
    assert list(training) == [False,True,False,True,False]


def test_synthetic_research_runs_are_hidden_and_cli_rejects_them(tmp_path,monkeypatch,capsys):
    import sys
    import pytest
    from trend_analyzer.__main__ import main
    prepare(tmp_path,monkeypatch)
    run = create_run(demo=True)
    app = AppTest.from_file(str(ROOT/"research_app.py")).run()
    assert not app.exception
    assert not any(w.label=="Research experiment" for w in app.selectbox)
    monkeypatch.setattr(sys,"argv",["trend_analyzer","features","--run-id",run.name])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Only yfinance historical" in capsys.readouterr().err
