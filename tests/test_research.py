import numpy as np
import pandas as pd
import pytest

from trend_analyzer.backtest import simulate
from trend_analyzer.data import synthetic_bars, validate_bars
from trend_analyzer.features import build_dataset, indicators
from trend_analyzer.modeling import fold_masks
from trend_analyzer.common import begin_stage, manifest, write_json


def test_features_are_causal():
    dates = pd.bdate_range("2020-01-01", periods=350)
    bars = synthetic_bars(["A"],dates,7).set_index("date")
    before = indicators(bars)
    altered = bars.copy()
    altered.loc[dates[280]:,["open","high","low","close"]] *= 2
    pd.testing.assert_frame_equal(before.loc[:dates[279]],indicators(altered).loc[:dates[279]])


def test_target_alignment_missing_windows_and_unknown_tail():
    dates = pd.bdate_range("2020-01-01",periods=350)
    bars = synthetic_bars(["A","SPY"],dates,4)
    bars.loc[(bars.ticker=="A")&(bars.date==dates[252]),"dividend"] = 1.25
    cfg = {"benchmark":"SPY","horizons":[1,5,21]}
    dataset, columns = build_dataset(bars,dates,cfg,["A"])
    for h in cfg["horizons"]:
        row = dataset[(dataset.date==dates[250])&(dataset.horizon==h)].iloc[0]
        a = bars[bars.ticker.eq("A")].set_index("date")
        expected = (a.loc[dates[251+h],"open"]+a.loc[dates[252]:dates[251+h],"dividend"].sum())/a.loc[dates[251],"open"]-1
        assert row.forward_return == pytest.approx(expected)
        assert row.entry_date==dates[251] and row.exit_date==dates[251+h]
        assert dataset[dataset.horizon.eq(h)].tail(h+1).target.isna().all()
    broken = bars[~((bars.ticker=="A")&(bars.date==dates[260]))]
    bad, _ = build_dataset(broken,dates,cfg,["A"])
    assert bad[(bad.date==dates[259])&(bad.horizon==5)].target.isna().all()
    assert bad[bad.date.eq(dates[270])][columns].isna().all().all()


def test_purging_is_by_label_exit_and_shared_dates():
    df = pd.DataFrame({"date":pd.to_datetime(["2020-01-01","2020-01-01","2020-01-02","2020-01-03"]),
        "exit_date":pd.to_datetime(["2020-01-02","2020-01-06","2020-01-03","2020-01-07"]),"target":[1,0,1,0]})
    train, evaluation = fold_masks(df,{"fit_cutoff":"2020-01-02","start":"2020-01-03","end":"2020-01-10"})
    assert train.tolist()==[True,False,False,False]
    assert evaluation.tolist()==[False,False,False,True]


def market_fixture():
    dates = pd.bdate_range("2024-01-02",periods=5)
    bars = pd.DataFrame({"date":dates,"ticker":"A","open":100.,"close":100.,"dividend":0.})
    signal = pd.DataFrame({"ticker":["A"],"date":[dates[0]-pd.Timedelta(days=1)],
        "entry_date":[dates[0]],"exit_date":[dates[-1]],"probability":[.9]})
    return dates,bars,signal


def test_slippage_roundtrip_and_reconciliation():
    dates,bars,signal = market_fixture()
    curve,trades,positions = simulate(bars,signal,dates,["A"],1000,10,.55)
    assert curve.equity.iloc[-1]==pytest.approx(1000*0.999/1.001)
    assert trades.executed_price.tolist()==[100.1,99.9]
    assert len(trades)==2
    assert np.allclose(positions.cash+positions.holdings_value,positions.equity)
    assert positions.cash.min()>=-1e-8
    assert positions.normalized_units.iloc[-1]==0


def test_distributions_entitlement_and_no_double_split():
    dates,bars,signal = market_fixture()
    bars["split"] = [0.,0.,2.,0.,0.]
    bars.loc[0,"dividend"] = 2.  # Not entitled on entry ex-date.
    bars.loc[2,"dividend"] = 1.
    bars.loc[4,"dividend"] = 1.  # Entitled on exit ex-date.
    curve,trades,_ = simulate(bars,signal,dates,["A"],1000,0,.55)
    assert curve.equity.iloc[-1]==pytest.approx(1020)
    assert trades.normalized_units.tolist()==[10.,10.]
    assert trades.realized_pnl.iloc[-1]==pytest.approx(20)


def test_missing_mark_fails_instead_of_inventing_execution():
    dates,bars,signal = market_fixture()
    bars.loc[2,"open"] = np.nan
    with pytest.raises(ValueError,match="price gap"):
        simulate(bars,signal,dates,["A"],1000,0,.55)


def test_quality_rejects_duplicates_and_bad_bars():
    dates = pd.bdate_range("2020-01-01",periods=5)
    bars = synthetic_bars(["A"],dates,1)
    bars.loc[2,"high"] = 0
    bars = pd.concat([bars,bars.iloc[[0]]],ignore_index=True)
    clean,issues = validate_bars(bars,dates)
    assert len(clean)==3
    assert len(issues)>=3


def test_upstream_rerun_invalidates_stale_results(tmp_path):
    write_json(tmp_path/"manifest.json",{"config":{},"stages":dict.fromkeys(
        ["collect","features","splits","train","train_h1","predict","backtest","report"],{})})
    (tmp_path/"performance.csv").write_text("old")
    (tmp_path/"report.html").write_text("old")
    begin_stage(tmp_path,"features")
    assert list(manifest(tmp_path)["stages"])==["collect"]
    assert not (tmp_path/"performance.csv").exists()
    assert not (tmp_path/"report.html").exists()


def test_overlapping_signals_do_not_overallocate_capital():
    dates,bars,signal = market_fixture()
    duplicate = signal.copy()
    duplicate["date"], duplicate["entry_date"] = dates[0], dates[1]
    signals = pd.concat([signal,duplicate],ignore_index=True)
    curve,trades,positions = simulate(bars,signals,dates,["A"],1000,0,.55)
    assert trades.side.tolist()==["buy","sell"]
    assert positions.normalized_units.max()==10
