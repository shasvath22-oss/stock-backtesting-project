"""Read-only run verification; independent checks of persisted research outputs."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .common import manifest,read_json,digest


def verify_run(run):
    run = Path(run)
    m = manifest(run)
    checks = {}
    if "collect" in m["stages"]:
        bars = pd.read_parquet(run/"bars.parquet")
        checks["unique_ticker_dates"] = not bars.duplicated(["ticker","date"]).any()
        checks["input_hash_matches"] = digest(run/"bars.parquet") == m["stages"]["collect"]["details"]["sha256"]
        with sqlite3.connect("data/market.sqlite") as connection:
            count = connection.execute("SELECT COUNT(*) FROM price_bars WHERE snapshot_id=?",[run.name]).fetchone()[0]
        checks["sql_row_count_matches"] = count == len(bars)
    if "features" in m["stages"]:
        columns = read_json(run/"feature_schema.json")["columns"]
        checks["future_columns_excluded_from_inputs"] = not bool(set(columns)&{"target","forward_return","entry_date","exit_date"})
    if "splits" in m["stages"]:
        partitions = read_json(run/"splits.json")["partitions"]
        checks["chronological_partitions"] = partitions["initial_training"]["end"] < partitions["validation"]["start"] <= partitions["validation"]["end"] < partitions["test"]["start"]
    if "predict" in m["stages"]:
        p = pd.read_parquet(run/"predictions.parquet")
        checks["all_horizons_predicted"] = set(p.horizon)==set(m["config"]["horizons"])
        checks["training_outcomes_matured"] = bool((pd.to_datetime(p.max_label_exit)<=pd.to_datetime(p.fit_cutoff)).all())
        checks["probabilities_valid"] = bool(p.probability.between(0,1).all())
        scored = p.dropna(subset=["target"])
        checks["scored_outcomes_within_phase"] = bool((scored.exit_date<=pd.to_datetime(scored.phase_end)).all())
    if "backtest" in m["stages"]:
        perf = pd.read_csv(run/"performance.csv")
        checks["all_costs_present"] = {0,5,10,20}.issubset(set(perf.bps))
        checks["spy_comparison_present"] = "SPY" in set(perf.model)
        checks["risk_metrics_finite"] = bool(np.isfinite(perf[["cagr","sharpe","max_drawdown"]]).all().all())
        details = m["stages"]["backtest"]["details"]
        checks["costs_reuse_predictions"] = details.get("predictions_sha256") == digest(run/"predictions.parquet")
        chosen = perf[perf.model.eq("xgboost")&perf.bps.eq(10)&perf.phase.eq("test")]
        if len(chosen):
            row = chosen.iloc[0]
            positions = pd.read_parquet(run/"backtests"/f"{row.artifact}_positions.parquet")
            curve = pd.read_parquet(run/"backtests"/f"{row.artifact}_equity.parquet")
            checks["sample_position_accounting"] = bool(np.allclose(positions.cash+positions.holdings_value,positions.equity))
            checks["sample_portfolio_reconciliation"] = bool(np.allclose(positions.groupby("date").equity.sum().to_numpy(),curve.equity.to_numpy()))
            spy = perf[perf.model.eq("SPY")&perf.phase.eq("test")&perf.bps.eq(10)].iloc[0]
            spy_curve = pd.read_parquet(run/"backtests"/f"{spy.artifact}_equity.parquet")
            checks["sample_spy_dates_identical"] = curve.date.equals(spy_curve.date)
    return {"run_id":run.name,"synthetic":m["synthetic"],"completed_stages":list(m["stages"]),
        "checks":{k:bool(v) for k,v in checks.items()},"passed":bool(checks) and all(checks.values())}
