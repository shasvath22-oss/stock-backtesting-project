"""Full multi-instrument research pipeline, explicit stage execution."""
from datetime import datetime
from pathlib import Path
import sqlite3
import re
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from trend_analyzer.__main__ import execute
from trend_analyzer.common import create_run,manifest,read_json
from trend_analyzer.period_ui import show_period_controls
from trend_analyzer.periods import period_caption
from trend_analyzer.verification import verify_run


st.set_page_config(page_title="Market research pipeline",layout="wide",initial_sidebar_state="expanded")
st.title("Stock research")
PAGES = {1:"1. Collect + SQL",2:"2. Features",3:"3. Validation splits",4:"4. Models + predictions",5:"5. Trading backtest",6:"6. Risk + report"}
with st.sidebar:
    st.header("Research steps")
    page = st.radio("Research step",list(PAGES),format_func=PAGES.get,key="research_page")
    st.caption("Validate date splits before fitting models. Each Run button executes only that stage.")
periods = show_period_controls()
st.caption("New-run defaults: " + period_caption(periods))
st.write("Enter a stock ticker → collect prices → train XGBoost → test predictions → download statistics. SPY supplies market context and a buy-and-hold comparison.")


def stage_button(stage,run,label):
    prerequisites = {"collect":None,"features":"collect","splits":"features","train":"splits","predict":"train","backtest":"predict","report":"backtest"}
    needed = prerequisites[stage]
    ready = needed is None or needed in manifest(run)["stages"]
    if not ready:
        st.info(f"Complete {needed} first.")
    if st.button(label,disabled=not ready,key=f"research_{stage}_{run.name}"):
        try:
            with st.status(f"Running {stage}...",expanded=True) as status:
                execute(stage,run)
                status.update(label=f"{stage} complete",state="complete")
            st.success("Saved. Inspect the output before selecting the next step.")
        except Exception as exc:
            st.error(str(exc))
    return stage in manifest(run)["stages"]


if page == 1:
    st.subheader("Step 1 — Collect and store prices")
    st.write("Choose one stock or ETF ticker, such as AAPL, MSFT, or QQQ. The analysis trains and tests on that ticker. SPY is downloaded separately for market context and comparison. At least 1,260 valid sessions are required.")
    st.write("")
    with st.form("research_create"):
        st.caption("Data source: Yahoo Finance via yfinance only. No generated/demo data.")
        ticker = st.text_input("Stock ticker", value="AAPL", help="Enter one Yahoo Finance ticker, for example AAPL, MSFT, QQQ, or BRK-B.")
        start = st.date_input("Collection start",value=datetime(2015,1,1).date())
        end = st.date_input("Collection end (excluded)",value=datetime.now(ZoneInfo("America/New_York")).date())
        capital = st.number_input("Portfolio capital",min_value=100.,value=100000.,step=1000.)
        create = st.form_submit_button("Create research experiment")
        run_all = st.form_submit_button("Run complete analysis for this ticker", type="primary")
    if create or run_all:
        try:
            ticker = ticker.strip().upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ticker):
                raise ValueError("Enter one valid stock ticker, such as AAPL or MSFT.")
            if (end-start).days < 365*5:
                raise ValueError("Request at least five years of historical prices.")
            run = create_run(demo=False,full=False,overrides={"symbols":[ticker],"ticker_limit":1,"start":str(start),"end":str(end),"capital":capital,"research_periods":periods})
            st.session_state["research_run"] = run.name
            if run_all:
                with st.status(f"Running complete analysis for {ticker}...", expanded=True) as status:
                    for stage in ["collect", "features", "splits", "train", "predict", "backtest", "report"]:
                        st.write(f"Running {stage}...")
                        execute(stage, run)
                        st.write(f"Completed {stage}.")
                    status.update(label=f"Analysis complete for {ticker}", state="complete")
                st.success(f"Results saved for {ticker}. Open submission metrics and select experiment {run.name} to download the statistics.")
            else:
                st.success(f"Experiment created for {ticker}. Run collection below, then follow the steps.")
        except Exception as exc:
            st.error(str(exc))
runs = sorted([p.name for p in Path("artifacts").glob("*") if (p/"manifest.json").exists()
    and manifest(p).get("workflow")=="multi_horizon_research" and not manifest(p).get("learning") and not manifest(p).get("synthetic",True)],reverse=True)
if not runs:
    st.info("Select Step 1 and create a research experiment first.")
    st.stop()
if st.session_state.get("research_run") not in runs:
    st.session_state.pop("research_run",None)
run = Path("artifacts") / st.sidebar.selectbox("Research experiment",runs,key="research_run")
m = manifest(run)
cfg = m["config"]
st.sidebar.write(f"{', '.join(cfg['tickers'])} · {cfg['start']} → {cfg['end']}")
st.sidebar.dataframe(pd.DataFrame({"stage":["collect","features","splits","train","predict","backtest","report"],
    "complete":[s in m["stages"] for s in ["collect","features","splits","train","predict","backtest","report"]]}),hide_index=True)
if m["synthetic"]:
    st.warning("SYNTHETIC DEMO — generated prices, 3 instruments, reduced history requirements. Demo settings use 2018–2022 development and 2023 final test.")
with st.expander("Saved experiment settings"):
    st.json(cfg)
st.caption("Sidebar dates are defaults for NEW experiments; this experiment uses its saved settings. Rerunning an upstream stage invalidates later completion flags. Filters never download or train.")

if page == 1:
    complete = stage_button("collect",run,"Run step 1: Collect prices + SQLite")
    if (run/"quality.json").exists():
        st.dataframe(pd.read_csv(run/"coverage.csv"),hide_index=True)
        st.json(read_json(run/"quality.json"))
    if complete:
        with sqlite3.connect("data/market.sqlite") as connection:
            sample = pd.read_sql_query("SELECT ticker,date,open,high,low,close,volume FROM price_bars WHERE snapshot_id=? ORDER BY ticker,date LIMIT 20",connection,params=[run.name])
        st.write("Stored SQL rows (sample)")
        st.dataframe(sample,hide_index=True)
        st.caption(f"Raw and cleaned Parquet: {run}; SQL: data/market.sqlite. Select Step 2 next.")
elif page == 2:
    st.subheader("Step 2 — Trailing features + future answers")
    st.write("Features: dividend-inclusive 1/5/21/63-session returns, completed calendar-week/month returns, RSI, MACD, SMA/EMA distances/slopes, ATR, volatility, volume and engulfing/doji/hammer rules. Warmup and missing windows are ineligible. Features look backward only; future entry/exit/target fields are never input columns.")
    st.write("")
    if stage_button("features",run,"Run step 2: Build features + targets"):
        schema = read_json(run/"feature_schema.json")
        st.json(schema)
        data = pd.read_parquet(run/"features.parquet")
        st.dataframe(data.dropna(subset=schema["columns"]).tail(20),hide_index=True)
elif page == 3:
    st.subheader("Step 3 — Train / validation / final test")
    st.write("All tickers share date boundaries. Initial history trains the first model; later development windows are out-of-sample validation. New-period windows are the final test. Training expands before each window, including only labels whose exits were known by the previous close. Scheduled test retraining may use matured earlier test labels; there is no test-driven tuning.")
    st.write("")
    if stage_button("splits",run,"Run step 3: Save chronological splits"):
        splits = read_json(run/"splits.json")
        st.json(splits["partitions"])
        st.dataframe(pd.DataFrame(splits["folds"]),hide_index=True)
elif page == 4:
    st.subheader("Step 4 — Fit models, then predict")
    st.write("Separate pooled classifiers for 1-session daily, 5-session weekly and 21-session monthly horizons. Compare XGBoost with always-up, majority, momentum and logistic regression. Hyperparameters and 55% trading threshold are fixed before evaluation. Accuracy uses 50%; overlapping labels are dependent.")
    st.write("")
    stage_button("train",run,"Run step 4A: Train all horizons")
    if stage_button("predict",run,"Run step 4B: Generate probabilities + scores"):
        accuracy = pd.read_csv(run/"accuracy.csv")
        phase = st.selectbox("Evaluation period",["test","development"],format_func=lambda s:"Validation" if s=="development" else "Final test")
        horizon = st.selectbox("Prediction horizon",[1,5,21])
        st.dataframe(accuracy[accuracy.phase.eq(phase)&accuracy.horizon.eq(horizon)],hide_index=True)
        predictions = pd.read_parquet(run/"predictions.parquet")
        st.dataframe(predictions[predictions.phase.eq(phase)&predictions.horizon.eq(horizon)&predictions.model.eq("xgboost")].tail(25),hide_index=True)
elif page in (5,6):
    st.subheader("Step 5 — Trading backtest" if page==5 else "Step 6 — Risk, costs and report")
    st.write("Signals observed after close execute at the next open. Hold for the selected 1/5/21-session horizon, long/cash, equal initial capital sleeves, one position per instrument, fractional normalized units, no leverage/commission/cash interest. Eligible dividends included. Missing execution prices fail instead of inventing fills. Compare SPY and equal-weight buy-and-hold on identical dates.")
    st.write("10 bps per side baseline; identical predictions at 0/5/10/20 bps. CAGR uses elapsed time; Sharpe uses daily returns, 252 sessions, zero risk-free rate. Drawdown includes initial execution costs.")
    ready = stage_button("backtest",run,"Run step 5: Backtest all horizons + costs") if page==5 else "backtest" in manifest(run)["stages"]
    if ready:
        perf = pd.read_csv(run/"performance.csv")
        phase = st.selectbox("Portfolio period",["test","development"])
        horizon = st.selectbox("Portfolio horizon",[1,5,21])
        cost = st.selectbox("Slippage bps per side",[0,5,10,20],index=2)
        subset = perf[perf.phase.eq(phase)&perf.bps.eq(cost)&(perf.horizon.eq(horizon)|perf.horizon.eq(0))]
        st.dataframe(subset,hide_index=True)
        main = subset[subset.model.eq("xgboost")]
        if len(main):
            result = main.iloc[0]
            a,b,c = st.columns(3)
            a.metric("Net profit/loss",f"${result.net_return*cfg['capital']:+,.2f}")
            b.metric("CAGR",f"{result.cagr:.2%}")
            c.metric("Sharpe",f"{result.sharpe:.2f}")
            curves = [pd.read_parquet(run/"backtests"/f"{row.artifact}_equity.parquet").set_index("date").equity.rename(row.model) for row in subset.itertuples()]
            st.line_chart(pd.concat(curves,axis=1),y_label="Portfolio value")
            kind = st.selectbox("Execution ledger",["trades","positions"])
            ledger = pd.read_parquet(run/"backtests"/f"{result.artifact}_{kind}.parquet")
            st.dataframe(ledger.tail(100),hide_index=True)
            st.download_button("Download ledger CSV",ledger.to_csv(index=False),file_name=f"{result.artifact}_{kind}.csv",mime="text/csv")
        if page == 6 and stage_button("report",run,"Run step 6: Generate HTML report"):
            st.download_button("Download HTML report",(run/"report.html").read_bytes(),file_name=f"{run.name}_report.html",mime="text/html")
            if st.button("Verify saved research outputs",key=f"verify_{run.name}"):
                checks = verify_run(run)
                st.json(checks)
                if checks["passed"]:
                    st.success("Saved-output integrity checks passed. This verifies engineering consistency, not market profitability.")
                else:
                    st.error("One or more verification checks failed; inspect the details.")
    else:
        st.info("Complete the Step 5 backtest first.")
