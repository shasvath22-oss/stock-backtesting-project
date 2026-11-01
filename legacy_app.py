"""Run with: python -m streamlit run streamlit_app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from trend_analyzer.common import create_run, manifest, read_json
from trend_analyzer.__main__ import execute
from trend_analyzer.learning import show_learning, show_setup

st.set_page_config(page_title="Trend analyzer",page_icon=":material/monitoring:",layout="wide")
st.title("Trend analyzer")
st.caption("Multi-horizon direction models · chronological evaluation · trading costs")


@st.cache_data(ttl=60,max_entries=60)
def table(path, modified):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def load(path):
    stages_for_file = {"coverage.csv":"collect","bars.parquet":"collect","accuracy.csv":"predict",
                       "accuracy_detail.csv":"predict","performance.csv":"backtest"}
    needed = stages_for_file.get(path.name,"backtest" if path.parent.name=="backtests" else None)
    if needed and needed not in manifest(run)["stages"]:
        if needed=="collect":
            st.warning("Collection is incomplete. These saved inputs are provisional; inspect download failures below.")
        else:
            st.info(f"Run {needed} to refresh these results after an upstream change.")
            st.stop()
    return table(str(path),path.stat().st_mtime_ns)


setup_area = st.container()
with st.sidebar:
    st.subheader("Research runs")
    st.caption("")
    with st.form("new_run"):
        learning = st.checkbox("Learn with one stock (AAPL)",value=True)
        demo = st.checkbox("Synthetic demo",value=True,help="Generated prices; not market performance.")
        full = st.checkbox("All 50 instruments",value=False,disabled=demo)
        new = st.form_submit_button("Create run",icon=":material/add:")
    if new:
        run = create_run(demo=demo,full=full,learning=learning)
        st.session_state["run_choice"] = run.name
        st.session_state["view_choice"] = "Learning walkthrough" if learning else "Overview"
    runs = sorted([p.name for p in Path("artifacts").glob("*") if (p/"manifest.json").exists()],reverse=True)
    if not runs:
        st.info("Create a run to get started.")
        with setup_area:
            show_setup()
        st.stop()
    run_id = st.selectbox("Run",runs,key="run_choice")
    run = Path("artifacts")/run_id
    m = manifest(run)
    if m["synthetic"]:
        st.warning("Synthetic demo data")
    else:
        st.info("Historical market data")
    view = st.selectbox("View",["Learning walkthrough","Overview","Data quality","Model evaluation","Portfolio","Trades and positions","Pipeline"],key="view_choice")

if m["synthetic"]:
    st.warning("SYNTHETIC DEMO — generated prices for checking the pipeline and UI. These results do not show market performance.")

if view=="Learning walkthrough":
    show_learning(run)

elif view=="Pipeline":
    st.subheader("Run the pipeline")
    st.write("Each step saves its output. Downloads use the network; training and simulations can take several minutes.")
    stage_names = ["collect","features","splits","train","predict","backtest","report"]
    prerequisites = {"collect":None,"features":"collect","splits":"features","train":"splits",
                     "predict":"train","backtest":"predict","report":"backtest"}
    with st.form("execute"):
        stage = st.selectbox("Step",["All remaining steps"]+stage_names)
        submit = st.form_submit_button("Run selected step",type="primary",icon=":material/play_arrow:")
    if submit:
        selected = [s for s in stage_names if s not in m["stages"]] if stage=="All remaining steps" else [stage]
        try:
            with st.status("Running pipeline",expanded=True) as status:
                for selected_stage in selected:
                    prereq = prerequisites[selected_stage]
                    if prereq and prereq not in manifest(run)["stages"]:
                        raise ValueError(f"Complete {prereq} first")
                    st.write(f"Running {selected_stage}…")
                    execute(selected_stage,run)
                    st.write(f"Completed {selected_stage}")
                status.update(label="Pipeline complete",state="complete")
            st.cache_data.clear()
            st.success("Results saved. Select a results view in the sidebar.")
        except Exception as exc:
            st.error(str(exc))
    st.dataframe(pd.DataFrame([{"step":s,"status":"Complete" if s in manifest(run)["stages"] else "Pending"} for s in stage_names]),hide_index=True)
    with st.expander("Configuration"):
        st.json(m["config"])

elif view=="Overview":
    st.subheader("Research overview")
    cfg = m["config"]
    with st.container(horizontal=True):
        st.metric("Instruments",len(cfg["tickers"]),border=True)
        st.metric("Horizons"," / ".join(map(str,cfg["horizons"])),border=True)
        st.metric("Initial capital",f"${cfg['capital']:,.0f}",border=True)
        st.metric("Completed stages",len([s for s in m["stages"] if not s.startswith("train_h")]),border=True)
    st.write("Features are observed after the close. Orders execute at the following open. Models predict 1, 5, and 21-session open-to-open direction.")
    st.write("Portfolios use equal capital sleeves, long/cash positions, no leverage, and no overlapping position in the same instrument.")
    if (run/"performance.csv").exists():
        p = load(run/"performance.csv")
        st.subheader("Final test at 10 bps per side")
        st.dataframe(p[p.phase.eq("test") & p.bps.eq(10)],hide_index=True)
    else:
        st.info("Open Pipeline and run the remaining steps, then inspect the results here.")
    st.caption("Fixed present-day universe: survivorship bias remains. Accuracy does not establish profitability. Vendor data may be revised.")

elif view=="Data quality":
    st.subheader("Coverage and quality")
    if not (run/"coverage.csv").exists():
        st.info("Run collection first.")
    else:
        st.dataframe(load(run/"coverage.csv"),hide_index=True)
        quality = read_json(run/"quality.json")
        st.write(f"{len(quality['issues'])} quality issues · {len(quality['failures'])} failed downloads")
        st.dataframe(pd.DataFrame(quality["issues"]),hide_index=True)
        if quality["failures"]:
            st.dataframe(pd.DataFrame(quality["failures"]),hide_index=True)
        bars = load(run/"bars.parquet")
        ticker = st.selectbox("Instrument",sorted(bars.ticker.unique()))
        selected = bars[bars.ticker.eq(ticker)]
        st.line_chart(selected,x="date",y="close",y_label="Split-normalized close")
        st.dataframe(selected.tail(50),hide_index=True)

elif view=="Model evaluation":
    st.subheader("Out-of-sample prediction quality")
    if not (run/"accuracy.csv").exists():
        st.info("Run training and prediction first.")
    else:
        accuracy = load(run/"accuracy.csv")
        phase = st.selectbox("Period",["test","development"])
        horizon = st.selectbox("Horizon (sessions)",sorted(accuracy.horizon.unique()))
        selected = accuracy[accuracy.phase.eq(phase)&accuracy.horizon.eq(horizon)]
        st.bar_chart(selected,x="model",y="accuracy",y_label="Directional accuracy")
        st.dataframe(selected,hide_index=True)
        st.caption("Classification uses probability 0.50. Trading uses the fixed threshold in the configuration. Overlapping labels are dependent; sample counts are not independent trials.")
        detail = load(run/"accuracy_detail.csv")
        st.subheader("By instrument and fold")
        st.dataframe(detail[detail.phase.eq(phase)&detail.horizon.eq(horizon)],hide_index=True)
        st.subheader("Predicted probabilities and signals")
        predictions = load(run/"predictions.parquet")
        ticker = st.selectbox("Instrument",sorted(predictions.ticker.unique()))
        selected_predictions = predictions[predictions.phase.eq(phase)&predictions.horizon.eq(horizon)
            &predictions.ticker.eq(ticker)&predictions.model.eq("xgboost")].sort_values("date")
        st.line_chart(selected_predictions,x="date",y="probability",y_label="Predicted probability of an up move")
        st.dataframe(selected_predictions[["date","entry_date","exit_date","probability","signal","target","forward_return"]].tail(30),hide_index=True)
        st.subheader("Validation windows")
        st.dataframe(pd.DataFrame(read_json(run/"splits.json")["folds"]),hide_index=True)

elif view in ["Portfolio","Trades and positions"]:
    if not (run/"performance.csv").exists():
        st.info("Run backtesting first.")
    else:
        perf = load(run/"performance.csv")
        phase = st.selectbox("Period",["test","development"])
        horizon = st.selectbox("Horizon (sessions)",sorted(set(perf.horizon)-{0}))
        models = sorted(set(perf.loc[perf.horizon.eq(horizon),"model"]))
        model = st.selectbox("Model",models,index=models.index("xgboost") if "xgboost" in models else 0)
        bps = st.selectbox("Slippage per side (bps)",sorted(perf.bps.unique()),index=list(sorted(perf.bps.unique())).index(10) if 10 in perf.bps.values else 0)
        chosen = perf[perf.phase.eq(phase)&perf.horizon.eq(horizon)&perf.model.eq(model)&perf.bps.eq(bps)]
        if chosen.empty:
            st.info("No results for these filters.")
            st.stop()
        result = chosen.iloc[0]
        prefix = run/"backtests"/result.artifact
        if view=="Portfolio":
            with st.container(horizontal=True):
                st.metric("Net return",f"{result.net_return:.2%}",border=True)
                st.metric("CAGR",f"{result.cagr:.2%}",border=True)
                st.metric("Sharpe",f"{result.sharpe:.2f}",border=True)
                st.metric("Maximum drawdown",f"{result.max_drawdown:.2%}",border=True)
            curves = []
            comparison = perf[perf.phase.eq(phase)&perf.bps.eq(bps)&(perf.artifact.eq(result.artifact)|perf.horizon.eq(0))]
            for row in comparison.itertuples():
                curve = load(run/"backtests"/f"{row.artifact}_equity.parquet")
                curves.append(curve.set_index("date").equity.rename(row.model))
            st.subheader("Portfolio value versus benchmarks")
            st.line_chart(pd.concat(curves,axis=1),y_label="Portfolio value ($)")
            curve = load(Path(str(prefix)+"_equity.parquet"))
            st.subheader("Drawdown")
            st.area_chart(curve,x="date",y="drawdown")
            st.subheader("Annual returns (partial years included)")
            annual = ((1+curve.set_index("date")["return"]).resample("YE").prod()-1)
            annual.index = annual.index.year.astype(str)
            st.bar_chart(annual.rename("Net return"))
            st.subheader("Cost sensitivity")
            costs = perf[perf.phase.eq(phase)&perf.horizon.eq(horizon)&perf.model.eq(model)]
            st.dataframe(costs.drop(columns=["artifact"]),hide_index=True)
        else:
            kind = st.segmented_control("Ledger",["trades","positions"],default="trades")
            ledger = load(Path(str(prefix)+f"_{kind or 'trades'}.parquet"))
            ticker = st.selectbox("Instrument",["All"]+sorted(ledger.ticker.unique().tolist()))
            if ticker!="All":
                ledger = ledger[ledger.ticker.eq(ticker)]
            st.dataframe(ledger,hide_index=True)
            st.download_button("Download ledger CSV",ledger.to_csv(index=False),file_name=f"{result.artifact}_{kind}.csv",mime="text/csv")

if "report" in manifest(run)["stages"] and (run/"report.html").exists():
    with st.sidebar:
        st.download_button("Download report",(run/"report.html").read_bytes(),file_name=f"{run_id}_report.html",mime="text/html")
