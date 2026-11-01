"""New swing-first learning workflow. Run: python -m streamlit run swing_app.py"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trend_analyzer.swing_data import collect_snapshot
from trend_analyzer.swing_ui import show_swings
from trend_analyzer.common import read_json
from trend_analyzer.periods import period_caption
from trend_analyzer.period_ui import show_period_controls
import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Swing research — step by step", layout="wide", initial_sidebar_state="expanded")
st.title("Swing research")
st.caption("New workflow: one stock, one visible step at a time.")
with st.sidebar:
    st.header("Steps")
    step = st.radio("Choose a step", [1, 2, 3, 4, "4b", 5, 6],
        format_func=lambda n: {1:"Step 1: Prices", 2:"Step 2: Swings + Structure", 3:"Step 3: Statistics + Combinations", 4:"Step 4A: Old-period Backtest", "4b":"Step 4B: New-period Backtest", 5:"Step 5: XGBoost Filter", 6:"Step 6: Model Decision Audit"}[n], key="workflow_step")
    st.caption("Select a step here. Nothing runs until you press its Run button.")
    st.caption("")
active_periods = show_period_controls()
st.info(period_caption(active_periods))
ROADMAP = [
    ("1. Prices", "Available", "Download one stock's daily OHLCV; inspect candles and missing data."),
    ("2. Swings", "Available", "Confirmed pivots with a 0.386% minimum price change; separate pivot and confirmation dates."),
    ("3. Structure", "Labels available", "HH, HL, LH, LL and equal-price labels. Trading break rules are a later step."),
    ("Step 3. Statistics", "First-break counts available", "Strict wick crossings after confirmation: upper/lower first, ambiguous, no break, incomplete or excluded. Return/excursion and uncertainty analysis are future work."),
    ("Step 4. Rules", "Available", "Long HH → HL: next-open entry, HL stop, prior-high target, holding limit; 5/10/20 bps and buy-and-hold comparison."),
    ("Step 5. XGBoost", "Available", "Train a fixed HH → HL profitability filter on matured old-period examples; validate chronologically and compare new-period portfolios."),
    ("Costs", "Available in Steps 4–5", "Baseline 10 bps per side; reuse signals at 5/10/20 bps."),
]
with st.expander("Project roadmap", expanded=False):
    st.dataframe(pd.DataFrame(ROADMAP, columns=["Step", "Status", "Purpose"]), hide_index=True)
st.info("Steps 1–4: prices → swings → statistics → historical trading simulation. Use Step 5 to train the optional XGBoost filter.")
if step == 1:
    st.subheader("Step 1 — collect and understand prices")
    st.info("Minimum swing size: 0.386% of the last accepted pivot price. This is a price-percentage filter, not a Fibonacci retracement of the prior leg.")
    with st.expander("0.386% price-change example"):
        st.write("From a $100 pivot: $99.80 is a 0.20% pullback, ignored. $99.614 is a 0.386% pullback, meeting the size threshold. For an upward move from $100, $100.386 meets the threshold. Pivot confirmation is still required.")
    st.write("Before identifying HH or HL, inspect the candles that will define swings. A candle contains an opening price, high, low and close. No model or future target is needed for this step.")
    st.write("HH/HL")
    with st.form("prices"):
        ticker = st.text_input("One US stock", "QQQ")
        st.caption("Data source: Yahoo Finance via yfinance only")
        start = st.date_input("Start date", value=pd.Timestamp(active_periods["old_start"]).date())
        default_end = (pd.Timestamp(max(active_periods["old_end"],active_periods["new_end"]))+pd.Timedelta(days=1)).date() if active_periods["new_end"] else datetime.now(ZoneInfo("America/New_York")).date()
        end = st.date_input("End date (excluded)", value=default_end)
        submit = st.form_submit_button("Run step 1: Save prices", type="primary")
    if submit:
        try:
            with st.spinner("Collecting and validating prices..."):
                run = collect_snapshot(ticker, start, end, demo=False, periods=active_periods)
            st.session_state["swing_run"] = str(run)
            st.success("Step 1 complete")
        except Exception as exc:
            st.error(str(exc))
            st.stop()
root = Path("data/swing_runs")
runs = sorted([p for p in root.glob("*") if (p / "manifest.json").exists() and not read_json(p/"manifest.json").get("synthetic",True)], reverse=True)
if not runs:
    st.write("Choose Step 1 on the left to save prices first.")
    st.stop()
names = [str(p) for p in runs]
if st.session_state.get("swing_run") not in names:
    st.session_state.pop("swing_run",None)
selected = st.sidebar.selectbox("Saved snapshot", names, key="swing_run")
run = Path(selected)
m = read_json(run / "manifest.json")
m["research_periods"] = active_periods
quality = read_json(run / "quality.json")
bars = pd.read_parquet(run / "bars.parquet").sort_values("date")
st.write(f"**Displayed snapshot: {m['ticker']} · {m['source']} · {m['start']} → {m['end_exclusive']} (end excluded)**")
if pd.Timestamp(m["start"]) > pd.Timestamp(active_periods["old_start"]):
    st.warning("This saved snapshot starts after your selected old-period start. Recollect a wider range in Step 1 for full coverage.")
st.caption("The chart shows this saved snapshot; editing the form does not change it until you run step 1 again.")
if m["synthetic"]:
    st.warning("Generated demonstration prices, not this stock's actual history.")
if step == 1:
    st.metric("Valid daily bars", len(bars))
    if quality["issues"] or quality["missing_requested_sessions"]:
        st.warning("Inspect missing/invalid data before swing analysis; no missing prices were filled.")
    with st.expander("Data quality"):
        st.json(quality)
    count = st.slider("Candles to display", 10, max(10, len(bars)), min(120, max(10,len(bars))))
    visible = bars.tail(count)
    base = alt.Chart(visible).encode(x=alt.X("date:T", title="Session date"),
        color=alt.condition("datum.close >= datum.open", alt.value("#16865b"), alt.value("#d44242")),
        tooltip=["date:T", "open:Q", "high:Q", "low:Q", "close:Q", "volume:Q"])
    wick = base.mark_rule().encode(y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price"), y2="high:Q")
    body = base.mark_bar(size=5).encode(y="open:Q", y2="close:Q")
    st.altair_chart((wick + body).properties(height=380), width="stretch")
    st.caption("Green: close ≥ open; red: close < open. Thin wick: low to high. Hover to inspect prices.")
    choice = st.selectbox("Inspect one session", list(range(len(visible))),
        format_func=lambda i: str(visible.iloc[i].date.date()))
    row = visible.iloc[choice]
    st.write(f"{row.date.date()}: open ${row.open:,.2f} → close ${row.close:,.2f}; high ${row.high:,.2f}, low ${row.low:,.2f}. Intraday change: {(row.close/row.open-1):+.2%}.")
    st.write("")
    st.dataframe(visible, hide_index=True)
    st.download_button("Download prices CSV", bars.to_csv(index=False), file_name=f"{m['ticker']}_prices.csv", mime="text/csv")
    st.caption(f"Price snapshot saved to {run}. Use the left sidebar to choose the next step.")
else:
    show_swings(run, bars, m, step=step)
