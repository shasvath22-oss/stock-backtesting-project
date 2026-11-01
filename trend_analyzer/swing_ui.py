from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from .common import digest, read_json, write_json
from .data import sessions
from .swings import detect_swings, swings_asof


def show_swings(run, bars, manifest, step=2):
    root = run / "swings"
    if step == 2:
        st.subheader("Step 2 — confirmed swings + structure")
        st.write("A strict pivot high exceeds neighboring highs; a pivot low is below neighboring lows. It becomes known after the right-side candles close. The minimum opposite move is 0.386% of the last accepted pivot PRICE, not 38.6% of a previous leg. Exactly 0.386% qualifies.")
        st.write("")
        with st.expander("Exact rules", expanded=True):
            st.write("The first pivot seeds the reference price. Each opposite pivot must then move at least 0.386% from that price. A more extreme same-type pivot extends the existing swing; the updated endpoint becomes available only on its new confirmation date.")
            st.write("")
            st.write("A missing session resets the swing sequence; windows never bridge gaps. Tied neighboring extremes are not strict pivots. This is a pivot-and-retracement definition, not an intrabar ZigZag. Labels describe structure, not buy/sell recommendations.")
        with st.form(f"swings_{run.name}"):
            left = st.number_input("Left candles", min_value=1, max_value=20, value=2)
            right = st.number_input("Right confirmation candles", min_value=1, max_value=20, value=2)
            st.write("Minimum price change: 0.386% = 0.00386 as a decimal. Formula: opposite move ÷ last accepted pivot price × 100.")
            submit = st.form_submit_button("Run step 2: Detect swings", type="primary")
        if submit:
            with st.spinner("Detecting confirmed pivots and filtering small retracements..."):
                calendar = sessions(manifest["start"], pd.Timestamp(manifest["end_exclusive"])-pd.Timedelta(days=1))
                events = detect_swings(bars, calendar, int(left), int(right))
                output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                output.mkdir(parents=True)
                events.to_parquet(output / "events.parquet", index=False)
                write_json(output / "settings.json", {"left":int(left), "right":int(right), "minimum_price_change":0.00386,
                    "input_sha256":digest(run / "bars.parquet"), "algorithm":"strict_pivot_price_change_v2",
                    "seed":"first_pivot", "outside_bar":"skip", "missing_session":"reset",
                    "same_type":"extend_only_if_more_extreme; retain_asof_event_history"})
            st.session_state[f"swing_analysis_{run.name}"] = str(output)
            st.success("Swings saved. No training or trading has run.")
    analyses = sorted([p for p in root.glob("*") if (p / "settings.json").exists()], reverse=True)
    if not analyses:
        if step != 2:
            st.subheader(f"Step {step}")
        st.info("Run step 2 after inspecting prices.")
        return
    chosen = st.sidebar.selectbox("Saved swing analysis", [str(p) for p in analyses], key=f"swing_analysis_{run.name}")
    from pathlib import Path
    output = Path(chosen)
    settings = read_json(output / "settings.json")
    if settings["input_sha256"] != digest(run / "bars.parquet"):
        st.error("Input prices changed. Run a new swing analysis.")
        return
    if settings.get("algorithm") != "strict_pivot_price_change_v2":
        st.warning("This saved analysis uses the earlier leg-retracement rule. Run step 2 again to use 0.386% of price.")
        return
    st.caption(f"Saved settings: left={settings['left']}, right={settings['right']}, minimum price change={settings['minimum_price_change']:.3%}. In the audit, ratio=0.00386 means 0.386%; reference_range is the reference pivot price.")
    if step == 3:
        from .stats_ui import show_statistics
        show_statistics(run, bars, manifest, output)
        return
    if step == 5:
        from .model_ui import show_model
        show_model(run,bars,manifest,output)
        return
    if step == 6:
        from .audit_ui import show_model_audit
        show_model_audit(run,bars,manifest,output)
        return
    if step in (4,"4b"):
        from .rule_ui import show_rule_backtest
        show_rule_backtest(run, bars, manifest, output, selected_period="old" if step == 4 else "new")
        return
    events = pd.read_parquet(output / "events.parquet")
    dates = bars.date.sort_values().tolist()
    cutoff = st.selectbox("Known by this close", dates, index=len(dates)-1,
        format_func=lambda d: str(d.date()), key=f"asof_{output.name}")
    endpoints = swings_asof(events, cutoff)
    shown = bars[bars.date.le(cutoff)].tail(180)
    visible = endpoints[endpoints.pivot_date.ge(shown.date.min())]
    base = alt.Chart(shown).encode(x=alt.X("date:T", title="Session date"))
    colors = alt.condition("datum.close >= datum.open", alt.value("#16865b"), alt.value("#d44242"))
    wick = base.mark_rule().encode(y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price"), y2="high:Q", color=colors)
    body = base.mark_bar(size=4).encode(y="open:Q", y2="close:Q", color=colors)
    pivots = alt.Chart(visible).encode(x="pivot_date:T", y="price:Q", tooltip=["label:N","pivot_date:T","confirmed_at:T","price:Q","ratio:Q"])
    line = pivots.mark_line(color="#5b54cf").encode(detail="segment:N", order="pivot_date:T")
    points = pivots.mark_point(color="#5b54cf", filled=True, size=75)
    labels = pivots.mark_text(dy=-13, color="#4338ca", fontSize=13).encode(text="label:N")
    confirmations = alt.Chart(visible).mark_point(shape="diamond", color="#e08016", size=65).encode(x="confirmed_at:T", y="price:Q", tooltip=["label:N","pivot_date:T","confirmed_at:T"])
    st.altair_chart((wick+body+line+points+labels+confirmations).properties(height=420), width="stretch")
    st.caption("Purple = pivot location; orange diamond = date it became known, drawn at the pivot price (not a fill). The line joins current endpoints as known by the selected close. New extreme extensions change endpoints only from their confirmation date.")
    st.dataframe(endpoints, hide_index=True)
    known = events[events.confirmed_at.le(cutoff)]
    st.write("Full event audit: accepted, extended and ignored")
    st.dataframe(known, hide_index=True)
    st.download_button("Download swing event audit CSV", known.to_csv(index=False), file_name="swing_events.csv", mime="text/csv")
