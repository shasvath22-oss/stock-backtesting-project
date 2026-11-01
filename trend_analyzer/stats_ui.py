from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .common import digest, read_json, write_json
from .data import sessions
from .structure_stats import outcome_summary, structure_outcomes
from .periods import period_settings, period_windows, period_caption


def show_statistics(run, bars, manifest, analysis):
    st.subheader("Step 3 — what happened next?")
    st.write("")
    st.code("Example only: high $500, confirmed HL $490\nNext wick high $502 → upper_first\nNext wick low $488  → lower_first\nOne candle crosses BOTH → order unknown", language="text")
    st.write("For every newly accepted HH/HL/LH/LL, freeze the latest confirmed high and low in its segment. Starting the NEXT session after confirmation, find the first strict wick crossing. Touches equal to the level are not breaks. Swing extensions are not counted as new events.")
    with st.expander("Counting rules", expanded=True):
        st.write("")
        st.write("If the range was already broken between pivot and confirmation, exclude that event. Exclude missing-data windows and unfinished histories unless a break was already observed. Percentages use only completed outcomes, including ambiguous and no-break cases; excluded counts are shown separately. Events may overlap and are not independent trials.")
        st.caption("")
    with st.form(f"statistics_{analysis.name}"):
        horizon = st.number_input("Observe next N trading sessions", min_value=1, max_value=252, value=20)
        submit = st.form_submit_button("Run step 3: Count outcomes", type="primary")
    root = analysis / "statistics"
    if submit:
        events = pd.read_parquet(analysis / "events.parquet")
        calendar = sessions(manifest["start"], pd.Timestamp(manifest["end_exclusive"])-pd.Timedelta(days=1))
        partitions = []
        research_periods = manifest.get("research_periods",period_settings())
        for period, dates in period_windows(calendar,research_periods).items():
            if dates.empty:
                continue
            # Keep older endpoints as context, but only count events confirmed
            # within the period and never observe outcomes past its last session.
            result = structure_outcomes(bars,events[events.confirmed_at.le(dates[-1])],
                calendar[calendar <= dates[-1]],int(horizon))
            result = result[result.confirmed_at.between(dates[0],dates[-1])].copy()
            result["period"] = period
            partitions.append(result)
        outcomes = pd.concat(partitions,ignore_index=True) if partitions else pd.DataFrame()
        output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True)
        outcomes.to_parquet(output / "outcomes.parquet", index=False)
        write_json(output / "settings.json", {"horizon":int(horizon), "break_basis":"strict_wick", "start":"next_session_after_confirmation",
            "price_sha256":digest(run / "bars.parquet"), "events_sha256":digest(analysis / "events.parquet"),
            "version":"first_break_periods_v3", "research_periods":research_periods})
        st.session_state[f"statistics_run_{analysis.name}"] = str(output)
        st.success("Step 3 complete")
    runs = sorted([p for p in root.glob("*") if (p / "settings.json").exists()], reverse=True)
    if not runs:
        st.info("Run step 3 when ready.")
        return
    output = Path(st.selectbox("Saved statistics", [str(p) for p in runs], key=f"statistics_run_{analysis.name}"))
    settings = read_json(output / "settings.json")
    if settings["price_sha256"] != digest(run / "bars.parquet") or settings["events_sha256"] != digest(analysis / "events.parquet"):
        st.error("Inputs changed; run step 3 again.")
        return
    st.caption(f"Saved observation horizon: {settings['horizon']} sessions. Editing the form does not recalculate results.")
    outcomes = pd.read_parquet(output / "outcomes.parquet")
    if outcomes.empty:
        st.info("No HH/HL/LH/LL events in this analysis. Try a longer history.")
        return
    if "period" not in outcomes.columns:
        st.info("This result used the full history. Run step 3 again for separate old/new results using the dates on the left.")
        return
    st.caption("Saved results: " + period_caption(settings["research_periods"]))
    if settings["research_periods"] != manifest.get("research_periods",period_settings()):
        st.info("These results use earlier period settings. Run Step 3 to apply the dates now selected on the left.")
    period = st.selectbox("Research period",sorted(outcomes.period.unique()),key=f"statistics_period_{output.name}")
    outcomes = outcomes[outcomes.period.eq(period)].copy()
    summary = outcome_summary(outcomes)
    st.dataframe(summary, hide_index=True)
    st.caption("fraction_of_completed: 0.60 means 60%. Denominator includes upper, lower, both and no-break outcomes.")
    st.subheader("Swing combinations")
    st.write("HH → HL")
    st.caption("A combination joins the immediately preceding confirmed swing endpoint to this new accepted swing. Use the label known at confirmation; ignore rejected minor pivots and do not cross data gaps. Seed H/L and equal-price EH/EL predecessors are excluded from two-label combinations, not skipped over to invent a pair.")
    has_combinations = "combination" in outcomes.columns
    if not has_combinations:
        st.info("This saved result predates combinations. Run step 3 again to calculate them.")
    else:
        pairs = outcomes.dropna(subset=["combination"])
        st.caption(f"Events without a preceding HH/HL/LH/LL: {len(outcomes)-len(pairs)}. These remain in single-label statistics.")
        if len(pairs):
            pair_summary = outcome_summary(pairs, "combination")
            st.dataframe(pair_summary, hide_index=True)
            st.download_button("Download combination summary CSV", pair_summary.to_csv(index=False), file_name="combination_summary.csv", mime="text/csv")
        else:
            st.info("No two-label combinations in this history yet.")
    views = ["Single swing"]
    if has_combinations and outcomes.combination.notna().any():
        views.append("Combination")
    view = st.selectbox("Example grouping", views, key=f"stat_group_{output.name}")
    group_by = "combination" if view.startswith("Combination") else "label"
    label = st.selectbox("Structure to inspect", sorted(outcomes[group_by].dropna().unique()), key=f"stat_label_{output.name}_{group_by}")
    selected = outcomes[outcomes[group_by].eq(label)].reset_index(drop=True)
    index = st.selectbox("Choose one example", list(range(len(selected))),
        format_func=lambda i: f"{selected.iloc[i].confirmed_at.date()} — {selected.iloc[i].outcome}", key=f"stat_example_{output.name}_{label}")
    row = selected.iloc[index]
    st.write(f"{label} confirmed: {row.confirmed_at.date()}. Upper level ${row.upper_level:,.2f}; lower level ${row.lower_level:,.2f}. Result: **{row.outcome}**.")
    meanings = {"upper_first":"Upper level crossed first", "lower_first":"Lower level crossed first",
        "both_same_candle":"Both levels crossed in one candle; order unknown", "no_break":"Neither level crossed within the observation window",
        "incomplete_history":"Observation window is not yet complete", "data_gap":"Required prices are missing",
        "already_broken_before_confirmation":"Level crossed before confirmation; excluded", "invalid_levels":"Upper and lower levels are not in valid order"}
    pass  # English explanations are provided above.
    if pd.notna(row.break_date):
        st.write(f"Break: {row.break_date.date()}, {int(row.sessions_to_break)} trading sessions after confirmation.")
    saved_dates = period_windows(pd.DatetimeIndex(bars.date),settings["research_periods"])[period]
    chart_bars = bars[bars.date.isin(saved_dates)]
    end_dates = chart_bars[chart_bars.date.gt(row.confirmed_at)].head(int(row.horizon)).date
    end = end_dates.max() if len(end_dates) else row.confirmed_at
    begin = row.previous_pivot_date if group_by == "combination" else row.pivot_date
    visible = bars[bars.date.ge(begin) & bars.date.le(end)]
    if group_by == "combination":
        st.write(f"Previous {row.previous_label}: pivot {row.previous_pivot_date.date()}, confirmed {row.previous_confirmed_at.date()}, price ${row.previous_price:,.2f}. Then {row.label}: pivot {row.pivot_date.date()}, confirmed {row.confirmed_at.date()}. Outcome observation starts only after the SECOND confirmation.")
    base = alt.Chart(visible).encode(x="date:T", tooltip=["date:T","high:Q","low:Q","open:Q","close:Q"])
    wick = base.mark_rule(color="#64748b").encode(y=alt.Y("low:Q",scale=alt.Scale(zero=False),title="Price"),y2="high:Q")
    body = base.mark_bar(size=7).encode(y="open:Q",y2="close:Q",color=alt.condition("datum.close >= datum.open",alt.value("#16865b"),alt.value("#d44242")))
    levels = alt.Chart(pd.DataFrame({"price":[row.upper_level,row.lower_level],"level":["upper","lower"]})).mark_rule(strokeDash=[5,4]).encode(y="price:Q",color="level:N",tooltip=["level:N","price:Q"])
    markers = pd.DataFrame({"date":[row.confirmed_at],"event":["confirmation"]})
    if pd.notna(row.break_date):
        markers.loc[len(markers)] = [row.break_date,"break"]
    marks = alt.Chart(markers).mark_rule(strokeWidth=2).encode(x="date:T",color="event:N",tooltip=["date:T","event:N"])
    chart = wick+body+levels+marks
    if group_by == "combination":
        price = row.upper_level if row.label in {"HH","LH"} else row.lower_level
        pivots = pd.DataFrame({"date":[row.previous_pivot_date,row.pivot_date], "price":[row.previous_price,price], "label":[row.previous_label,row.label]})
        points = alt.Chart(pivots).mark_point(color="#6d28d9",filled=True,size=90).encode(x="date:T",y="price:Q",tooltip=["label:N","date:T","price:Q"])
        labels = alt.Chart(pivots).mark_text(dy=-14,color="#6d28d9").encode(x="date:T",y="price:Q",text="label:N")
        chart = chart+points+labels
    st.altair_chart(chart.properties(height=350).resolve_scale(color="independent"),width="stretch")
    st.dataframe(selected, hide_index=True)
    st.download_button("Download outcome audit CSV", outcomes.to_csv(index=False), file_name="structure_outcomes.csv",mime="text/csv")
