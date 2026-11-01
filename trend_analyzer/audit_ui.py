from datetime import datetime,timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .common import digest,read_json,write_json
from .data import sessions
from .periods import period_windows,period_caption
from .model_audit import audit_candidates


def show_model_audit(run,bars,manifest,analysis):
    st.subheader("Step 6 — audit model decisions")
    st.write("Model")
    root = analysis / "model_runs"
    models = sorted([p for p in root.glob("*") if (p / "settings.json").exists()],reverse=True)
    if not models:
        st.info("Run Step 5 first to save a model and its predictions.")
        return
    model_path = Path(st.selectbox("Model experiment to audit",[str(p) for p in models],key=f"audit_model_{analysis.name}"))
    cfg = read_json(model_path / "settings.json")
    if cfg["price_sha256"] != digest(run / "bars.parquet") or cfg["events_sha256"] != digest(analysis / "events.parquet"):
        st.error("Inputs changed; select an experiment with matching prices and swing events.")
        return
    st.caption("Uses the model's saved dates: " + period_caption(cfg["research_periods"]))
    st.write(f"Fixed model threshold {cfg['threshold']:.0%}. Each hypothetical trade starts with the SAME ${cfg['capital']:,.0f}, at {cfg['max_hold']} maximum sessions. Hypothetical outcomes are independent and can overlap; adding them is NOT portfolio profit. Actual portfolio trade P&L is shown separately and uses its changing capital.")
    with st.form(f"audit_form_{model_path.name}"):
        bps = st.selectbox("Audit execution cost",[5,10,20],index=1)
        submit = st.form_submit_button("Run step 6: Audit selected and rejected signals",type="primary")
    outputs = model_path / "signal_audits"
    if submit:
        calendar = sessions(manifest["start"],pd.Timestamp(manifest["end_exclusive"])-pd.Timedelta(days=1))
        dates = period_windows(calendar,cfg["research_periods"])["new"]
        data = bars[bars.date.isin(dates)]
        if len(data)<2 or len(dates.difference(pd.DatetimeIndex(data.date))):
            st.error("The saved new period needs complete price sessions.")
            return
        features = pd.read_parquet(model_path / "features.parquet")
        candidates = features[features.confirmed_at.isin(dates)]
        predictions = pd.read_parquet(model_path / "new_predictions.parquet")
        result = audit_candidates(data,candidates,predictions,
            pd.read_parquet(model_path / f"xgboost_filter_{bps}bps_trades.parquet"),
            pd.read_parquet(model_path / f"all_HH_HL_{bps}bps_trades.parquet"),
            pd.read_parquet(model_path / f"xgboost_filter_{bps}bps_audit.parquet"),
            cfg["capital"],int(bps),cfg["max_hold"])
        output = outputs / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True)
        result.to_parquet(output / "candidate_audit.parquet",index=False)
        write_json(output / "settings.json",{"version":"candidate_audit_v1","bps":bps,
            "predictions_sha256":digest(model_path / "new_predictions.parquet"),"capital":cfg["capital"],
            "price_sha256":cfg["price_sha256"],"threshold":cfg["threshold"]})
        st.session_state[f"selected_audit_{model_path.name}"] = str(output)
        st.success("Audit saved. Model unchanged.")
    saved = sorted([p for p in outputs.glob("*") if (p / "settings.json").exists()],reverse=True)
    if not saved:
        return
    output = Path(st.selectbox("Saved decision audit",[str(p) for p in saved],key=f"selected_audit_{model_path.name}"))
    settings = read_json(output / "settings.json")
    if settings["predictions_sha256"] != digest(model_path / "new_predictions.parquet"):
        st.error("Predictions changed; generate a new audit.")
        return
    audit = pd.read_parquet(output / "candidate_audit.parquet")
    st.caption(f"Saved audit cost: {settings['bps']} bps per side. Form changes apply only on Run.")
    if audit.empty:
        st.info("No new-period candidates in this experiment.")
        return
    selected, rejected = audit[audit.decision.eq("selected")], audit[audit.decision.eq("rejected")]
    a,b = st.columns(2)
    a.metric("Selected profitable",int(selected.outcome.eq("profit").sum()))
    b.metric("Selected losing",int(selected.outcome.eq("loss").sum()))
    c,d = st.columns(2)
    c.metric("Rejected profitable",int(rejected.outcome.eq("profit").sum()))
    d.metric("Rejected losing",int(rejected.outcome.eq("loss").sum()))
    st.caption("Counts use mature standalone hypothetical outcomes, not portfolio execution. Skipped entries, incomplete histories, flat trades and missing inputs remain separate below.")
    st.dataframe(audit.groupby(["decision","outcome"]).size().rename("count").reset_index(),hide_index=True)
    st.dataframe(audit,hide_index=True)
    choice = st.selectbox("Inspect one model decision",range(len(audit)),
        format_func=lambda i:f"{audit.iloc[i].confirmed_at.date()} — {audit.iloc[i].decision}",key=f"audit_example_{output.name}")
    row = audit.iloc[choice]
    st.write(f"Probability: {row.probability:.1%}; decision: {row.decision}; standalone result: {row.outcome}; model portfolio: {row.model_execution}.")
    if pd.notna(row.hypothetical_pnl):
        pass  # English explanations are provided above.
    end = row.exit_date if pd.notna(row.exit_date) else row.confirmed_at
    view = bars[bars.date.between(row.confirmed_at-pd.Timedelta(days=10),end)]
    base = alt.Chart(view).encode(x="date:T",tooltip=["date:T","open:Q","high:Q","low:Q","close:Q"])
    wick = base.mark_rule(color="#64748b").encode(y=alt.Y("low:Q",scale=alt.Scale(zero=False)),y2="high:Q")
    body = base.mark_bar(size=6).encode(y="open:Q",y2="close:Q",color=alt.condition("datum.close >= datum.open",alt.value("#16865b"),alt.value("#d44242")))
    levels = alt.Chart(pd.DataFrame({"price":[row.stop,row.target],"level":["stop","target"]})).mark_rule(strokeDash=[4,4]).encode(y="price:Q",color="level:N")
    st.altair_chart((wick+body+levels).properties(height=330),width="stretch")
    st.download_button("Download decision audit CSV",audit.to_csv(index=False),file_name=f"new_model_decisions_{settings['bps']}bps.csv",mime="text/csv")
