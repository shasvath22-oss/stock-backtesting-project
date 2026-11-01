from datetime import datetime,timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from .common import digest,read_json,write_json
from .data import sessions
from .periods import period_settings,period_windows,period_caption
from .swing_backtest import simulate_rule
from .swing_model import FEATURES,PARAMETERS,signal_features,label_old_signals,fit_filter


def show_model(run,bars,manifest,analysis):
    st.subheader("Step 5 — XGBoost filter")
    cfg = manifest.get("research_periods",period_settings())
    st.caption(period_caption(cfg))
    st.write("")
    st.write("Inputs: returns, volatility, distance to stop/target, swing size/duration and relative volume, observed at confirmation close. Answer: did this candidate trade finish profitably after 10 bps per side, with the same stop/target/holding rules? Each candidate is labeled independently, even when candidates overlap; skipped next-open entries and unfinished old-period trades are excluded.")
    st.write("Old period: first 75% of labeled signals for chronological training; last 25% for validation. Remove training outcomes that do not exit before validation starts. Then refit fixed settings on all matured old examples and freeze the model for the new period. No new-period targets enter fitting. Threshold is fixed at 0.55; no automated tuning.")
    st.caption("Signals are few and dependent. The UI reports sample counts and compares validation with a constant training-profit-rate baseline. Previously explored dates are not an untouched test. Model probabilities are not calibrated confidence levels.")
    with st.expander("Input columns and fixed model settings"):
        st.write(FEATURES)
        st.json(PARAMETERS)
        st.write("distance_to_stop/target uses confirmation CLOSE, not the unknown next open. target_profit and label_exit are answers/metadata and are excluded from inputs.")
    with st.form(f"model_form_{analysis.name}"):
        capital = st.number_input("Model comparison capital",min_value=100.,value=10000.,step=100.)
        hold = st.number_input("Model maximum holding sessions",min_value=1,max_value=252,value=20)
        submit = st.form_submit_button("Run step 5: Train and compare",type="primary")
    root = analysis / "model_runs"
    if submit:
        try:
            with st.spinner("Preparing past examples, validating and fitting the filter..."):
                calendar = sessions(manifest["start"],pd.Timestamp(manifest["end_exclusive"])-pd.Timedelta(days=1))
                if len(calendar.difference(pd.DatetimeIndex(bars.date))):
                    raise ValueError("Resolve missing price sessions before training; feature windows must not bridge gaps.")
                windows = period_windows(calendar,cfg)
                if len(windows["old"])<2 or len(windows["new"])<2:
                    raise ValueError("Collect prices covering both selected old and new periods first.")
                events = pd.read_parquet(analysis / "events.parquet")
                features = signal_features(bars,events)
                examples = label_old_signals(bars,features,windows["old"],int(hold))
                model,validation,scores,audit = fit_filter(examples)
                new = features[features.confirmed_at.isin(windows["new"])].copy()
                predictions = new.dropna(subset=FEATURES).copy()
                predictions["probability"] = model.predict_proba(predictions[FEATURES])[:,list(model.classes_).index(1)] if len(predictions) else pd.Series(dtype=float)
                predictions["selected"] = predictions.probability.ge(.55)
                filtered = predictions[predictions.selected]
                new_bars = bars[bars.date.isin(windows["new"])]
                results,frames = [],{}
                for cost in [5,10,20]:
                    for name,signals,benchmark in [("all_HH_HL",new,False),("xgboost_filter",filtered,False),("buy_hold",new,True)]:
                        metrics,curve,trades,entries = simulate_rule(new_bars,signals,float(capital),cost,int(hold),benchmark)
                        results.append(metrics | {"model":name,"net_pnl":metrics["final_equity"]-capital})
                        for kind,frame in [("equity",curve),("trades",trades),("audit",entries)]:
                            frames[f"{name}_{cost}bps_{kind}.parquet"] = frame
                output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                output.mkdir(parents=True)
                joblib.dump({"model":model,"features":FEATURES},output / "model.joblib")
                for name,frame in [("features.parquet",features),("training_examples.parquet",examples),
                    ("validation_predictions.parquet",validation),("new_predictions.parquet",predictions)]:
                    frame.to_parquet(output / name,index=False)
                for name,frame in frames.items():
                    frame.to_parquet(output / name,index=False)
                scores.to_csv(output / "validation_scores.csv",index=False)
                pd.DataFrame(results).to_csv(output / "performance.csv",index=False)
                write_json(output / "settings.json",{"version":"swing_xgb_v1","research_periods":cfg,"threshold":.55,
                    "capital":capital,"max_hold":hold,"label_bps":10,"costs":[5,10,20],"parameters":PARAMETERS,
                    "price_sha256":digest(run / "bars.parquet"),"events_sha256":digest(analysis / "events.parquet"),
                    "audit":audit,"new_candidates":len(new),"new_missing_features":len(new)-len(predictions),"selected_candidates":len(filtered)})
                st.session_state[f"model_run_{analysis.name}"] = str(output)
                st.success("Model and comparison saved")
        except ValueError as exc:
            st.error(str(exc))
    runs = sorted([p for p in root.glob("*") if (p / "settings.json").exists()],reverse=True)
    if not runs:
        st.info("Run step 5 to prepare and train. Minimum: 20 matured old examples, at least 10 purged training examples and both target classes.")
        return
    output = Path(st.selectbox("Saved model experiment",[str(p) for p in runs],key=f"model_run_{analysis.name}"))
    settings = read_json(output / "settings.json")
    if settings["price_sha256"] != digest(run / "bars.parquet") or settings["events_sha256"] != digest(analysis / "events.parquet"):
        st.error("Inputs changed. Run a new model experiment.")
        return
    st.caption("Saved experiment: " + period_caption(settings["research_periods"]))
    if settings["research_periods"] != cfg:
        st.info("These results use earlier periods. Run Step 5 again to use current sidebar dates.")
    audit = settings["audit"]
    st.write(f"Old examples: {audit['old_examples']}; training: {audit['training_examples']}; validation: {audit['validation_examples']}; purged: {audit['purged_examples']}.")
    if audit["old_examples"]<100:
        st.warning("Fewer than 100 old examples: treat this as a small research experiment.")
    st.dataframe(pd.read_csv(output / "validation_scores.csv"),hide_index=True)
    st.caption("Validation accuracy uses 50%; trading uses 55%. Brier score measures probability error; lower is better. This validation table is from the earlier training-only model, before final refitting.")
    st.write(f"New candidates: {settings['new_candidates']}; missing inputs: {settings['new_missing_features']}; model selected: {settings['selected_candidates']}. Executed trades may be fewer because of entry rules and existing positions.")
    cost = st.selectbox("Model comparison cost",[5,10,20],index=1,key=f"model_cost_{output.name}")
    performance = pd.read_csv(output / "performance.csv")
    st.dataframe(performance[performance.bps.eq(cost)],hide_index=True)
    curves = [pd.read_parquet(output / f"{name}_{cost}bps_equity.parquet").set_index("date").equity.rename(name)
        for name in ["all_HH_HL","xgboost_filter","buy_hold"]]
    st.line_chart(pd.concat(curves,axis=1),y_label="New-period portfolio value")
    st.write("New-period estimated probabilities")
    predictions = pd.read_parquet(output / "new_predictions.parquet")
    st.dataframe(predictions,hide_index=True)
    with st.expander("Old training inputs and answers"):
        st.dataframe(pd.read_parquet(output / "training_examples.parquet"),hide_index=True)
        st.json(audit)
    trades = pd.read_parquet(output / f"xgboost_filter_{cost}bps_trades.parquet")
    st.write("Model-filtered trades")
    st.dataframe(trades,hide_index=True)
    st.download_button("Download model-filtered trades CSV",trades.to_csv(index=False),file_name=f"new_xgboost_{cost}bps_trades.csv",mime="text/csv")
