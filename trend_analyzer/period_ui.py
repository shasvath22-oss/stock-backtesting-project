from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from .periods import period_settings, validate_periods


def show_period_controls():
    defaults = validate_periods(period_settings())
    if st.session_state.get("period_defaults_signature") != defaults:
        st.session_state["active_periods"] = defaults.copy()
        st.session_state["period_defaults_signature"] = defaults.copy()
        for key in ["research_old_start","research_old_end","research_new_start","research_new_end","research_latest"]:
            st.session_state.pop(key,None)
    st.session_state.setdefault("active_periods", defaults.copy())
    cfg = st.session_state["active_periods"]
    yesterday = datetime.now(ZoneInfo("America/New_York")).date()-timedelta(days=1)
    with st.sidebar.expander("Research periods", expanded=True):
        with st.form("research_period_form"):
            old_start = st.date_input("Old start",pd.Timestamp(cfg["old_start"]).date(),key="research_old_start")
            old_end = st.date_input("Old end",pd.Timestamp(cfg["old_end"]).date(),key="research_old_end")
            new_start = st.date_input("New start",pd.Timestamp(cfg["new_start"]).date(),key="research_new_start")
            latest = st.checkbox("New end: latest completed session",value=cfg["new_end"] is None,key="research_latest")
            new_end = st.date_input("New end",pd.Timestamp(cfg["new_end"]).date() if cfg["new_end"] else yesterday,
                help="Used only when latest completed session is unchecked.",key="research_new_end")
            apply = st.form_submit_button("Apply periods")
        if apply:
            try:
                candidate = validate_periods({"old_start":str(old_start),"old_end":str(old_end),
                    "new_start":str(new_start),"new_end":None if latest else str(new_end)})
                if not latest and new_end > yesterday:
                    raise ValueError("Choose a completed date (yesterday or earlier) for New end.")
                st.session_state["active_periods"] = candidate
                st.success("Periods applied. Run Step 3/4 to calculate new results.")
            except ValueError as exc:
                st.error(str(exc))
        st.caption("End dates are inclusive. Changing periods does not download or recalculate. Old and new periods cannot overlap.")
    return st.session_state["active_periods"]
