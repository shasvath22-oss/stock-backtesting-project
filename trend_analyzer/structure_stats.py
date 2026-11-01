"""Descriptive first-wick-break outcomes after structural event confirmation."""
import pandas as pd

from .swings import swings_asof


def structure_outcomes(bars, events, calendar, horizon=20):
    if horizon < 1:
        raise ValueError("Observation horizon must be positive")
    frame = bars.set_index("date").reindex(calendar)
    records = []
    # Extensions update an existing swing; do not count them as fresh events.
    candidates = events[events.status.eq("accepted") & events.label.isin(["HH","HL","LH","LL"])]
    for event in candidates.itertuples():
        endpoints = swings_asof(events, event.confirmed_at)
        endpoints = endpoints[endpoints.segment.eq(event.segment)]
        highs, lows = endpoints[endpoints.kind.eq("high")], endpoints[endpoints.kind.eq("low")]
        if highs.empty or lows.empty:
            continue
        upper, lower = float(highs.iloc[-1].price), float(lows.iloc[-1].price)
        # The previous endpoint's label must be the version known at this close,
        # not a label revised by a later extension. Never bridge a gap segment.
        preceding = endpoints[endpoints.swing_id.ne(event.swing_id)]
        previous = preceding.iloc[-1] if len(preceding) else None
        previous_label = previous.label if previous is not None else None
        combination = f"{previous_label} → {event.label}" if previous_label in {"HH","HL","LH","LL"} else None
        record = {"label":event.label, "pivot_date":event.pivot_date, "confirmed_at":event.confirmed_at,
            "previous_label":previous_label, "combination":combination,
            "previous_pivot_date":previous.pivot_date if previous is not None else pd.NaT,
            "previous_confirmed_at":previous.confirmed_at if previous is not None else pd.NaT,
            "previous_price":previous.price if previous is not None else None,
            "upper_level":upper, "lower_level":lower, "horizon":horizon,
            "outcome":None, "break_date":pd.NaT, "sessions_to_break":None, "observed_sessions":0}
        if upper <= lower:
            records.append(record | {"outcome":"invalid_levels"})
            continue
        prior = frame.loc[(frame.index > event.pivot_date) & (frame.index <= event.confirmed_at)]
        if prior[["high","low"]].isna().any().any():
            records.append(record | {"outcome":"data_gap"})
            continue
        if (prior.high > upper).any() or (prior.low < lower).any():
            records.append(record | {"outcome":"already_broken_before_confirmation"})
            continue
        future = frame.loc[frame.index > event.confirmed_at].head(horizon)
        for number, (date, bar) in enumerate(future.iterrows(), start=1):
            if pd.isna(bar.high) or pd.isna(bar.low):
                record["outcome"] = "data_gap"
                break
            record["observed_sessions"] = number
            above, below = bar.high > upper, bar.low < lower
            if above or below:
                record.update(outcome="both_same_candle" if above and below else "upper_first" if above else "lower_first",
                              break_date=date, sessions_to_break=number)
                break
        if record["outcome"] is None:
            record["outcome"] = "no_break" if len(future) == horizon else "incomplete_history"
        records.append(record)
    result = pd.DataFrame(records, columns=["label","previous_label","combination","previous_pivot_date",
        "previous_confirmed_at","previous_price","pivot_date","confirmed_at","upper_level","lower_level",
        "horizon","outcome","break_date","sessions_to_break","observed_sessions"])
    for col in ["pivot_date", "confirmed_at", "break_date", "previous_pivot_date", "previous_confirmed_at"]:
        result[col] = pd.to_datetime(result[col])
    return result


def outcome_summary(outcomes, group_by="label"):
    if group_by not in {"label", "combination"}:
        raise ValueError("Group by label or combination")
    rows = []
    complete = {"upper_first","lower_first","both_same_candle","no_break"}
    for label, group in outcomes.groupby(group_by):
        eligible = group[group.outcome.isin(complete)]
        count = len(eligible)
        for outcome in sorted(complete):
            n = int(eligible.outcome.eq(outcome).sum())
            rows.append({group_by:label, "outcome":outcome, "count":n,
                "completed_events":count, "excluded_events":len(group)-count,
                "fraction_of_completed":n/count if count else None})
    return pd.DataFrame(rows)
