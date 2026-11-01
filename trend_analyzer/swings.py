"""Causal pivot events and a minimum retracement filter. No future-label inputs."""
import math

import pandas as pd


def detect_swings(bars, calendar, left=2, right=2, minimum=0.00386, mode="price_change"):
    if left < 1 or right < 1 or not 0 < minimum <= 1:
        raise ValueError("Use positive candle windows and a retracement ratio in (0, 1].")
    if mode not in {"price_change", "leg_retracement"}:
        raise ValueError("Unknown swing size mode")
    if bars.date.duplicated().any():
        raise ValueError("Expected one stock with unique session dates.")
    frame = bars.set_index("date").reindex(calendar)
    events, accepted, segment, last_gap = [], [], 0, -1
    next_id = 0
    for known in range(len(frame)):
        if frame.iloc[known][["high", "low"]].isna().any():
            accepted = []
            segment += 1
            last_gap = known
            continue
        pivot = known-right
        if pivot-left <= last_gap:
            continue
        window = frame.iloc[pivot-left:known+1]
        if len(window) != left+right+1 or window[["high","low"]].isna().any().any():
            continue
        row = frame.iloc[pivot]
        neighbors = window.drop(frame.index[pivot])
        high = row.high > neighbors.high.max()
        low = row.low < neighbors.low.min()
        if not high and not low:
            continue
        event = {"pivot_date":frame.index[pivot], "confirmed_at":frame.index[known],
                 "segment":segment, "kind":"high" if high else "low", "price":float(row.high if high else row.low),
                 "ratio":None, "reference_range":None, "swing_id":None, "label":None}
        if high and low:
            events.append(event | {"kind":"both", "status":"ignored", "reason":"outside_bar_ambiguous"})
            continue
        kind, price = event["kind"], event["price"]
        if accepted and kind == accepted[-1]["kind"]:
            more_extreme = price > accepted[-1]["price"] if kind == "high" else price < accepted[-1]["price"]
            if not more_extreme:
                events.append(event | {"status":"ignored", "reason":"same_type_not_more_extreme"})
                continue
            previous_same = next((p for p in reversed(accepted[:-1]) if p["kind"] == kind), None)
            event["swing_id"] = accepted[-1]["swing_id"]
            status, reason = "extended", "same_swing_new_extreme"
        else:
            required = 1 if mode == "price_change" else 2
            if len(accepted) >= required:
                reference = abs(accepted[-1]["price"]) if mode == "price_change" else abs(accepted[-1]["price"]-accepted[-2]["price"])
                move = accepted[-1]["price"]-price if kind == "low" else price-accepted[-1]["price"]
                ratio = move/reference if reference else 0
                event.update(ratio=ratio, reference_range=reference)
                if ratio < minimum and not math.isclose(ratio, minimum, abs_tol=1e-12, rel_tol=0):
                    events.append(event | {"status":"ignored", "reason":"below_minimum_retracement"})
                    continue
            previous_same = next((p for p in reversed(accepted) if p["kind"] == kind), None)
            event["swing_id"] = next_id
            next_id += 1
            status = "accepted"
            reason = "seed_no_previous_leg" if len(accepted) < required else "minimum_retracement_met"
        if previous_same is None:
            label = "H" if kind == "high" else "L"
        elif math.isclose(price, previous_same["price"], abs_tol=1e-10, rel_tol=0):
            label = "EH" if kind == "high" else "EL"
        elif kind == "high":
            label = "HH" if price > previous_same["price"] else "LH"
        else:
            label = "HL" if price > previous_same["price"] else "LL"
        event.update(label=label, status=status, reason=reason)
        events.append(event.copy())
        if status == "extended":
            accepted[-1] = event
        else:
            accepted.append(event)
    result = pd.DataFrame(events, columns=["pivot_date","confirmed_at","segment","kind","price","ratio",
        "reference_range","swing_id","label","status","reason"])
    for column in ["ratio", "reference_range", "price"]:
        result[column] = pd.to_numeric(result[column]).astype(float)
    result["swing_id"] = result.swing_id.astype("Int64")
    for column in ["pivot_date", "confirmed_at"]:
        result[column] = pd.to_datetime(result[column])
    return result


def swings_asof(events, date):
    """Reconstruct endpoints from the event history available by a chosen close."""
    known = events[events.confirmed_at.le(pd.Timestamp(date)) & events.status.isin(["accepted","extended"])]
    return known.drop_duplicates("swing_id", keep="last").sort_values("pivot_date")
