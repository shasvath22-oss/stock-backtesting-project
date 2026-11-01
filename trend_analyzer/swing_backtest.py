"""Long/cash structural rule simulation using only confirmation-time inputs."""
import numpy as np
import pandas as pd

from .swings import swings_asof


def rule_signals(bars, events, combination="HH → HL"):
    signals = []
    for event in events[events.status.eq("accepted")].itertuples():
        points = swings_asof(events, event.confirmed_at)
        points = points[points.segment.eq(event.segment)]
        previous = points[points.swing_id.ne(event.swing_id)]
        if previous.empty or f"{previous.iloc[-1].label} → {event.label}" != combination:
            continue
        highs, lows = points[points.kind.eq("high")], points[points.kind.eq("low")]
        if highs.empty or lows.empty:
            continue
        stop, target = float(lows.iloc[-1].price), float(highs.iloc[-1].price)
        known = bars[bars.date.gt(event.pivot_date) & bars.date.le(event.confirmed_at)]
        if stop >= target or (known.high > target).any() or (known.low < stop).any():
            continue
        signals.append({"confirmed_at":event.confirmed_at,"stop":stop,"target":target,"combination":combination})
    return pd.DataFrame(signals, columns=["confirmed_at","stop","target","combination"])


def simulate_rule(bars, signals, capital=10000, bps=10, max_hold=20, benchmark=False):
    if capital <= 0 or bps < 0 or max_hold < 1 or bars.empty:
        raise ValueError("Invalid backtest settings")
    data = bars.sort_values("date").reset_index(drop=True)
    if not np.isfinite(data[["open","high","low","close","dividend"]].to_numpy()).all():
        raise ValueError("Backtest requires complete finite price/distribution data")
    slip = bps/10000
    if slip >= 1:
        raise ValueError("Slippage must be less than 10000 bps")
    dates = list(data.date)
    by_date = {date:i for i,date in enumerate(dates)}
    entries = {}
    for signal in signals.itertuples():
        if signal.confirmed_at in by_date and by_date[signal.confirmed_at]+1 < len(data):
            entries[by_date[signal.confirmed_at]+1] = signal
    cash, position = float(capital), None
    ledger, curves, audit = [], [], []
    for i, bar in enumerate(data.itertuples()):
        if position is not None and i > position["entry_index"]:
            distribution = position["units"] * bar.dividend
            cash += distribution
            position["dividends"] += distribution
        signal = entries.get(i)
        if position is not None and signal is not None:
            audit.append({"entry_date":bar.date,"status":"skipped_already_holding"})
        if position is None and i < len(data)-1 and (signal is not None or (benchmark and i == 0)):
            valid = benchmark or signal.stop < bar.open < signal.target
            if valid:
                cost = bar.open*(1+slip)
                units = cash/cost
                position = {"entry_date":bar.date,"entry_index":i,"entry_price":cost,"entry_reference":bar.open,
                    "units":units,"investment":cash,"dividends":0.,
                    "stop":signal.stop if not benchmark else 0.,"target":signal.target if not benchmark else np.inf,
                    "confirmed_at":signal.confirmed_at if not benchmark else pd.NaT}
                cash = 0.
                audit.append({"entry_date":bar.date,"status":"entered"})
            else:
                audit.append({"entry_date":bar.date,"status":"skipped_open_outside_levels"})
        if position is not None:
            reference, reason = None, None
            if not benchmark:
                if bar.open <= position["stop"]:
                    reference,reason = bar.open,"gap_stop"
                elif bar.open >= position["target"]:
                    reference,reason = position["target"],"target_at_open"
                elif bar.low <= position["stop"]:
                    reference,reason = position["stop"],"stop_first_if_both" if bar.high >= position["target"] else "stop"
                elif bar.high >= position["target"]:
                    reference,reason = position["target"],"target"
                elif i-position["entry_index"]+1 >= max_hold:
                    reference,reason = bar.close,"max_hold_close"
            if i == len(data)-1 and reference is None:
                reference,reason = bar.close,"end_of_data_close"
            if reference is not None:
                proceeds = position["units"]*reference*(1-slip)
                cash += proceeds
                pnl = proceeds+position["dividends"]-position["investment"]
                ledger.append(position | {"exit_date":bar.date,"exit_reference":reference,
                    "exit_price":reference*(1-slip),"reason":reason,"pnl":pnl,
                    "slippage_cost":position["units"]*(position["entry_reference"]+reference)*slip})
                position = None
        equity = cash+(position["units"]*bar.close if position else 0.)
        curves.append({"date":bar.date,"equity":equity})
    curve = pd.DataFrame(curves)
    curve["drawdown"] = curve.equity/curve.equity.cummax().clip(lower=capital)-1
    trades = pd.DataFrame(ledger, columns=["confirmed_at","entry_date","entry_index","entry_price","entry_reference",
        "units","investment","dividends","stop","target","exit_date","exit_reference","exit_price","reason","pnl","slippage_cost"])
    metrics = {"bps":bps,"net_return":curve.equity.iloc[-1]/capital-1,"max_drawdown":curve.drawdown.min(),
        "trades":len(trades),"win_rate":float(trades.pnl.gt(0).mean()) if len(trades) else 0.,
        "final_equity":curve.equity.iloc[-1],"slippage_cost":trades.slippage_cost.sum()}
    return metrics,curve,trades,pd.DataFrame(audit,columns=["entry_date","status"])
