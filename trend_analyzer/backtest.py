from pathlib import Path

import numpy as np
import pandas as pd

from .common import complete, read_json, require, digest


def simulate(bars, signals, calendar, tickers, capital, bps, threshold, benchmark=False):
    """Equal capital sleeves in split-normalized units; ex-date cash before open fills.

    Yahoo historical OHLC and distributions share a split-normalized basis. A split
    must NOT also change quantities here. These quantities are research units, not
    contemporaneous exchange share counts. No dividend-adjusted prices are traded.
    """
    if not tickers or capital <= 0 or bps < 0:
        raise ValueError("Invalid simulation configuration")
    slip = bps / 10000
    by_ticker = {t: bars[bars.ticker.eq(t)].set_index("date").reindex(calendar) for t in tickers}
    for ticker, frame in by_ticker.items():
        if frame.close.isna().any() or frame.open.isna().any():
            raise ValueError(f"Unresolved backtest price gap for {ticker}; no stale marks or invented fills allowed")
    # Materialize lightweight rows once, avoiding a pandas lookup for every held
    # instrument on every session across each cost/model/horizon scenario.
    bar_lookup = {t:dict(zip(calendar,g[["open","close","dividend"]].itertuples(index=False,name="Bar")))
                  for t,g in by_ticker.items()}
    cash = dict.fromkeys(tickers, capital/len(tickers))
    holdings = dict.fromkeys(tickers, 0.)
    exits, entries, cost_basis, distributions = {}, {}, {}, dict.fromkeys(tickers, 0.)
    ledger, equity, positions = [], [], []
    opened = {}
    session_number = {date:i for i,date in enumerate(calendar)}
    lookup = {}
    if not signals.empty:
        for row in signals.itertuples():
            if pd.notna(row.entry_date) and pd.notna(row.exit_date) and row.exit_date <= calendar[-1]:
                if row.date >= row.entry_date:
                    raise ValueError("Signal is not available before entry")
                lookup[(row.ticker, row.entry_date)] = row
    def fill(ticker, date, side, reference, quantity, reason):
        executed = reference*(1+slip if side=="buy" else 1-slip)
        value = quantity*executed
        ledger.append({"date": date, "ticker": ticker, "side": side, "reference_price": reference,
            "executed_price": executed, "normalized_units": quantity, "notional": value,
            "slippage_cost": quantity*reference*slip, "reason": reason,
            "signal_date": entries.get(ticker), "scheduled_exit": exits.get(ticker)})
        return value
    for date in calendar:
        for ticker in tickers:
            bar = bar_lookup[ticker][date]
            dividend = holdings[ticker]*float(bar.dividend or 0)
            cash[ticker] += dividend
            distributions[ticker] += dividend
            if holdings[ticker] and (date >= exits[ticker] or date == calendar[-1]):
                quantity = holdings[ticker]
                proceeds = fill(ticker,date,"sell",bar.open,quantity,"scheduled_exit")
                cash[ticker] += proceeds
                ledger[-1]["realized_pnl"] = proceeds + distributions[ticker] - cost_basis[ticker]
                ledger[-1]["holding_sessions"] = session_number[date]-session_number[opened[ticker]]
                holdings[ticker], distributions[ticker] = 0., 0.
            signal = lookup.get((ticker,date))
            should_buy = (benchmark and date==calendar[0]) or (signal is not None and signal.probability>=threshold)
            if not holdings[ticker] and should_buy and date < calendar[-1]:
                exits[ticker] = calendar[-1] if benchmark else signal.exit_date
                entries[ticker] = None if benchmark else signal.date
                opened[ticker] = date
                quantity = cash[ticker]/(bar.open*(1+slip))
                cost_basis[ticker] = fill(ticker,date,"buy",bar.open,quantity,"buy_and_hold" if benchmark else "model_signal")
                cash[ticker] -= cost_basis[ticker]
                holdings[ticker], distributions[ticker] = quantity, 0.
            value = holdings[ticker]*bar.close
            positions.append({"date":date,"ticker":ticker,"cash":cash[ticker],"normalized_units":holdings[ticker],
                "holdings_value":value,"equity":cash[ticker]+value})
            if cash[ticker] < -1e-7:
                raise AssertionError("Negative cash")
        todays = positions[-len(tickers):]
        total = sum(p["equity"] for p in todays)
        equity.append({"date":date,"equity":total,"cash":sum(p["cash"] for p in todays),
            "exposure":sum(p["holdings_value"] for p in todays)/total})
    curve = pd.DataFrame(equity)
    # Include first day's implementation costs/P&L relative to initial capital.
    curve["return"] = curve.equity.pct_change()
    curve.loc[0,"return"] = curve.loc[0,"equity"]/capital-1
    curve["drawdown"] = curve.equity / curve.equity.cummax().clip(lower=capital)-1
    trades = pd.DataFrame(ledger, columns=["date","ticker","side","reference_price","executed_price",
        "normalized_units","notional","slippage_cost","reason","signal_date","scheduled_exit","realized_pnl","holding_sessions"])
    return curve, trades, pd.DataFrame(positions)


def performance(curve, trades, capital):
    daily = curve["return"]
    years = max((curve.date.iloc[-1]-curve.date.iloc[0]).days/365.25, 1/365.25)
    vol = daily.std(ddof=1)*np.sqrt(252)
    sells = trades[trades.side.eq("sell")]
    longest, current = 0, 0
    for below_peak in curve.drawdown.lt(-1e-12):
        current = current+1 if below_peak else 0
        longest = max(longest,current)
    return {"net_return": curve.equity.iloc[-1]/capital-1,
        "cagr": (curve.equity.iloc[-1]/capital)**(1/years)-1,
        "annualized_volatility": vol, "sharpe": daily.mean()*252/vol if vol>0 else 0.,
        "max_drawdown": curve.drawdown.min(), "average_exposure": curve.exposure.mean(),
        "max_drawdown_sessions": longest,
        "average_holding_sessions": sells.holding_sessions.mean() if len(sells) else 0.,
        "trades": len(sells), "win_rate": sells.realized_pnl.gt(0).mean() if len(sells) else 0.,
        "turnover": trades.notional.sum()/capital,
        "slippage_cost": trades.slippage_cost.sum()}


def backtest(run, bps=None):
    run = Path(run)
    cfg = require(run, "predict")
    bars = pd.read_parquet(run/"bars.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(run/"calendar.parquet").date)
    pred = pd.read_parquet(run/"predictions.parquet")
    tickers = read_json(run/"feature_schema.json")["eligible"]
    root = run/"backtests"
    root.mkdir(exist_ok=True)
    costs = [bps] if bps is not None else [0]+cfg["slippage_bps"]
    records = []
    for phase in ["development", "test"]:
        phase_pred = pred[pred.phase.eq(phase)]
        if phase_pred.empty:
            continue
        first = phase_pred.entry_date.min()
        # Identical evaluation dates for every horizon and model, including exit tail.
        last = min(phase_pred.exit_date.max(), calendar[-1])
        if "phase_end" in phase_pred:
            last = min(last,pd.to_datetime(phase_pred.phase_end).max())
        if phase=="development":
            test_start = pd.Timestamp(read_json(run/"splits.json")["test_start"])
            last = min(last, calendar[calendar < test_start][-1])
        dates = calendar[(calendar>=first)&(calendar<=last)]
        if len(dates)<2:
            continue
        for cost in costs:
            for (horizon, model), signals in phase_pred.groupby(["horizon","model"]):
                curve, trades, positions = simulate(bars,signals,dates,tickers,cfg["capital"],cost,cfg["threshold"])
                name = f"{phase}_h{horizon}_{model}_{cost}bps"
                for label, frame in [("equity",curve),("trades",trades),("positions",positions)]:
                    frame.to_parquet(root/f"{name}_{label}.parquet", index=False)
                records.append({"phase":phase,"horizon":horizon,"model":model,"bps":cost,
                    "artifact":name, **performance(curve,trades,cfg["capital"])})
            # Compare buy-and-hold benchmarks using the same dates and trading costs.
            for name, universe in [("SPY",[cfg["benchmark"]]),("equal_weight",tickers)]:
                curve,trades,positions = simulate(bars,pd.DataFrame(),dates,universe,cfg["capital"],cost,0,benchmark=True)
                artifact = f"{phase}_{name}_{cost}bps"
                for label, frame in [("equity",curve),("trades",trades),("positions",positions)]:
                    frame.to_parquet(root/f"{artifact}_{label}.parquet", index=False)
                records.append({"phase":phase,"horizon":0,"model":name,"bps":cost,"artifact":artifact,
                    **performance(curve,trades,cfg["capital"])})
    incoming = pd.DataFrame(records)
    if (run/"performance.csv").exists() and bps is not None:
        previous = pd.read_csv(run/"performance.csv")
        incoming = pd.concat([previous[~previous.bps.isin(costs)],incoming],ignore_index=True)
    incoming.to_csv(run/"performance.csv",index=False)
    complete(run,"backtest",{"costs":costs,"baseline_bps_per_side":10,
        "predictions_sha256":digest(run/"predictions.parquet")})
