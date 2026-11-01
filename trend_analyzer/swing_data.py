"""Step 1 of swing research: one immutable, inspectable price snapshot."""
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

from .common import digest, write_json
from .data import sessions, synthetic_bars, validate_bars
from .periods import period_settings


def collect_snapshot(ticker, start, end, demo=False, root="data/swing_runs", periods=None):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ticker):
        raise ValueError("Use one US stock or ETF symbol, for example QQQ or MSFT.")
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start >= end:
        raise ValueError("Start must be before the exclusive end date.")
    if end > pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None):
        raise ValueError("End must be today or earlier; the end date is excluded.")
    calendar = sessions(start, end-pd.Timedelta(days=1))
    if calendar.empty:
        raise ValueError("No NYSE sessions in this date range.")
    if demo:
        raw = synthetic_bars([ticker], calendar, seed=42)
    else:
        import yfinance as yf
        cache = Path(root) / "vendor_cache"
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))
        raw = yf.Ticker(ticker).history(start=str(start.date()), end=str(end.date()),
            interval="1d", auto_adjust=False, back_adjust=False, actions=True,
            repair=False, keepna=True, raise_errors=True, timeout=20)
        if raw.empty:
            raise ValueError("No prices returned. Check the symbol, dates and connection.")
        raw.index = raw.index.tz_localize(None).normalize()
        raw = raw.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close",
            "Adj Close":"adj_close", "Volume":"volume", "Dividends":"dividend", "Stock Splits":"split"})
        raw["date"], raw["ticker"] = raw.index, ticker
        if "Capital Gains" in raw:
            raw["dividend"] += raw["Capital Gains"].fillna(0)
        raw = raw[["date","ticker","open","high","low","close","adj_close","volume","dividend","split"]]
    bars, issues = validate_bars(raw, calendar)
    if bars.empty:
        raise ValueError("No valid price rows remain after validation.")
    # Report all requested missing sessions, including range boundaries. Do not fill.
    missing = calendar.difference(pd.DatetimeIndex(bars.date))
    run = Path(root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    raw.to_parquet(run / "raw_bars.parquet", index=False)
    bars.to_parquet(run / "bars.parquet", index=False)
    write_json(run / "quality.json", {"issues": issues,
        "missing_requested_sessions": [str(d.date()) for d in missing],
        "expected_sessions": len(calendar), "valid_rows": len(bars)})
    write_json(run / "manifest.json", {"workflow":"swing_research", "completed_steps":["prices"],
        "ticker":ticker, "start":str(start.date()), "end_exclusive":str(end.date()),
        "synthetic":demo, "source":"generated" if demo else "Yahoo Finance",
        "research_periods":periods or period_settings(),
        "swing_rule":{"minimum_price_change_ratio":0.00386,
            "reference":"last accepted pivot price", "mode":"price_change",
            "comparison":"greater_than_or_equal", "status":"configured_for_separate_step_2"},
        "baseline_slippage_bps_per_side":10, "sensitivity_bps_per_side":[5,10,20],
        "price_convention":"Vendor split-normalized OHLC; distributions stored separately; adjustments may be revised.",
        "sha256":digest(run / "bars.parquet")})
    return run
