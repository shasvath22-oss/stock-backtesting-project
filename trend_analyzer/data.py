from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import complete, digest, manifest, write_json


def sessions(start, end):
    import pandas_market_calendars as mcal
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index).tz_localize(None)


def synthetic_bars(tickers, calendar, seed):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.009, len(calendar))
    frames = []
    for ticker in tickers:
        close = 100 * np.exp(np.cumsum(market + rng.normal(0, 0.008, len(calendar))))
        op = close * np.exp(rng.normal(0, 0.003, len(calendar)))
        frames.append(pd.DataFrame({"date": calendar, "ticker": ticker, "open": op,
            "high": np.maximum(op, close)*1.005, "low": np.minimum(op, close)*0.995,
            "close": close, "adj_close": close, "volume": rng.integers(100000, 5000000, len(calendar)),
            "dividend": 0., "split": 0.}))
    return pd.concat(frames, ignore_index=True)


def validate_bars(frame, calendar):
    issues, cleaned = [], []
    for ticker, g in frame.groupby("ticker"):
        g = g.reset_index(drop=True).sort_values("date").copy()
        duplicate = g.duplicated("date", keep=False)
        valid = (g[["open", "high", "low", "close"]].gt(0).all(axis=1)
                 & g.volume.ge(0) & g.high.ge(g[["open", "close", "low"]].max(axis=1))
                 & g.low.le(g[["open", "close", "high"]].min(axis=1))
                 & np.isfinite(g[["open", "high", "low", "close", "volume"]]).all(axis=1)
                 & np.isfinite(g[["dividend","split"]]).all(axis=1)
                 & g.dividend.ge(0) & g.split.ge(0)
                 & g.date.isin(calendar) & ~duplicate)
        for date in g.loc[~valid, "date"]:
            issues.append({"ticker": ticker, "date": str(date), "issue": "invalid_or_duplicate_bar"})
        g = g.loc[valid]
        if len(g):
            missing = calendar[(calendar >= g.date.min()) & (calendar <= g.date.max())].difference(g.date)
            issues.extend({"ticker": ticker, "date": str(d), "issue": "missing_session"} for d in missing)
        cleaned.append(g)
    return pd.concat(cleaned, ignore_index=True), issues


def collect(run):
    run = Path(run)
    m = manifest(run)
    cfg = m["config"]
    # End is exclusive, matching yfinance. Never request an incomplete current session.
    calendar = sessions(cfg["start"], pd.Timestamp(cfg["end"]) - pd.Timedelta(days=1))
    symbols = list(dict.fromkeys(cfg["tickers"] + [cfg["benchmark"]]))
    failed = []
    if m["synthetic"]:
        bars = synthetic_bars(symbols, calendar, cfg["seed"])
    else:
        import yfinance as yf
        cache = Path("data/yfinance-cache")
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))
        frames = []
        for ticker in symbols:
            error = "empty response"
            for attempt in range(3):
                try:
                    g = yf.Ticker(ticker).history(start=cfg["start"], end=cfg["end"], interval="1d",
                        auto_adjust=False, back_adjust=False, actions=True, repair=False, keepna=True,
                        raise_errors=True, timeout=20)
                    if g.empty:
                        raise ValueError("empty response")
                    g.index = g.index.tz_localize(None).normalize()
                    g = g.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close",
                        "Adj Close":"adj_close", "Volume":"volume", "Dividends":"dividend", "Stock Splits":"split"})
                    g["date"], g["ticker"] = g.index, ticker
                    # ETFs may distribute capital gains; include these in cash distributions.
                    if "Capital Gains" in g:
                        g["dividend"] += g["Capital Gains"].fillna(0)
                    frames.append(g[["date","ticker","open","high","low","close","adj_close","volume","dividend","split"]])
                    break
                except Exception as exc:
                    error = str(exc)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            else:
                failed.append({"ticker": ticker, "error": error})
        if not frames:
            write_json(run / "download_failures.json", failed)
            raise ValueError("All yfinance downloads failed; see download_failures.json and check network access. No fallback data is used.")
        bars = pd.concat(frames, ignore_index=True)
    bars.to_parquet(run / "raw_bars.parquet",index=False)
    bars, issues = validate_bars(bars, calendar)
    coverage = bars.groupby("ticker").agg(first=("date","min"), last=("date","max"), sessions=("date","size")).reset_index()
    coverage["eligible"] = coverage.sessions >= cfg["minimum_sessions"]
    coverage["required_sessions"] = cfg["minimum_sessions"]
    coverage["requested_missing_sessions"] = [len(calendar.difference(pd.DatetimeIndex(bars.loc[bars.ticker.eq(t),"date"]))) for t in coverage.ticker]
    coverage.to_csv(run / "coverage.csv", index=False)
    write_json(run / "quality.json", {"issues": issues, "failures": failed,
               "survivorship_bias": "Fixed present-day universe; delisted coverage is not guaranteed."})
    bars.to_parquet(run / "bars.parquet", index=False)
    pd.DataFrame({"date": calendar}).to_parquet(run / "calendar.parquet", index=False)
    Path("data").mkdir(exist_ok=True)
    # Store prices in SQL with a unique snapshot, ticker and date key.
    with sqlite3.connect("data/market.sqlite") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS price_bars (snapshot_id TEXT, ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL, dividend REAL, split REAL, PRIMARY KEY(snapshot_id,ticker,date))")
        # One immutable snapshot per run; retry atomically replaces only this run's snapshot.
        with conn:
            conn.execute("DELETE FROM price_bars WHERE snapshot_id=?", (run.name,))
            rows = bars.copy()
            rows["date"] = rows.date.dt.strftime("%Y-%m-%d")
            conn.executemany("INSERT INTO price_bars VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(run.name, *r) for r in rows[["ticker","date","open","high","low","close","adj_close","volume","dividend","split"]].itertuples(index=False, name=None)])
    if cfg["benchmark"] not in bars.ticker.unique():
        raise ValueError("Benchmark download missing; resolve before proceeding")
    if failed:
        raise ValueError("Some downloads failed; review quality.json and retry collection before proceeding")
    complete(run, "collect", {"bars": len(bars), "sha256": digest(run / "bars.parquet")})
