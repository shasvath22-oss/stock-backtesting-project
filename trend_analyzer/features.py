from pathlib import Path

import numpy as np
import pandas as pd

from .common import complete, require, write_json


def indicators(g):
    """Only backward-looking, scale-independent features; input uses session grid."""
    c, o, h, l, v = (g[k] for k in ["close", "open", "high", "low", "volume"])
    # Dividend-inclusive trailing returns; split-normalized vendor price basis.
    ret = (c + g.dividend) / c.shift() - 1
    wealth = (1 + ret.fillna(0)).cumprod()
    out = pd.DataFrame(index=g.index)
    for n in [1, 5, 21, 63]:
        out[f"return_{n}"] = (1 + ret).rolling(n).apply(np.prod, raw=True) - 1
    for n in [10, 20, 50, 200]:
        sma = c.rolling(n).mean()
        ema = c.ewm(span=n, min_periods=n, adjust=False).mean()
        out[f"sma_distance_{n}"] = c / sma - 1
        out[f"ema_distance_{n}"] = c / ema - 1
        out[f"sma_slope_{n}"] = sma / sma.shift(5) - 1
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    out["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss.ne(0), 100)
    out.loc[gain.eq(0) & loss.eq(0), "rsi"] = 50
    macd = c.ewm(span=12, min_periods=12, adjust=False).mean() - c.ewm(span=26, min_periods=26, adjust=False).mean()
    signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    out["macd"], out["macd_signal"], out["macd_hist"] = macd/c, signal/c, (macd-signal)/c
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()/c
    out["volatility"] = ret.rolling(21).std()*np.sqrt(252)
    out["relative_volume"] = v / v.rolling(20).mean()
    out["volume_change"] = v / v.shift(5).replace(0, np.nan) - 1
    body, span = (c-o).abs(), (h-l).replace(0, np.nan)
    upper, lower = h-pd.concat([c,o], axis=1).max(axis=1), pd.concat([c,o], axis=1).min(axis=1)-l
    out["body"], out["upper_wick"], out["lower_wick"] = body/span, upper/span, lower/span
    out["gap"] = o/c.shift()-1
    out["doji"] = (body <= span*0.1).astype(float)
    out["hammer"] = ((lower >= 2*body) & (upper <= span*0.15) & (body <= span*0.35)).astype(float)
    out["bull_engulfing"] = ((c>o) & (c.shift()<o.shift()) & (o<=c.shift()) & (c>=o.shift())).astype(float)
    out["bear_engulfing"] = ((c<o) & (c.shift()>o.shift()) & (o>=c.shift()) & (c<=o.shift())).astype(float)
    for freq, name in [("W-FRI", "completed_week"), ("M", "completed_month")]:
        period = g.index.to_period(freq)
        # Shift by whole period: no partial/current calendar period is used.
        period_last = wealth.groupby(period).last()
        prev_return = period_last.pct_change(fill_method=None).shift(1)
        out[name] = pd.Series(period.map(prev_return), index=g.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


def build_dataset(bars, calendar, cfg, eligible):
    market = bars[bars.ticker.eq(cfg["benchmark"])].set_index("date").reindex(calendar)
    market_features = indicators(market)[["return_5", "volatility"]].add_prefix("market_")
    frames, columns = [], None
    for ticker in eligible:
        g = bars[bars.ticker.eq(ticker)].set_index("date").reindex(calendar)
        x = indicators(g).join(market_features)
        x["relative_market"] = x.return_5 - x.market_return_5
        # Reject a full warmup after any absent bar; never bridge a data hole.
        valid_window = g.close.notna().rolling(206, min_periods=206).sum().eq(206)
        x.loc[~valid_window, :] = np.nan
        columns = x.columns.tolist()
        dates = pd.Series(calendar, index=calendar)
        for horizon in cfg["horizons"]:
            row = x.copy()
            row["date"], row["ticker"], row["horizon"] = calendar, ticker, horizon
            row["entry_date"], row["exit_date"] = dates.shift(-1), dates.shift(-(horizon+1))
            entry, exit_price = g.open.shift(-1), g.open.shift(-(horizon+1))
            # Entitled if holding before ex-date open: entry < ex-date <= exit.
            dividends = sum(g.dividend.shift(-k) for k in range(2, horizon+2))
            good = g.close.notna().rolling(horizon+2).sum().shift(-(horizon+1)).eq(horizon+2)
            forward = (exit_price + dividends) / entry - 1
            row["forward_return"] = forward.where(good)
            row["target"] = np.where(row.forward_return.notna(), row.forward_return.gt(0).astype(float), np.nan)
            frames.append(row.reset_index(drop=True))
    if not frames:
        raise ValueError("No instruments meet minimum history requirement; inspect coverage.csv")
    return pd.concat(frames, ignore_index=True), columns


def features(run):
    run = Path(run)
    cfg = require(run, "collect")
    coverage = pd.read_csv(run / "coverage.csv")
    eligible = [t for t in cfg["tickers"] if t in coverage.loc[coverage.eligible, "ticker"].tolist()]
    data, columns = build_dataset(pd.read_parquet(run/"bars.parquet"),
        pd.DatetimeIndex(pd.read_parquet(run/"calendar.parquet").date), cfg, eligible)
    data.to_parquet(run / "features.parquet", index=False)
    write_json(run / "feature_schema.json", {"columns": columns, "eligible": eligible,
        "rows": len(data), "fully_labeled_rows": int(data.dropna(subset=columns+["target"]).shape[0])})
    complete(run, "features", {"eligible": eligible})
