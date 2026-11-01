# Research requirements and verification

Current incremental rollout: **SPY + QQQ only**. All six research stages use these
two instruments by default. The 50-symbol implementation and validation below describe
the supported expansion, not the current starter universe. SPY is also the benchmark
and is collected once. Historical 50-symbol artifacts retain their original metadata.

The full research interface runs separately on port **8502** using `research_app.py`.
The original swing interface stays on port **8501** using `streamlit_app.py`.
Both supported interfaces use yfinance historical data only; generated data is
limited to internal automated fixtures and is hidden from user-facing run lists.

| Requirement | Implemented behavior | Primary files | Validation |
|---|---|---|---|
| 1. Collection/storage | 50 configured stocks/ETFs plus SPY; daily yfinance OHLCV/actions; raw and cleaned snapshots; pandas/NumPy validation; SQLite `(snapshot_id,ticker,date)` primary key; minimum 1,260 sessions; invalid/missing/failed data reported | `data.py`, `common.py` | 51-symbol downloader fixture and SQL idempotency test; real historical collection; quality and hash checks |
| 2. Features | 1/5/21/63-session returns, completed week/month returns; RSI/MACD/SMA/EMA/ATR/volatility/volume; engulfing/doji/hammer and candle geometry; missing-window warmup; separate future targets | `features.py` | Future-price perturbation causality test, feature-schema checks, label date/dividend alignment, missing-window checks |
| 3. Models | One pooled classifier per 1/5/21-session horizon; XGBoost, logistic, majority, always-up and momentum comparisons; probability and signal outputs | `modeling.py` | Complete multi-horizon demo, model/prediction horizon coverage, valid probability checks |
| 4. Validation | Shared chronological initial-train/validation/test dates; expanding folds; training requires known label exits by previous-close cutoff; date gaps excluded; validation scores cannot consume final-test outcomes | `modeling.py` | Partition ordering, mature-label and purging assertions, gap exclusion, scored-exit boundaries |
| 5. Backtest | Next-open execution; fixed horizon exits; one long/cash position per instrument; equal initial sleeves; fractional normalized units; daily returns/equity/positions/trades; distributions; same-date SPY/equal-weight comparison | `backtest.py` | Cash/holdings reconciliation, overlapping-signal allocation, distributions/no-double-split, missing-fill failure, benchmark-date checks |
| 6. Risk/costs | CAGR, annualized volatility, Sharpe (252 sessions, zero risk-free), maximum drawdown; 10 bps baseline and 0/5/10/20 bps cases using the identical saved predictions | `backtest.py`, `reporting.py` | Slippage arithmetic, finite risk metrics, prediction-hash reuse, persisted outcome verification |

## How to run each step

1. `python -m streamlit run research_app.py --server.port 8502`.
2. Open http://localhost:8502; the original swing UI can remain running on 8501.
3. Step 1: create a historical experiment, then **Collect prices + SQLite**.
   Historical collection requires network access. Defaults request more than five
   years. No generated/demo option or fallback is available in the supported UI/CLI.
4. Step 2: build features and targets; inspect input columns.
5. Step 3: save chronological splits; inspect train/validation/test dates and folds.
6. Step 4A: train; Step 4B: generate probabilities and accuracy tables.
7. Step 5: simulate all horizons/cost cases; inspect SPY and equal-weight comparison.
8. Step 6: review risks, generate HTML and optionally verify saved outputs.

The workflow deliberately places split declaration before model fitting. Every
stage runs from a submitted button and saves reusable outputs. An upstream rerun
invalidates downstream completion; old files are not displayed as current results.

## Persisted records

- `data/market.sqlite`: all per-run cleaned price snapshots.
- `artifacts/<run_id>/manifest.json`: effective configuration, package versions,
  synthetic/historical status and stage completion.
- `raw_bars.parquet`, `bars.parquet`, `calendar.parquet`, `coverage.csv`,
  `quality.json`: source and cleaned data, eligibility and issues.
- `features.parquet`, `feature_schema.json`: input columns and targets.
- `splits.json`: explicit partitions, fold boundaries and train cutoffs.
- `models/`, `predictions.parquet`, `accuracy.csv`, `accuracy_detail.csv`: models,
  audit metadata, probabilities and scores by horizon, instrument and fold.
- `backtests/`, `performance.csv`: daily equity/returns, positions, fills and risk/cost metrics.
- `report.html`, `verification.json` when generated: report and integrity record.

## Validation record

Automated tests cover the complete offline three-instrument, all-horizon pipeline,
all six UI pages, 50-instrument collection using a deterministic vendor fixture,
SQL keys/idempotency, feature causality, train-label maturity, date-gap exclusion,
portfolio accounting, costs, and the separately preserved swing workflow.

Historical collection on October 24, 2026 successfully downloaded all 50 configured
instruments plus SPY: each had 2,718 daily records, and all passed the configured
minimum-history requirement. The corresponding run is
`artifacts/20261024T175117382984Z`. All historical stages completed: 48 model
bundles, 717,000 prediction rows and 136 portfolio cases across validation/final
test, horizons/models/benchmarks and costs. Its `verification.json` passed all 16
persisted-output checks, including label maturity, SQL counts, portfolio accounting
and benchmark-date matching. The saved offline end-to-end example is
`artifacts/20261024T175016147533Z`; its verification also passed.
The complete automated suite passed **48 tests** after these changes. These are
engineering checks and do not establish investment performance or statistical significance.

## Interpretation boundaries

This validates research software, not guaranteed profitability. The universe is a
present-day list with survivorship bias; vendor data/adjustments may be revised.
Weekly/monthly forecast horizons mean 5/21 sessions, while completed calendar
returns are separate inputs. Model settings and the 0.55 trading threshold are
fixed rather than tuned on final test. Scheduled expanding-window test refits may
use matured earlier test outcomes. Cross-ticker and overlapping-horizon samples
are dependent; raw sample counts do not imply independent statistical evidence.
The final-test dates have been inspected in previous swing experiments and must
not be described as a pristine, never-seen holdout. No live orders are placed.
