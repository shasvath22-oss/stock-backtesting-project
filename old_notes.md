# Stock Direction Research Pipeline — Implementation Plan

Status: Proposed for review  
Date: October 18, 2026  
Scope: Research and historical simulation; automated live trading is outside this plan.

Implementation update: A first working CLI and Streamlit version now exists. See README.md for commands,
implemented conventions, and research limitations. Streamlit provides run creation, explicit pipeline controls,
data quality, model comparisons, equity/drawdown charts, cost sensitivity, and downloadable ledgers/reports.
This document preserves the original proposed architecture; README.md records implementation differences.

## 1. Objective and definition of success

Build a reproducible Python pipeline that collects at least five years of daily OHLCV data for 50 stocks/ETFs, predicts price direction over three horizons, and evaluates executable portfolio strategies after trading costs.

Engineering success means every stage runs independently, produces traceable artifacts, and passes data, timing, and accounting checks. Research success means an honest comparison with simple models and SPY. Positive returns or outperformance are outcomes to measure, not implementation guarantees.

The initial version will use 10 instruments plus SPY and one daily target. After the full path works, expand to 50 instruments and all three horizons.

## 2. Proposed decisions for review

| Area | Proposed default | Reason |
|---|---|---|
| Market | US-listed liquid stocks and ETFs | Consistent daily market calendar and USD accounting |
| History | Prefer 10 years where available; require at least 5 for core universe | More market regimes and usable validation periods |
| Storage | SQLite through SQLAlchemy | Local, inspectable SQL storage without a database server |
| Models | Three pooled XGBoost classifiers, one per horizon | Shares information across instruments without maintaining 150 separate models |
| Horizons | 1, 5, and 21 trading sessions | Explicit daily, weekly, and monthly approximations |
| Signals | Generated after session close | Current session OHLCV is then available |
| Execution | Following session open | Consistent executable timing |
| Strategy | Long/cash, no leverage or shorting | Clear first implementation and accounting |
| Portfolio | Equal capital sleeves across eligible instruments | Prevents overlapping signals from exceeding shared capital |
| Costs | 10 bps adverse slippage per side; repeat at 5 and 20 | Matches requested sensitivity analysis |
| Benchmark | SPY total-return comparison over identical dates | Accounts for distributions in the comparison |
| Interface | Command-line stages first | Independent runs, debugging, and reproducibility |

The 1/5/21-session targets are not calendar-week or calendar-month returns. Historical calendar-week and calendar-month returns will also be reported as descriptive features, using only completed periods.

## 3. Architecture and proposed folder structure

```text
config/
  universe.csv
  research.yaml
src/trend_analyzer/
  cli.py
  data/          # downloads, SQL tables, calendar, quality checks
  features/      # indicators, patterns, targets
  validation/    # date splits, purging, walk-forward folds
  models/        # baselines, XGBoost, evaluation
  backtest/      # execution, sizing, cash and holdings
  reporting/     # metrics, tables, plots
tests/
data/
  market.sqlite
artifacts/<run_id>/
  manifest.json
  data_quality/
  datasets/
  splits/
  models/
  predictions/
  backtests/
  reports/
project_notes.md
README.md
pyproject.toml
```

Each run records its configuration, software versions, random seeds, code version when available, input snapshot identifiers, and artifact locations. Large datasets and model artifacts are excluded from source control by default.

## 4. Stage 1 — Data collection and SQL storage

### Inputs and collection

- Maintain an explicit, versioned universe with ticker, asset type, inclusion date, and optional sector.
- Include SPY even when it is outside the 50 modeling instruments.
- Download daily data with yfinance using explicit adjustment and corporate-action options; do not rely on library defaults.
- Use bounded batches, retries with backoff, download logs, and incremental updates.
- Refresh a configurable overlap of recent sessions on updates to capture corrections. Preserve snapshots or hashes so a previous experiment remains traceable.

### Proposed SQL tables

| Table | Key | Contents |
|---|---|---|
| instruments | instrument_id | Ticker mapping, metadata, eligibility |
| price_bars | instrument_id + session_date + snapshot_id | Vendor OHLCV, adjusted close, source and retrieval time |
| corporate_actions | instrument_id + date + action_type + snapshot_id | Splits and cash distributions |
| ingestion_runs | ingestion_id | Request ranges, outcomes, failures, snapshot information |
| quality_issues | issue_id | Missing sessions, invalid bars, action anomalies, resolution |

Ticker/date identifies a bar within a snapshot. Instrument identifiers allow future handling of ticker changes without treating every symbol as a new economic instrument.

### Cleaning and price conventions

- Validate uniqueness, chronological order, nonnegative volume, positive prices, and OHLC ordering.
- Compare dates against the exchange session calendar. Holidays are not missing observations.
- Do not backward-fill prices or fabricate bars. Exclude affected feature/label windows and log why.
- Keep pre-inception periods empty. Require enough contiguous history for indicator warmup and target calculation.
- Retain vendor fields and corporate actions. Do not assume vendor OHLC represents fully unadjusted historical exchange quotes; verify split conventions on known examples.
- Define separate, consistent series for feature calculations, executable prices, and total returns. Apply split changes to holdings and dividend cash flows consistently in the backtest, avoiding double counting with adjusted prices.
- Treat questionable corporate-action adjustments as quality failures requiring resolution before evaluation.

### Universe limitations

A list selected from today's surviving instruments introduces survivorship and selection bias. The first report will explicitly describe the fixed universe and its limitations. A point-in-time universe with delisted securities is a later enhancement requiring suitable data coverage.

**Acceptance:** Re-running ingestion creates no duplicate bars; failed tickers and incomplete histories are visible; missing windows and corporate-action handling are verified before downstream use.

## 5. Stage 2 — Features and targets

### Initial feature set

| Group | Features |
|---|---|
| Returns | Trailing 1, 5, 21, and 63-session returns; completed calendar-week/month returns |
| Trend | SMA/EMA distances and slopes at a small fixed set of windows |
| Momentum | RSI, MACD line/signal/histogram |
| Risk | Rolling return volatility, ATR relative to price |
| Volume | Relative volume and trailing volume change |
| Candle shape | Body/range, upper/lower wick fractions, opening gap |
| Patterns | Bullish/bearish engulfing, doji, hammer; explicit numerical definitions |
| Market context | Trailing SPY return and volatility, relative performance versus SPY |

Use scale-independent features where practical so pooled learning does not depend on nominal share price. Keep the initial feature set modest. Pattern thresholds are configuration values frozen before the final test.

### Target and timing contract

For a signal formed after close of session `t`, enter at open of session `t+1`. For horizon `h`, exit at open of session `t+h+1`. Thus a daily label measures one complete open-to-open holding interval.

The forward return includes corporate actions over the holding interval under the chosen accounting convention. The classification target is `1` when that gross return is positive and `0` otherwise. Costs are applied in portfolio evaluation; a positive direction probability alone does not establish a profitable trade.

Every labeled row stores:

- Instrument, feature timestamp, earliest execution timestamp.
- Target horizon, entry timestamp, exit timestamp, forward return, and direction.
- Feature version, data snapshot, and eligibility flags.

The last observations without known exits remain unlabeled and are excluded from training and historical scoring.

### Leakage prevention

- Compute rolling features independently per instrument in chronological order.
- Use trailing windows only; no centered rolling calculations or future fills.
- Join market context by information availability, not merely matching a later timestamp.
- Fit imputers, feature selection, and any learned transformations only on each training fold.
- Do not convert unknown labels into negative examples through comparisons with NaN.
- Preserve warmup missing values or exclude rows with a documented policy; never fill them with future values.

**Acceptance:** Changing data after a feature timestamp cannot change that row's features. Manually verified examples match each target's entry and exit prices.

## 6. Stage 3 — Chronological validation

### Final holdout

Reserve approximately the most recent 20% of eligible dates, preferably at least one year, as an untouched final test. Use earlier data for development, expanding-window validation, and limited tuning. Exact date boundaries are written to the split artifact before model selection.

If only five years are available, a practical starting schedule is approximately three years of initial training, one year of development validation, and one year of final testing, adjusted for warmup and purging.

### Expanding-window procedure

1. Train on the initial historical window.
2. Purge training examples whose targets would not be known at the simulated fit time or overlap the next evaluation interval.
3. Predict the next three-month block.
4. Expand training through the next allowed cutoff and repeat.
5. Concatenate only predictions from periods unseen at their corresponding fit time.

Split by unique session dates across all tickers. Never split a stacked multi-ticker frame by raw row count: that can place the same date in training and evaluation.

Purging uses the stored label entry/exit timestamps for each horizon. A fixed gap is acceptable only after proving it implements the same boundary rule. Weekly/monthly overlapping labels also create dependent observations within test blocks; report that limitation and use block-based uncertainty estimates rather than treating every row as independent.

For the final walk-forward test, freeze features, hyperparameters, thresholds, sizing, and retraining schedule. Previously elapsed test observations may enter later training only once their labels are known, following that frozen schedule. No manual adaptation to observed final-test results.

**Acceptance:** No training label crosses its information cutoff; no evaluation date appears in that fold's training features; all final predictions carry a model fit cutoff.

## 7. Stage 4 — Models and predictive evaluation

### Models

- Always-up classifier and training-window majority classifier.
- Simple momentum direction rule.
- Logistic regression with transformations fitted inside each fold.
- XGBoost binary classifier, separately fitted for each horizon.

Start with pooled models across instruments. Compare per-instrument models only as a later experiment if there is sufficient history. Fifty instruments share market information and do not supply 50 fully independent histories.

Use a bounded hyperparameter search on development folds. Record all trials. Early stopping must use an internal chronological validation interval within the training data, with the same label-boundary protections; never use the evaluation block for early stopping.

### Outputs and measures

- Save up-direction probability, predicted class, actual label, horizon, model version, and timestamps.
- Report accuracy, balanced accuracy, precision, recall, confusion matrix, ROC AUC where defined, log loss, and Brier score.
- Show results by horizon, fold, instrument, and pooled dataset, including sample counts and class balance.
- Assess probability calibration on development data. If calibration is used, fit it on a separate chronological development interval.
- Report simple accuracy at 0.50 separately from the trading threshold. Select a small number of trading thresholds on development data and freeze the chosen value before final testing.
- Compare model performance on identical eligible observations.

**Acceptance:** Models reload and reproduce saved predictions; every model has baseline comparisons; final-test data never selects parameters.

## 8. Stage 5 — Trading simulation

### Initial strategy rules

- Run daily, weekly, and monthly portfolios separately. Combining them is a later experiment.
- Allocate equal initial capital to instrument sleeves. Each sleeve holds either cash or one long position; no borrowing and no shorting.
- At an eligible close, enter when the model probability exceeds the development-selected threshold. Fill at the following open with adverse slippage.
- Hold for the target's exact 1/5/21-session interval and exit at its scheduled open. Do not layer additional overlapping positions in the same sleeve.
- Permit a new scheduled position at an exit open only when its signal was available at the previous close. Record both sell and buy costs when they represent separate trades.
- Keep proceeds in the sleeve. Idle cash earns zero in the initial simulation; state this assumption in reports.
- Size purchases including execution cost so cash cannot become negative. Use fractional shares in the initial research model; whole-share execution can be an optional later mode.

The initial implementation should use a small, transparent accounting engine or a verified framework adapter. PyBroker is a candidate, not a committed dependency until its fill timing, multi-instrument capital, corporate-action conventions, and cost behavior pass a short integration check.

### Accounting and exceptional cases

- Record signal time, order time, fill time, reference price, executed price, quantity, costs, and reason for every trade.
- Mark holdings to daily close and track cash, holdings value, and equity.
- Apply splits and distributions exactly once under the established price convention.
- Do not execute on a fabricated or missing open. Skip unavailable entries; defer unavailable exits to the next valid executable session and flag the event. Report material unresolved data gaps.
- Stop opening trades whose scheduled exits exceed the evaluation range. Resolve remaining positions under a documented terminal policy.

### Cost implementation

For slippage fraction `s`, buy at `reference_open * (1+s)` and sell at `reference_open * (1-s)`. Ten basis points means `s=0.001` per side. Do not deduct the same slippage again as a separate fee.

Use identical saved predictions and frozen rules for the 5, 10, and 20 bps runs. Capital and quantities may differ as costs change. Commissions are zero initially and remain a separate configurable cost.

**Acceptance:** Cash and equity reconcile; positions obey capital limits; synthetic examples verify fill timing, round-trip costs, dividends, splits, and exits.

## 9. Stage 6 — Performance and reports

Produce a readable HTML or Markdown report plus machine-readable CSV/JSON tables.

| Category | Required outputs |
|---|---|
| Prediction | Per-horizon and per-fold accuracy and baseline comparisons |
| Returns | Net cumulative return, CAGR, annualized volatility |
| Risk | Sharpe ratio, maximum drawdown, drawdown duration |
| Trading | Trade count, turnover, exposure, holding duration, win rate |
| Costs | Gross/net results and 5/10/20 bps comparison |
| Benchmarks | SPY and equal-weight buy-and-hold of the research universe |
| Charts | Equity curves, drawdowns, annual returns, fold performance |
| Reliability | Data coverage, excluded rows, failed downloads, run configuration |

Compute CAGR from actual elapsed time. Compute daily Sharpe from daily excess returns using 252 sessions per year; use a stated zero risk-free rate initially. Calculate maximum drawdown from the daily marked-to-market equity curve.

Compare all portfolios over identical dates and initial capital. Include SPY distributions and apply a clearly stated entry/liquidation cost convention to benchmark trades. Report exposure because a cash-heavy strategy can have lower risk and lower returns for that reason alone.

**Acceptance:** Metrics recompute from the saved equity curve; all cost cases share the same signals; final results include unfavorable folds and instruments.

## 10. Separately runnable commands

Proposed interface; these commands are not implemented yet:

```powershell
python -m trend_analyzer collect --config config/research.yaml
python -m trend_analyzer features --config config/research.yaml
python -m trend_analyzer splits --config config/research.yaml
python -m trend_analyzer train --config config/research.yaml --horizon 1
python -m trend_analyzer predict --run-id <run_id>
python -m trend_analyzer backtest --run-id <run_id> --slippage-bps 10
python -m trend_analyzer report --run-id <run_id>
python -m trend_analyzer run-all --config config/research.yaml
```

Each stage validates input versions, fails clearly when prerequisites are absent, and writes a completion manifest. Downstream stages reuse saved inputs. A feature-definition change invalidates models and predictions; a slippage change requires only backtesting and reporting.

Configuration covers universe, data dates, adjustment conventions, feature windows, horizons, split boundaries, retraining frequency, model settings, signal thresholds, starting capital, and costs.

## 11. GitHub reuse strategy

These sources were reviewed at the repository/documentation level. Their code has not yet been audited or tested against this project's requirements.

| Source | Intended use | Review before adoption |
|---|---|---|
| [yfinance](https://github.com/ranaroussi/yfinance) | Market-data adapter | Adjustment defaults, action fields, retries, historical consistency |
| [ta](https://github.com/bukosabino/ta) | Technical indicator functions | Warmup behavior, fill options, indicator definitions |
| [XGBoost](https://github.com/dmlc/xgboost) | Direction classifiers | Pin compatible release and use reproducible configuration |
| [PyBroker](https://github.com/edtechre/pybroker) | Candidate backtesting framework | Execution delay, portfolio accounting, cost application, actions |
| [skfolio WalkForward](https://skfolio.org/generated/skfolio.model_selection.WalkForward.html) | Validation reference or reusable splitter | Correct mapping from session dates to multi-ticker rows and label intervals |
| [Predictive stock trading with XGBoost](https://github.com/Seeevaa/predictive-stock-trading-xgboost) | Example architecture | Verify actual implementation, assumptions, license, and tests |
| [Machine-learning stock backtesting](https://github.com/handiko/machine-learning-stock-backtesting) | Example workflow | Market-specific assumptions and validity of evaluation |

Prefer installing focused libraries over copying entire repositories. Check licenses before incorporating source. A framework's walk-forward feature does not automatically protect our custom labels from overlap leakage.

## 12. Implementation milestones and review gates

| Milestone | Deliverable | Completion gate |
|---|---|---|
| A — Foundation | Package, configuration, CLI, dependency lock | Commands and run manifests work |
| B — Data | 10 instruments plus SPY in SQL, quality report | Ingestion and corporate-action checks pass |
| C — Daily research | Features, daily target, splits, baselines, XGBoost | Timing/leakage checks and held-out predictions pass |
| D — Daily simulation | Portfolio ledger, costs, benchmark report | Accounting examples reconcile |
| E — Full scope | 50 instruments and 1/5/21-session models | All horizons complete walk-forward evaluation |
| F — Final evaluation | Frozen final-test run, sensitivity report, README | Reproducible run from saved inputs |

At milestone D, inspect the full pipeline before adding scope. Poor predictive results trigger investigation and transparent reporting, not repeated tuning against the final holdout.

## 13. Verification priorities

Use targeted tests where an error could invalidate the research:

1. Feature causality and correct per-ticker rolling behavior.
2. Entry/exit target alignment for all horizons.
3. Date-wide splits and label-interval purging.
4. No fit or threshold selection on evaluation data.
5. Buy/sell timing and exact slippage arithmetic.
6. Shared cash constraints, daily equity reconciliation, corporate actions.
7. Deterministic tiny end-to-end run using fixed local fixtures.

Use a small download smoke check separately from deterministic tests; external market-data availability should not determine whether accounting tests pass.

## 14. Risks and scope boundaries

- Five years offers limited independent long-horizon observations. Prefer additional history and avoid interpreting stacked rows as independent evidence.
- Vendor corrections, missing histories, and split inconsistencies can change results. Preserve data provenance and quality exclusions.
- Choosing today's successful tickers biases the research universe; label this limitation explicitly.
- Repeated feature and parameter experiments can overfit development data. Keep a trial log and an untouched final evaluation.
- Directional accuracy and trading profitability measure different things. Report both, including costs and exposure.
- A research backtest omits some real execution effects, such as market impact and uncertain opening fills. The stated fixed-slippage scenarios define this version's simulation scope.

Deferred: live brokerage integration, intraday data, short selling, leverage, alternative data, deep learning, portfolio optimization, and a dashboard. These can be considered after the core results are reliable.

## 15. Expected review outcome

Review the proposed market universe, 1/5/21-session timing, long/cash sleeve allocation, price/corporate-action conventions, and final-test policy. Once those choices are accepted, implement milestones A through D first, then expand through E and F.
