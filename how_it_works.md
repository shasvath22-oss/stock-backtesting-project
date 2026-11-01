# Workflow reference

Detailed run instructions and notes from earlier versions. Some sections describe
legacy or demo workflows; use the main README for the current starting points.

## Main programs

- `streamlit_app.py` is the main entry point for the swing analysis interface (port 8501). It loads `swing_app.py`.
- `research_app.py` runs the complete multi-horizon machine-learning research workflow (port 8502).

```powershell
.\run_swing.ps1
.\run_research.ps1
```

The metrics and presentation viewer has been extracted into its own project at
[`projects/metrics_presentation/`](projects/metrics_presentation/README.md).
It is optional and is not launched by either main program.

**Research UI: enter one stock ticker.** On port 8502, choose a ticker such as
AAPL or MSFT and click **Run complete analysis for this ticker** to collect data,
build features, split dates, train, predict, backtest, and generate a report.
SPY is downloaded as market context and a buy-and-hold benchmark, and is not
added to the selected stock's training universe. Choosing SPY itself downloads it
only once. **Create research experiment** retains the manual step-by-step option.
The CLI's defaults in `config/research.json` still use SPY and QQQ;
the larger `config/universe.csv` remains available for a later explicit `--full`
CLI run. Existing 50-instrument artifacts are retained with their original settings.

The original swing interface stays on **http://localhost:8501**:

```powershell
python -m streamlit run streamlit_app.py --server.port 8501
```

Open a second terminal for the new full research interface on **http://localhost:8502**:

```powershell
python -m streamlit run research_app.py --server.port 8502
```

Alternatively run `.\run_swing.ps1` and `.\run_research.ps1` in separate terminals.
The apps use separate entry points and browser sessions; there is no Workflow menu
replacing the original UI. Saved swing data and research artifacts remain in their
respective folders. Both supported interfaces use yfinance historical data only.

# Full market research pipeline - separate application

The separate `research_app.py` interface implements the full multi-horizon workflow.
The original `streamlit_app.py` entry point opens the swing UI.

Follow the left sidebar in order. No download or training happens from changing a filter.

| Step | Action | Saved output |
|---|---|---|
| 1. Collect + SQL | Create experiment, then collect 50 instruments plus SPY | raw/clean Parquet, coverage/quality, SQLite snapshot |
| 2. Features | Build trailing indicators, candle rules and future targets | features.parquet + explicit input schema |
| 3. Validation splits | Declare initial training, validation and final test; expanding folds | splits.json with per-fold training cutoffs |
| 4. Models + predictions | Train, then predict separately; horizons 1/5/21 sessions | models, probabilities, accuracy/baseline tables |
| 5. Trading backtest | Next-open entries/exits, long/cash sleeves; identical cost cases | daily equity/returns, positions, trades, SPY comparisons |
| 6. Risk + report | CAGR, annualized volatility, Sharpe, drawdown, costs; integrity checks | performance.csv, HTML report, verification results |

**Historical mode** requests at least five years, defaults to 50 configured tickers,
and requires 1,260 valid sessions per eligible instrument. SPY is an additional
benchmark. Missing downloads block progress; incomplete histories are explicitly
excluded. Prices are vendor split-normalized OHLC; dividends/capital gains are
separate cash distributions. `Adj Close` is retained but never an execution price.
The present-day universe has survivorship bias and vendor adjustments can be revised.

**Data policy:** the supported apps and CLI use Yahoo Finance via yfinance only.
There is no synthetic/demo selector or automatic fallback. Earlier generated runs
are hidden from these apps. Deterministic generated/vendor fixtures remain only for
automated tests; they are not market results. Historical connection failures stop
collection and show an error.

Research defaults: old/development 2020?2024, new/final test 2025 onward. Initial
training is the first 756 eligible old-period sessions; remaining old dates are
out-of-sample validation. Expanding-window models refit every 63 sessions, admitting
only outcomes completed at the previous close; scheduled test refits can use matured
earlier test labels. Demo uses shorter declared windows. Validation labels and
trades cannot extend into the final test. Overlapping observations are dependent.
Settings/thresholds are fixed, not tuned on the test; previously explored dates do
not become untouched simply because an experiment is recreated.

Storage: `data/market.sqlite`, table `price_bars`, primary key
`(snapshot_id, ticker, date)`; reusable stage artifacts in `artifacts/<run_id>/`.
Each manifest saves its effective settings, versions and stage state. An upstream
rerun invalidates later completion flags. Reports and the UI show whether a run is
historical or synthetic. See [project_checklist.md](project_checklist.md)
for implementation and verification coverage.

## Existing swing workflow reference

# Swing research ? start here

The project is restarting with swing structure, one stock and one step at a time.
Default research periods are in `config/periods.json`: old/development is
01-Jan-2020 through 31-Dec-2024; new/evaluation is 01-Jan-2025 through the latest
completed session in the saved snapshot. Step 1 defaults to January 2020 through
today (end excluded, so an incomplete current session is not downloaded).
Change **Old start**, **Old end**, **New start** and **New end** under
**Research periods** in the left sidebar, then click **Apply periods**. Leave
**New end: latest completed session** checked for a moving latest endpoint, or
uncheck it to use a fixed end date. End dates here are inclusive. Periods may have
a gap but must not overlap. Applied choices persist in the current session and
are copied into each new snapshot/statistics/backtest; default settings remain
unchanged for a new session. Changing dates does not download or recalculate.
Saved outputs display their own recorded periods; rerun Step 3/4 to use new choices.
Steps 3 and 4 save separate old/new results with a period selector. Observation
windows do not cross the split; backtests reset capital and close positions at
each period's final close. Older swings can provide causal structural context,
but new trades require a signal confirmed inside the new period. Existing shorter
snapshots cover only part of these ranges: collect again from your selected start date for full coverage.
Previously saved full-history results remain unchanged; rerun the stages to obtain
split results. A date split does not make already-explored data an untouched test.

```powershell
python -m streamlit run streamlit_app.py
```

**Step 1: Prices** and **Step 2: Confirmed swings** are implemented. Choose a US
stock or ETF (default QQQ), dates and source, then click **Run step 1**. Historical prices
need internet. Synthetic demo works offline and is clearly marked as generated.
Inspect the candle chart, individual sessions, quality checks and saved CSV.
Changing a chart control does not download data again. Each run saves a new snapshot
under `data/swing_runs/`.

After saving prices, click **Run step 2**. Configure left/right pivot windows
(default 2/2). The minimum opposite price change is 0.386% (decimal 0.00386)
of the last accepted pivot price, per the user's clarified choice. This is not a
Fibonacci leg retracement. The first pivot seeds the reference. More extreme
same-type pivots extend a swing; each update retains
its confirmation timestamp. Inspect HH/HL/LH/LL and EH/EL labels, ignored-event reasons,
and purple pivot versus orange confirmation markers. The as-of selector reconstructs
only what was known by that close. Every analysis saves its own settings and audit.

Step 3 now counts the first strict wick crossing after a new structural event is
confirmed. Choose an observation horizon (default 20 trading sessions), then run
the stage explicitly. Levels are frozen from the latest confirmed high/low at the
event's confirmation. Inspect an example with levels, candles and break date.
Same-candle dual crossings are ambiguous; exact touches are not breaks. Already-broken
levels, missing data and unfinished outcomes are separately excluded from percentages.
No-break and ambiguous outcomes remain in the completed-event denominator. Counts
are descriptive; overlapping samples are not independent trials. Each run saves its
settings and outcome audit under the selected swing analysis.
Step 3 also groups outcomes by the immediately preceding confirmed structure label:
for example HH → HL versus LH → HL. Combination context uses only endpoint labels
known at the new swing's confirmation, including earlier confirmed extensions;
later revisions cannot alter an older event. Seeds, equal labels and gap boundaries
do not create artificial two-label pairs. View combination counts, exclusions and
percentages, download the summary, or select a pair to inspect both pivot labels
and the subsequent wick break on its chart. Existing older statistics must be rerun
with **Run step 3** to add combination fields; chart filters never rerun the stage.

Choose **Step 4A: Old-period Backtest** or **Step 4B: New-period Backtest** in the left sidebar; statistics results
are not required to access that page. Steps 1–5 are
always listed on the left. Each step renders only its own controls, with shared
saved price/swing selectors in the sidebar. Missing prerequisites show instructions.
Step 4 implements a first long/cash HH → HL rule.
Results have two explicit sidebar pages: **Step 4A** (default 2020–2024) and
**Step 4B** (default January 2025 onward). Displayed dates reflect the saved run's
configured periods. Each has its own 10 bps default selector, net profit/return,
trade count, curve, trade explanations and period-prefixed CSV download. Switching
pages or costs never runs another backtest. A single Run saves both period results.
If a period has no data, its page remains
visible with instructions to collect the missing dates and rerun.

The rule:
enter next open only when it lies between the fixed HL stop and previous-high target. Stop/target wick touches
can execute on the entry candle; if both touch, the stop executes first. Gap-down
stops fill at the open; gap-up targets fill at target, before adverse slippage.
Exit at the holding-limit close (default session 20 counting entry as session 1),
or the final data close. One full-cash position at a time, no leverage or same-day
reentry after an earlier position exits; eligible distributions are credited.
The same confirmation-time signals are simulated at 5/10/20 bps per side, with a
buy-and-hold comparison on identical dates. Prices must have no missing sessions.
Saved settings, signals, trades, skipped-entry audits and equity are inspectable.
These historical results reuse data already explored; no untouched evaluation is claimed.

Step 5 implements a fixed XGBoost profitability filter. Eight confirmation-time
inputs describe returns, volatility, stop/target distances, range, duration and
relative volume. Label each old-period HH → HL candidate independently with the
same execution rules at 10 bps, excluding non-executable and unfinished outcomes.
Train on the first 75% of old examples, validate on the last 25%, purging training
labels that have not exited before validation begins. Report accuracy and Brier
against a constant training-profit-rate baseline. Refit on all matured old examples;
freeze for new-period predictions at a fixed 0.55 selection threshold. Compare all
signals, model-filtered signals and buy-and-hold at 5/10/20 bps without retraining
per cost. Training requires at least 20 examples and both classes; small/overlapping
samples remain a research limitation. Previously explored new dates are not pristine.
Each `model_runs/<id>/` saves the model, feature schema, input examples, validation,
probabilities, settings/hashes, performance, equity, trades and entry audits.
Step 6 audits saved model decisions without retraining or changing predictions.
At the saved new-period dates, simulate every HH/HL candidate independently with
the same fixed capital, stop/target/holding rules and selected cost. Show selected
profits/losses, rejected profits/losses, missing-input cases, invalid next-open entries
and unfinished outcomes separately. Independently simulated P&L must not be summed
as portfolio profit; actual model and baseline portfolio P&L have separate columns.
Select a decision to inspect its probability, outcome and candle chart. Export CSV
or reopen saved `model_runs/<id>/signal_audits/<id>/` settings and audit parquet.

Next come return/excursion statistics and uncertainty. Baseline slippage is 10 bps per side, with identical signals
retested at 5/10/20 bps. See [project_notes.md](project_notes.md).

The previous application and results remain available for reference:
`python -m streamlit run legacy_app.py`. Its commands and configuration below apply
to the **legacy pipeline**, not the new incremental swing workflow.

---

# Trend analyzer

A local research pipeline and Streamlit dashboard for predicting 1, 5, and 21-session stock/ETF direction. Each stage saves reusable artifacts. Start with the synthetic demo to check the workflow; its results are not evidence of market performance.

## Setup

Python 3.11+ is required. From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m trend_analyzer demo
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Open the localhost address printed by Streamlit. If you are using the existing Python environment in this workspace, `python -m trend_analyzer demo` and `python -m streamlit run streamlit_app.py` also work with the project-local `.deps` packages.

## UI workflow

### Learn with one stock

In the sidebar, leave **Learn with one stock (AAPL)** checked and click **Create run**.
The **Learning walkthrough** starts with a setup lesson showing installation commands,
data-source choices and the exact saved run settings. Continue into seven execution
lessons. Run step 1 to inspect prices,
then select lesson 2 when ready. Each lesson runs only its own stage and displays
the saved output with a plain-language explanation. Later lessons stay locked until
their prerequisites are complete. Refreshing or changing lessons does not run work.
Each lesson explains its inputs, operations and output before execution. Inspect
individual decision dates to see entry/exit prices, dividends and the calculated
target; inspect predictions and trade fills to connect probabilities to outcomes.
Use **Continue to step** when ready. **Run again** repeats a stage and invalidates
later steps so their results must be refreshed.
The training lesson lets you inspect each window's actual input columns and sample
rows alongside their historical answers before training. It explains row selection,
XGBoost settings, logistic regression, baselines, model saving and later evaluation.
Setup and each lesson show a flowchart with the current step highlighted, alongside
an invented AAPL example. Switch between profit, loss and no-purchase scenarios to
see how a probability leads to an action and how execution costs affect cash.

This mode trains on AAPL only and predicts a single 5-session horizon. SPY is also
collected for market context and comparison. Leave **Synthetic demo** checked for
offline practice; uncheck it for actual historical AAPL prices. Generated demo
prices do not represent Apple's history. Other research views remain available.

For the same settings from the CLI, create a run with
`python -m trend_analyzer init --learning`, then run individual stages with its run ID.
`python -m trend_analyzer demo --learning` executes all stages offline.

1. Create a synthetic or historical run in the sidebar. Historical runs use 10 instruments by default; select all 50 for the full universe.
2. Choose **Pipeline**, then **All remaining steps**, or execute one step at a time.
3. Inspect **Data quality**, **Model evaluation**, **Portfolio**, and **Trades and positions**.
4. Filter by horizon, model, development/test period, and transaction cost. Export trade/position CSVs and the HTML report.

The UI never downloads or retrains merely because a chart filter changes. It runs a stage only when its form is submitted. Keep the app open while a stage runs. For long runs, use the CLI and refresh the UI after completion.

## CLI workflow

```powershell
# Offline end-to-end example with generated prices
python -m trend_analyzer demo

# Full historical pipeline: 10 instruments plus SPY
python -m trend_analyzer run-all

# Full configured universe: 50 instruments plus SPY
python -m trend_analyzer run-all --full

# Or create a run, then use the printed run ID for each stage
python -m trend_analyzer init
python -m trend_analyzer collect --run-id <run_id>
python -m trend_analyzer features --run-id <run_id>
python -m trend_analyzer splits --run-id <run_id>
python -m trend_analyzer train --run-id <run_id>
python -m trend_analyzer predict --run-id <run_id>
python -m trend_analyzer backtest --run-id <run_id>
python -m trend_analyzer report --run-id <run_id>

# Train one horizon or rerun just one cost case
python -m trend_analyzer train --run-id <run_id> --horizon 5
python -m trend_analyzer backtest --run-id <run_id> --slippage-bps 20
python -m pytest -q
```

`config/research.json` controls history dates (end exclusive), universe size, minimum history, train/test splits, fold size, capital, threshold, and fixed XGBoost settings. Edit configuration before creating a run. Each run copies its effective configuration into its manifest. Do not tune settings based on the final test and claim it remains untouched; create a new research experiment and document the reuse.

## Saved results

- `data/market.sqlite`: SQL bars keyed by snapshot, ticker, and session date.
- `artifacts/<run_id>/manifest.json`: effective configuration, versions, stage status and input hash.
- `bars.parquet`, `coverage.csv`, `quality.json`: the input snapshot and quality checks.
- `features.parquet`, `feature_schema.json`: trailing features, eligibility, and future labels.
- `splits.json`: fixed date-wide development/test folds and training cutoffs.
- `models/`: trusted-local joblib model bundles. Never load joblib files from untrusted sources.
- `predictions.parquet`, `accuracy.csv`, `accuracy_detail.csv`: fold predictions and baseline comparisons.
- `performance.csv`, `backtests/`: 0/5/10/20 bps results, daily equity, trades and positions.
- `report.html`: standalone tabular report; interactive charts are in Streamlit.

## Research conventions

- A feature observed after close `t` predicts entry at open `t+1` and exit at open `t+h+1`. Weekly/monthly horizons mean 5/21 sessions; completed calendar-week/month returns are separate features.
- Models are pooled across eligible instruments, one classifier per horizon. Baselines are always-up, training majority, momentum, and logistic regression.
- The most recent 20% of feature-eligible dates is the final test. Expanding-window folds train only on rows whose label exits are known at the previous close. All tickers share date boundaries.
- Model parameters and the 0.55 trading threshold are fixed before testing. This first version deliberately has no automated tuning or early stopping. Retraining during the final test follows the predeclared fold schedule and can use only matured earlier labels.
- Features contain trailing returns, SMA/EMA distances/slopes, RSI, MACD, ATR, volatility, volume ratios, candle geometry, engulfing/doji/hammer rules, and SPY context. Missing sessions invalidate affected windows and restart warmup eligibility.
- Yahoo OHLC are treated as split-normalized prices. Quantities are split-normalized research units, not historical exchange share counts. Distributions are credited to positions held before the ex-date open. Splits are not applied again. `Adj Close` is retained but not used as an execution price. Vendor revisions and adjustment conventions remain a data-source limitation.
- Long/cash, equal initial capital sleeves, fractional normalized units, zero cash interest, no leverage, zero commission. Scheduled exits precede new entries at the same open; both sides incur slippage.
- Signals that cannot exit within the evaluation interval are skipped. The implementation fails on unresolved backtest price gaps rather than fabricating fills or stale marks.
- SPY and equal-weight buy-and-hold start/end on the same dates as strategies. Both incur entry/exit costs and include cash distributions. Gross 0 bps and 5/10/20 bps results reuse identical model predictions.
- Sharpe uses daily returns, 252 sessions/year, zero risk-free rate. CAGR uses actual elapsed time. Drawdown includes initial execution costs relative to initial capital.

## Scope and remaining research limitations

The implementation uses direct SQLite access and a small accounting engine instead of SQLAlchemy/PyBroker, keeping the first pipeline inspectable and avoiding an unverified framework adapter. Collection creates full immutable per-run snapshots rather than incremental vendor updates. The current universe has survivorship bias. Missing or failed downloads are reported; failed requested downloads block progression. Incomplete-history instruments are excluded using the configured minimum-session count.

Daily labels and market observations are dependent across tickers; 5/21-session targets also overlap in time. The reports do not claim statistical significance. Block bootstrap confidence intervals, bounded hyperparameter tuning, automated probability calibration, point-in-time universes, and live execution are future improvements. Directional accuracy can exceed a baseline while net trading returns underperform SPY.

Dependencies referenced: [yfinance documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [market calendars](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html). The original design is in [project_notes.md](project_notes.md).
