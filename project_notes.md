# UI separation and data-source policy

Current rollout starts with **SPY and QQQ only**, using yfinance. The research UI
creates this two-instrument universe by default, with SPY also the nonduplicated
benchmark. The 50-symbol capability and prior validation remain available for later
expansion; old artifacts retain their saved universe. New experiments must be created
to use the starter defaults. config/research.json owns the `symbols` setting.

Current launch commands: original swing UI = `streamlit_app.py --server.port 8501`;
new full research UI = `research_app.py --server.port 8502`, both with
`python -m streamlit run`. Launch scripts `run_swing.ps1` and `run_research.ps1`
set the working directory and explicit ports. There is no combined Workflow menu.
Supported UI/CLI collection is yfinance-only. Synthetic generation remains only as
an internal test fixture; old synthetic runs are hidden from supported interfaces.

# Full research requirements ? current implementation

Updated October 24, 2026. The default app now provides the full multi-stock,
multi-horizon research pipeline; the swing workflow remains available via the
Workflow selector. See project_checklist.md for the six-requirement coverage
and validation record. Follow the six research sidebar stages explicitly. Dates,
SQL lineage, input schemas, chronological partitions and mature-label cutoffs are
saved and inspectable. Historical and synthetic runs are clearly distinguished.

The prior swing implementation plan is preserved below as reference.

---

# Swing-first research: incremental implementation plan

New direction: October 23, 2026. The previous plan is preserved in old_notes.md.
Period update (October 24, 2026): old/development = 2020-01-01 through 2024-12-31;
new/evaluation = 2025-01-01 through the latest completed saved session. Definitions
are in config/periods.json.
These are defaults: sidebar controls expose both starts and inclusive ends, with
an optional latest-completed endpoint for the new period. Apply validates ranges
and prevents overlap; choices persist in the session and are saved with new outputs.
Results retain their saved period definitions. A rerun is required to apply changes.
Statistics freeze outcomes within their event's period;
unfinished old outcomes are excluded instead of observing new-year prices.
Backtests restart capital, close at each boundary, and require confirmation inside
the selected period for new entries. Confirmed older pivots remain usable as context.
Saved results record these definitions; previous full-history artifacts are not migrated.

Build and understand one stage at a time with one US stock.
Navigation: a persistent left sidebar lists Prices, Swings/Structure, Statistics/
Combinations, Trading Backtest and XGBoost filter. Only the selected
stage renders. Price and swing snapshots are selected on the left; missing inputs
show prerequisite instructions. Trading access does not depend on statistics output.

Each stage must explain
its inputs, rules, outputs and timing in English. Later stages are planned,
not automatically executed. Prices, confirmed swings, structural labels and initial
wick-based descriptive outcome counts are implemented.

## 1. Prices ? implemented
Choose one ticker, date range and historical/generated source. Save immutable raw
and validated daily OHLCV snapshots, missing-session checks, source and input hash.
Inspect candles and individual sessions. Never fill missing prices silently.
Historical OHLC uses the vendor split-normalized convention; distributions are separate.

## 2. Confirmed swings ? next
Start with configurable left/right candle counts (proposed: 2/2).
Latest user clarification: minimum movement is 0.386% of the last accepted pivot
PRICE, not a 38.6% retracement of the prior leg. Decimal threshold = 0.00386.
Opposite move / last accepted pivot price must be >= 0.00386. From $100, a
$0.20 move is ignored; $0.386 qualifies in either direction, subject to confirmation.
First pivot seeds the reference. More extreme same-type pivots extend the endpoint;
ignored minor moves never change the reference. Keep immutable confirmation events.
Old v1 analyses retain the previous rule and must be rerun to use the v2 rule.
Define strict
comparisons and equal-price handling. Keep both pivot_date and confirmed_at; signals
cannot use a pivot before confirmation. Missing-session gaps must invalidate affected
windows. Define how to handle consecutive same-type pivots and outside bars before
assigning structure. A percent/ATR reversal detector may later be a separate method;
its confirmation timing must also be explicit. Never revise a historical live signal
using later, unobserved prices.

## 3. Structure
Implemented labels: H/L seeds, HH/HL/LH/LL, EH/EL. Break signals are not implemented.
Classify confirmed same-type pivots into HH/HL/LH/LL, with explicit tie and initial
state rules. Define a break (close or wick, reference level, confirmation). Track
state using only confirmed information; show each transition on the chart.

## 4. Descriptive statistics before ML
Implemented as UI Step 3: first strict high/low wick crossings after confirmation,
with configurable observation horizon (default 20 sessions). Freeze the latest
confirmed upper/lower endpoints in the event's segment. Count newly accepted
HH/HL/LH/LL events, not later extensions of an existing endpoint. Start observing
the next session. Skip events whose frozen levels were already broken between
pivot and confirmation. Equal-price touches are not breaks. Same-candle dual
crossings are ambiguous. A full horizon without crossing is no_break; an unfinished
history without crossing is incomplete_history. Gaps/invalid levels are exclusions.
Percentages include upper_first, lower_first, both_same_candle and no_break in the
completed denominator; show exclusions separately. Save settings, source hashes,
and the full outcome audit; provide a selected-event chart in English.
Return/excursion measurements and uncertainty estimates below remain future work.
Implemented combination analysis: join the immediately preceding confirmed endpoint
label to each new accepted HH/HL/LH/LL using the as-of state at its confirmation.
Group by previous-label → current-label; report the same completed denominator and
separate exclusions as single-label statistics. Do not skip over seed/equal labels,
bridge gap segments, or use a later extension to revise an event's earlier context.
The UI provides both summary CSV and pair-specific chart examples. Old saved
statistics stay readable in single-label mode and require an explicit rerun for pairs.

Declare an event precisely, e.g. confirmed HH followed by confirmed HL. Define the
outcome BEFORE counting: continuation/break by a maximum observation horizon, with
neither/ambiguous outcomes handled explicitly. Measure counts, event frequency,
continuation/break rates, return distributions and favorable/adverse excursion.
Report sample sizes and dependence: overlapping events are not independent trials.
Use time blocks for uncertainty estimates and regime breakdowns. No significance or
predictive-value claim based solely on in-sample percentages.

## 5. Executable baseline rules
Implemented as UI Step 4: fixed long HH → HL rule.
Results are displayed in separate sidebar pages Step 4A (old) and Step 4B (new), labeled with their saved
date ranges (defaults: through 2024 and January 2025 onward). Each page has independent
cost controls, profit/return/trade-count metrics, charts, audits and period-prefixed
CSV downloads. Both pages remain visible even if one period lacks data. Page changes
and display filters do not run simulation.

Next-open entry only strictly
inside the fixed HL stop and prior-high target. Orders activate on entry day;
touches execute, with stop first when both levels are touched. Gap-down stop fills
at open, gap-up target at target; adverse slippage applied afterward. Full-cash,
single-position long/cash; no same-day reentry after an existing position exits.
Default maximum 20 sessions counting entry as session 1, exit at that close;
liquidate remaining holdings at final data close. Eligible dividends included.
Costs 5/10/20 bps reuse one signal list; buy-and-hold shares dates/capital/costs.
Inputs require no missing NYSE sessions. This stage uses previously explored history,
not a pristine holdout; parameters and the full audit are saved for later comparison.

Fix entry at the next available open after confirmation. Define stop, exit, maximum
holding time, one-position rule, capital and dividends. If daily bars cross both
stop and target, use an explicit conservative convention or obtain finer data.
Compare cash, buy-and-hold and a simple structural rule over identical dates.

## 6. Statistics and XGBoost together
Implemented in UI Step 5: fixed 40-tree, depth-2 XGBoost, learning_rate 0.05,
seed 42. Eight confirmation-close features; independently labeled old HH/HL trades
at 10 bps with the same stop/target/max-hold rules. Exclude invalid entries and
unmatured boundary outcomes. Last 25% of chronological old examples are validation;
purge earlier examples with exits at/after validation start. Compare validation
accuracy/Brier with a constant prior baseline. Refit all matured old examples and
freeze a 0.55 threshold for new-period prediction. Compare new-period rule, filter
and buy/hold at 5/10/20 bps, reusing probabilities. Minimum 20 old examples, 10
purged training examples and both classes; show small-sample warnings. Store all
inputs, schema, models, validation, predictions, settings and portfolio audits.
No calibration, automated tuning, test-based threshold selection or live trading.
Implemented UI Step 6: retrospective selected/rejected signal audit. Reuse saved
features/predictions and model dates, threshold and holding settings; no refit.
Simulate each candidate with identical standalone capital at 5/10/20 bps, preserve
execution skips and incomplete histories, and separate missing inputs from rejected
predictions. Compare with the saved model/baseline portfolio's actual execution and
P&L. Display a four-way selected/rejected versus profit/loss count plus all excluded
categories, candidate chart, CSV and immutable settings/hash audit. Standalone
outcomes can overlap and are never presented as additive portfolio results.

Statistics establish the simple conditional baseline, e.g. an event's continuation
rate. XGBoost may combine confirmed swing size/duration, HH/HL state, distance to
confirmed levels, volatility and later market context. It learns an explicitly
chosen future outcome; structure features do not eliminate the need for labels.
Use chronological evaluation and purge training examples whose outcomes have not
finished. Tune only on development windows. Keep a final later test; compare model
probabilities and the resulting strategy with simple statistical/rule baselines.
Keep confirmation timestamps, feature schema and all parameters with each experiment.

## 7. Cost sensitivity and reporting
Baseline: adverse execution slippage of 10 bps per side. Buy at open*(1+bps/10000);
sell at open*(1-bps/10000). Reuse IDENTICAL predictions and signals at 5, 10 and 20
bps per side. Add an optional gross 0 bps reference. Do not retrain per cost case.
Report net return, drawdown, trade count, turnover, expectancy and benchmark results.
A 10 bps buy plus 10 bps sell is roughly 0.20% round-trip friction, not exactly a
fixed 0.20% deduction for every return. Show arithmetic from actual fills.

## Flexibility
Expose only the controls needed for the implemented stage. Save every effective
setting in a new run; never silently change an existing experiment. Future controls:
pivot method/window, minimum swing size, break definition, event/outcome horizon,
entry/stop/exit, model settings and costs. Changing rules after seeing test results
creates a new experiment and does not make the reused test untouched.

## Run
python -m streamlit run streamlit_app.py

The old application remains available via python -m streamlit run legacy_app.py.
Old artifacts and config/research.json belong to the legacy pipeline; the new
step-1 snapshots live in data/swing_runs. The new workflow does not train models yet.
