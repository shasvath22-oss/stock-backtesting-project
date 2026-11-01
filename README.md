# Stock Backtesting Project

By Shasvath Suresh

I got interested in the stock market and wanted to understand why stock prices
move up and down. That led me to wonder whether past prices could help predict
what happens next. I started this project to try it out using XGBoost and compare
it with simpler models. I also wanted to see whether the predictions could lead
to better returns after trading costs.

The project uses Python, SQLite and Streamlit. It has a model research app and a
separate app for studying swing highs and lows. This is a learning project using
historical data, not a live trading system.

## How to run it

You need Python 3.11 or newer. From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m streamlit run research_app.py --server.port 8502
```

Open the local address printed in the terminal. Enter a ticker such as AMZN and
choose **Run complete analysis for this ticker**. The app downloads prices through
yfinance, so it needs an internet connection. SPY is also downloaded for comparison
and market context. Downloads can fail; the app reports the error instead of
substituting generated prices.

To open the swing app instead:

```powershell
python -m streamlit run streamlit_app.py --server.port 8501
```

The launch scripts `run_research.ps1` and `run_swing.ps1` run these same commands.
The optional [results viewer](projects/metrics_presentation/README.md) runs separately.

## What the code does

The pipeline downloads daily prices and checks them before saving a snapshot in
SQLite. It builds 37 features from past prices, volume and SPY data, then compares
XGBoost with logistic regression and simple baseline predictions. Forecasts cover
1, 5 and 21 trading sessions.

Dates stay in order during training and evaluation. Models refit every 63 sessions
using only outcomes that have finished by the training cutoff. During the test
period, later refits can use completed outcomes from earlier test dates.

The model predicts the probability of a positive return from the next opening price
to the opening price at the end of the forecast horizon. A 50% threshold is used
for prediction accuracy. The trading simulation uses a 55% entry threshold and
holds cash when there is no entry. It includes slippage on purchases and sales.

## Database notes

My mentor, a Data Warehouse Architect at New York University, guided me on database
organization and avoiding duplicate price records. The price table uses
`(snapshot_id, ticker, date)` as its primary key. This allows the same stock and
date to appear in separate experiments without duplicating it within one snapshot.

When collection is rerun, the code replaces only that snapshot's rows inside a
transaction. Other snapshots remain available. The implementation is in
[trend_analyzer/data.py](trend_analyzer/data.py). Data-quality checks are in Python;
the table does not currently have explicit NOT NULL or CHECK constraints.

The guidance also helped me separate prediction accuracy from investment results.
My short [review notes](mentor_notes.md) record the main points.

## My AMZN results

The saved AMZN experiment is `20261024T205450515026Z`. It uses SPY as a benchmark.
Initial training covers 2020-2022, validation covers 2023-2024, and the test period
runs from January 2, 2025 through October 23, 2026.

The 21-session XGBoost model had 63.43% test accuracy, compared with 54.86% for
always predicting up. Its test AUC was 0.670, but its validation AUC was only 0.489.
That difference is a reason to be cautious about the result.

These are cumulative simulated test returns with 10 basis points (0.10%) of
slippage on each side of a trade:

| Strategy | Return | Sharpe ratio | Largest drawdown |
|---|---:|---:|---:|
| XGBoost, 1 session | -21.84% | -0.473 | -33.59% |
| XGBoost, 5 sessions | -4.63% | 0.048 | -26.26% |
| XGBoost, 21 sessions | +26.74% | 0.564 | -31.29% |
| Logistic regression, 21 sessions | +49.31% | 0.828 | -29.93% |
| AMZN buy-and-hold | +11.61% | 0.347 | -30.88% |
| SPY buy-and-hold | +31.27% | 0.987 | -18.72% |

The simpler logistic regression model did better than XGBoost in this experiment.
The shorter-horizon XGBoost strategies lost money. This helped me understand why
prediction accuracy alone is not enough to judge a strategy. Sharpe here uses a
zero risk-free rate; drawdown measures the drop from a portfolio peak.

The optional viewer includes [copies of the saved experiment tables](projects/metrics_presentation/artifacts/).
New runs save their own results locally; they will not necessarily match this run
because the dates and downloaded data can change.

## Files in this project

| Path | Purpose |
|---|---|
| `trend_analyzer/` | Data collection, features, models and backtests |
| `research_app.py` | Model research interface |
| `streamlit_app.py`, `swing_app.py` | Swing analysis interface |
| `config/` | Research settings and ticker list |
| `tests/` | Automated checks |
| `projects/metrics_presentation/` | Separate viewer for saved results |
| `data/`, `artifacts/` | Locally generated data and experiment outputs |

Local downloads and generated outputs are excluded from Git. The original
`legacy_app.py` is kept for reference. [longer notes](how_it_works.md)
contains longer notes, including older workflows; some of its commands are outdated.
The project notes include older ideas, so use the setup commands above to get started.

## Running from the terminal

```powershell
# Run the historical pipeline with config/research.json
python -m trend_analyzer run-all

# Run the tests
python -m pytest -q
```

The default configuration uses SPY and QQQ. `--full` uses the configured larger
universe. Change settings before creating an experiment; each run saves its settings.

## What I still need to check

These are historical simulations. The AMZN example does not show that the method
works for other stocks. Longer forecast horizons overlap, so their observations
are not independent. The current stock list also leaves out companies that did
not survive to today. Statistical significance has not been established.

Some dates have already been explored, so creating another run does not make them
an untouched test set. I would like to test on new data and examine how stable the
results are before drawing stronger conclusions.

My next steps are:

- Run the same experiment on other stocks to see whether the AMZN result holds up.
- Compare results across different market periods instead of relying on one overall return.
- Check how much the results change when trading costs increase.
