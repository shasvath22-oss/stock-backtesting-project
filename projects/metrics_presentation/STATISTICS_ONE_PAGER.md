# Stock Direction Prediction: Statistical Results Summary

**Experiment:** `20261024T192312985153Z` · **Stock:** AAPL · **Benchmark:** SPY · **Starting capital:** $100,000.
**Initial training:** Jan 2, 2020–Dec 30, 2022; **validation:** Jan 3, 2023–Dec 31, 2024; **final test:** Jan 2, 2025–Oct 23, 2026. Results below are saved historical results, not teaching examples.

## What was implemented

**Prediction:** Separate XGBoost models predict positive dividend-inclusive, open-to-open returns over **1, 5, and 21 trading sessions** (daily, approximately weekly and monthly). Trailing price/volume indicators and SPY context form the inputs. Models output probabilities; **50%** classifies direction for accuracy, while **55%** is the trading-signal threshold. Comparisons include always-up, training-majority, momentum, and logistic regression.

**Validation:** Chronological splits and expanding-window walk-forward refits every **63 sessions**. Training admits only observations whose target exits occurred by the preceding-close cutoff. Validation outcomes cannot cross into final test. Scheduled test refits may use completed earlier test outcomes. These controls prevent future-label leakage at fitting boundaries; overlapping evaluation targets still create statistical dependence.

**Backtesting:** Signals observed after close enter at the next open, hold for the horizon, then exit at an open. Long/cash, one position per stock, with daily equity, returns, positions and trade records. Slippage is applied adversely to buys and sells; baseline **10 bps (0.10%) per side**, with separate **5/10/20 bps** simulations using identical predictions. SPY uses matching portfolio evaluation dates and costs. Zero commission and cash interest; eligible dividends included.

## Statistics gathered and what they tell us

| Statistical question | Recorded metrics | Interpretation |
|---|---|---|
| Does the model predict direction? | Sample count, up fraction, accuracy, balanced accuracy, precision, recall, confusion counts | Accuracy measures correct directions; balanced accuracy accounts for class imbalance. Precision measures reliability of predicted rises; recall measures rises found. |
| Are the probabilities useful? | ROC AUC, log loss, Brier score | AUC measures ranking (0.5 = random ranking); lower log loss/Brier indicates better probability forecasts. These do not establish calibration by themselves. |
| Does the strategy make money after costs? | Net return, CAGR, daily returns/equity, closed trades, trade win rate | Return measures investment growth; CAGR annualizes it. Trade win rate differs from prediction accuracy. |
| What risks and costs accompany the return? | Annualized volatility, Sharpe, maximum drawdown and duration, exposure, holding time, turnover, slippage cost | Measures variability, risk-adjusted return, losses from peaks, market participation and trading intensity. Sharpe uses 252 sessions and zero risk-free rate. |

## Final-test evidence

**Trading columns use 10 bps per side.** N counts scored predictions, not independent trials or trades. SPY buy-and-hold: **+31.27%** return, **16.30%** CAGR, **0.99** Sharpe, **−18.72%** maximum drawdown.

| Horizon | N | XGBoost accuracy | Always-up accuracy | AUC | Net return | CAGR | Sharpe | Max drawdown | Closed trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Daily (1) | 452 | 47.57% | 53.76% | 0.440 | −40.27% | −24.88% | −1.04 | −42.91% | 224 |
| Weekly (5) | 448 | 52.68% | 55.58% | 0.513 | +24.72% | +13.04% | 0.58 | −20.93% | 71 |
| Monthly (21) | 432 | 55.09% | 65.28% | 0.530 | +19.56% | +10.43% | 0.49 | −30.50% | 19 |

**Cost sensitivity — total net return:**

| Slippage per side | Daily | Weekly | Monthly | SPY |
|---|---:|---:|---:|---:|
| 5 bps | −25.27% | +33.89% | +21.86% | +31.40% |
| 10 bps | −40.27% | +24.72% | +19.56% | +31.27% |
| 20 bps | −61.84% | +8.21% | +15.11% | +31.01% |

**Conclusion:** XGBoost did not beat always-up directional accuracy at any horizon. At baseline costs, all three strategies underperformed SPY on total return and Sharpe; daily trading was particularly cost-sensitive. Weekly return exceeded SPY at 5 bps, but not at 10 or 20 bps. These are descriptive results, not evidence of statistical significance: confidence intervals, dependence-adjusted hypothesis tests and calibration diagnostics have not been produced. The AAPL selection and vendor history also limit generalization.

**Evidence:** `accuracy.csv`, `accuracy_detail.csv`, `performance.csv`, `splits.json`, `manifest.json`, and the `backtests/` ledgers in the experiment folder. Preserve all horizons and cost scenarios rather than selecting only the best result.
