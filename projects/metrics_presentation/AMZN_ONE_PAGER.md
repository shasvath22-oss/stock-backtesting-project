# AMZN - Prediction process and research results

**Experiment:** `20261024T205450515026Z` | **Initial capital:** $100,000 | **Final-test phase:** January 2, 2025 onward, with data through October 23, 2026 | **Slippage:** 10 bps (0.10%) per buy and sell.

## How the model works

1. Create past-only inputs from returns, moving averages, RSI, volatility, volume and SPY market context.
2. Define the historical answer as a positive dividend-inclusive return from the next open to the open after 1, 5 or 21 trading sessions. Slippage is excluded from this prediction target; unavailable future outcomes remain missing.
3. Train separate XGBoost models using historical inputs and completed answers. Future outcome columns never become prediction inputs.
4. Score direction at a 50% probability threshold. Trade at a stricter 55% threshold, entering at the next open when position rules permit and exiting at the scheduled horizon open. A 60% probability estimates a chance of a rise, not a 60% price increase.

## Evaluation design

Initial training: 2020-2022. Development validation: 2023-2024. Final test: 2025 onward. Expanding-window models refit every 63 sessions using outcomes completed by the training cutoff. Matured earlier test outcomes can enter later scheduled refits. Validation targets cannot cross into the final-test phase. Select model changes using development validation rather than repeatedly tuning to final-test results.

## Five key metrics - final test, 10 bps per side

| Metric | 1 session | 5 sessions | 21 sessions |
|---|---:|---:|---:|
| Scored predictions | 452 | 448 | 432 |
| **1. Prediction accuracy** | **48.89%** | **50.22%** | **63.43%** |
| Always-up accuracy | 50.66% | 52.01% | 54.86% |
| **2. Improvement over baseline** | **-1.77 pp** | **-1.79 pp** | **+8.56 pp** |
| **3. Strategy Sharpe** | **-0.47** | **0.05** | **0.56** |
| **4. Maximum drawdown** | **-33.59%** | **-26.26%** | **-31.29%** |
| **5. After-cost CAGR** | **-12.78%** | **-2.60%** | **+14.06%** |

**SPY comparison on matching dates and costs:** Sharpe 0.99; maximum drawdown -18.72%; CAGR +16.30%.

Accuracy is the share of correct directional forecasts. Baseline improvement is measured in percentage points (pp). Sharpe measures annualized return relative to daily variability with a zero risk-free rate. Maximum drawdown measures the deepest decline from a portfolio peak. CAGR is compound annual growth, not the return in each individual year. Prediction accuracy and trade win rate are different.

**Conclusion:** The 21-session model beat always-up directional accuracy but lagged SPY on after-cost CAGR and Sharpe and suffered a deeper drawdown. The shorter-horizon strategies had negative CAGR. Better classification does not automatically produce better investment results.

Results are historical simulations, not actual broker trades. Overlapping outcomes are dependent; statistical significance is not established. Preserve all horizons and 5/10/20 bps cost scenarios when reviewing results.

**Evidence:** accuracy.csv, performance.csv, splits.json, manifest.json, predictions.parquet and backtests/ under the experiment folder. Saved verification passed 16/16 checks; this validates pipeline integrity, not predictive significance or future profitability.
