"""English presentation page with saved-result evidence."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from sample_data import sample_links

def show_presentation():
    st.header('How to present this project')
    st.caption('English explanation and results')
    with st.expander('Sample data links for every step', expanded=True):
        sample_links()
    st.subheader('1. What question does the project answer?')
    st.write('Can past stock-price information help XGBoost predict whether a stock will rise over the next day, week, or month? If we trade those predictions, do we earn useful returns after costs compared with simple alternatives?')
    st.info('Prediction quality and investment performance are separate questions. Positive trading returns alone do not prove the model adds value.')
    st.subheader('2. What did we build?')
    st.table([{'Part': 'Data collection', 'Purpose': 'Download the chosen ticker’s historical daily prices and volume from Yahoo Finance.'}, {'Part': 'Features', 'Purpose': 'Create past-only clues such as momentum, moving averages, RSI and volatility.'}, {'Part': 'XGBoost', 'Purpose': 'Train separate models for positive returns over 1, 5 and 21 trading sessions.'}, {'Part': 'Validation', 'Purpose': 'Learn from earlier dates and evaluate predictions on later dates.'}, {'Part': 'Backtesting', 'Purpose': 'Simulate next-open entries and horizon-based open exits, including slippage.'}, {'Part': 'Evaluation', 'Purpose': 'Compare predictions with simple models, and investment performance with SPY.'}])
    st.subheader('3. Explain it with one example')
    st.write('Invented illustration: Apple rises from 100 to 105 over five sessions. That +5% change and other past indicators become inputs. Suppose XGBoost estimates a 60% probability of a positive return over the next five sessions. This clears the default 55% trading threshold; other position rules determine whether a trade can enter at the next open. Later, we compare the prediction with what actually happened and calculate trading profit or loss after costs.')
    st.subheader('4. How is the evaluation designed?')
    st.write('Chronological training, validation and final-test periods preserve time order. Expanding-window walk-forward refits occur every 63 sessions using only outcomes completed by the cutoff. Matured earlier test outcomes may enter scheduled later test refits. We compare XGBoost with always-up, majority, momentum and logistic regression; SPY is a market-context input and buy-and-hold benchmark.')
    st.subheader('5. Five main results and what they mean')
    definitions = [('Out-of-sample accuracy', 'How often was the predicted direction correct on later dates? 70 correct out of 100 means 70%.'), ('Baseline improvement', 'XGBoost accuracy minus always-up accuracy. 52.68% − 55.58% = −2.90 percentage points: the model did worse.'), ('Sharpe ratio', 'Average daily return relative to daily variability, annualized, using zero risk-free rate. This is not a percentage.'), ('Maximum drawdown vs SPY', 'Largest decline from a portfolio peak. 120,000 → 90,000 means −25%. A −30% drawdown is deeper than −20%.'), ('After-cost annualized return (CAGR)', 'Compound annual growth after costs. 100,000 → 121,000 in two years means 10% CAGR; it does not mean each year earned exactly 10%.')]
    for name, en in definitions:
        with st.container(border=True):
            st.markdown(f'**{name}**')
            st.write(en)
    run = Path(__file__).parent / 'artifacts' / '20261024T192312985153Z'
    st.subheader('6. Actual saved AAPL results')
    st.caption('Experiment: 20261024T192312985153Z | Final test: 2025-01-02–2026-10-23 | Slippage: 10 bps (0.10%) per side | Initial capital: $100,000. Fixed experiment, not a live ticker selection.')
    if (run / 'accuracy.csv').exists() and (run / 'performance.csv').exists():
        a, p = (pd.read_csv(run / 'accuracy.csv'), pd.read_csv(run / 'performance.csv'))
        a = a[a.phase.eq('test')]
        p = p[p.phase.eq('test') & p.bps.eq(10)]
        records = []
        for _, row in a[a.model.eq('xgboost')].iterrows():
            baseline = a[a.horizon.eq(row.horizon) & a.model.eq('always_up')].iloc[0]
            strategy = p[p.horizon.eq(row.horizon) & p.model.eq('xgboost')].iloc[0]
            records.append({'Horizon': {1: 'Daily (1)', 5: 'Weekly (5)', 21: 'Monthly (21)'}[int(row.horizon)], 'Accuracy (%)': row.accuracy * 100, 'Baseline (%)': baseline.accuracy * 100, 'Improvement (pp)': (row.accuracy - baseline.accuracy) * 100, 'Sharpe': strategy.sharpe, 'Drawdown (%)': strategy.max_drawdown * 100, 'After-cost CAGR (%)': strategy.cagr * 100})
        table = pd.DataFrame(records)
        st.dataframe(table.round(2), hide_index=True)
        spy = p[p.model.eq('SPY')].iloc[0]
        st.write(f'SPY: Sharpe {spy.sharpe:.2f} | Maximum drawdown {spy.max_drawdown:.2%} | After-cost CAGR {spy.cagr:.2%}')
        st.download_button('Download results CSV', table.to_csv(index=False), file_name='AAPL_presentation_metrics.csv', mime='text/csv')
    else:
        st.warning('The saved AAPL experiment is unavailable. Use View and download my results for another completed experiment.')
    st.write('Weekly accuracy was 52.68%, below the 55.58% always-up baseline. Weekly CAGR was positive at 13.04%, but below SPY’s 16.30%; Sharpe was lower and drawdown deeper. The other horizons also underperformed SPY on these investment measures at baseline costs.')
    st.subheader('7. A short script you can present')
    st.success('“My project investigates whether XGBoost can predict stock direction and support useful trading. It trains daily, weekly and monthly models on past information and evaluates them chronologically using walk-forward testing. We simulate trades after costs and compare against simple prediction baselines and SPY. In our Apple experiment, the weekly strategy earned a positive return but did not beat the accuracy baseline or SPY at standard costs. The contribution is a reproducible evaluation pipeline that measures whether the model adds value, rather than assuming that positive returns prove success.”')
    st.caption('These are descriptive statistics. Statistical significance is not established; overlapping outcomes are not independent samples. Historical performance does not guarantee future returns.')
