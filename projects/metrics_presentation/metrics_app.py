"""A compact, exportable view of saved research statistics."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from process_example import show_simple_example
from feature_example import show_feature_example
from presentation_pages import show_presentation
from sample_data import show_sample
from amzn_walkthrough import show_amzn

st.set_page_config(page_title="Submission metrics", layout="wide")
st.title("Submission metrics")
st.write("Choose one experiment, evaluation period, and cost. Download the tables below for your submission.")
st.link_button("Open research workflow", "http://localhost:8502")
if st.query_params.get('sample'):
    show_sample(st.query_params['sample'])
    st.stop()
view = st.sidebar.radio('View', ['AMZN — one-page summary', 'AMZN — full real-data walkthrough', 'Presentation — English', 'Statistics one-pager', 'Simple example — start here', 'Understand the process', 'View and download my results'])
if view == 'AMZN — one-page summary':
    summary = Path(__file__).with_name('AMZN_ONE_PAGER.md').read_text(encoding='utf-8')
    st.download_button('Download AMZN one-page summary', summary, file_name='AMZN_ONE_PAGER.md', mime='text/markdown')
    st.markdown(summary)
    st.stop()
if view == 'AMZN — full real-data walkthrough':
    show_amzn()
    st.stop()
if view == 'Presentation — English':
    show_presentation()
    st.stop()
if view == 'Statistics one-pager':
    st.header('Five results for Economics, Statistics, and Quantitative Finance')
    st.write('Select a completed historical experiment. All figures below use its final-test period and the same transaction-cost assumption.')
    completed = []
    for manifest_path in sorted((Path(__file__).resolve().parent / 'artifacts').glob('*/manifest.json'), reverse=True):
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (not metadata.get('synthetic', True) and not metadata.get('learning')
                and metadata.get('workflow') == 'multi_horizon_research'
                and all(stage in metadata.get('stages', {}) for stage in ('predict', 'backtest'))
                and all((manifest_path.parent / name).exists() for name in ('accuracy.csv', 'performance.csv'))):
            completed.append(manifest_path.parent)
    if completed:
        focus_run = st.selectbox('Completed experiment', completed,
            format_func=lambda p: f"{p.name} · {', '.join(json.loads((p/'manifest.json').read_text(encoding='utf-8'))['config']['tickers'])}")
        focus_accuracy = pd.read_csv(focus_run/'accuracy.csv')
        focus_perf = pd.read_csv(focus_run/'performance.csv')
        available_costs = sorted(focus_perf.bps.unique().tolist())
        focus_cost = st.selectbox('Slippage (bps per side)', available_costs,
            index=available_costs.index(10) if 10 in available_costs else 0)
        focus_test = focus_accuracy[focus_accuracy.phase.eq('test')]
        focus_trading = focus_perf[focus_perf.phase.eq('test') & focus_perf.bps.eq(focus_cost)]
        spy_rows = focus_trading[focus_trading.model.eq('SPY')]
        rows = []
        for _, result in focus_test[focus_test.model.eq('xgboost')].iterrows():
            baseline = focus_test[focus_test.horizon.eq(result.horizon) & focus_test.model.eq('always_up')]
            strategy = focus_trading[focus_trading.horizon.eq(result.horizon) & focus_trading.model.eq('xgboost')]
            if baseline.empty or strategy.empty or spy_rows.empty:
                continue
            strategy, spy = strategy.iloc[0], spy_rows.iloc[0]
            rows.append({'Horizon (sessions)': int(result.horizon), 'Scored observations': int(result.samples),
                '1. OOS accuracy (%)': result.accuracy*100,
                'Baseline accuracy (%)': baseline.iloc[0].accuracy*100,
                '2. Accuracy improvement (pp)': (result.accuracy-baseline.iloc[0].accuracy)*100,
                '3. Strategy Sharpe': strategy.sharpe, 'SPY Sharpe': spy.sharpe,
                '4. Strategy max drawdown (%)': strategy.max_drawdown*100,
                'SPY max drawdown (%)': spy.max_drawdown*100,
                '5. After-cost CAGR (%)': strategy.cagr*100, 'SPY after-cost CAGR (%)': spy.cagr*100})
        focus_table = pd.DataFrame(rows)
        st.dataframe(focus_table.round(2), hide_index=True)
        if (focus_run/'splits.json').exists():
            focus_splits = json.loads((focus_run/'splits.json').read_text(encoding='utf-8'))
            st.caption(f"Saved test signal dates: {focus_splits.get('partitions', {}).get('test', {})}. Targets need completed exits; scored counts therefore differ by horizon. Trading covers its saved execution dates.")
        st.download_button('Download five-result summary CSV', focus_table.to_csv(index=False),
            file_name=f'{focus_run.name}_five_results_{focus_cost}bps.csv', mime='text/csv')
        st.markdown('**How to interpret the five results**')
        st.write('1. **Out-of-sample accuracy:** correct direction predictions on later dates. A 50% probability cutoff defines the predicted class.\n\n2. **Baseline improvement:** XGBoost accuracy minus always-up accuracy, in percentage points. Negative means XGBoost performed worse. Always-up is the declared reference here; the full results also include majority, momentum, and logistic regression.\n\n3. **Sharpe ratio:** annualized daily average return relative to daily variability, using zero risk-free rate. Compare with SPY at the same cost.\n\n4. **Maximum drawdown:** worst decline from a portfolio peak. −30% is a deeper loss than −20%. Compare strategy and SPY on the same backtest period.\n\n5. **Annualized return after costs:** CAGR, rather than total return. It includes the selected adverse slippage on buys and sells. Commissions and cash interest are assumed zero.')
        st.info('Example: 52.68% accuracy minus 55.58% baseline accuracy = −2.90 percentage points, not an improvement. A positive CAGR can coexist with weak directional accuracy and a lower return than SPY.')
        st.caption('Walk-forward out-of-sample evaluation allows scheduled refits using matured earlier test labels. Overlapping outcomes are dependent. These five results are descriptive; confidence intervals and significance tests are not supplied.')
    else:
        st.info('Complete prediction and backtesting for a historical experiment to populate these five results.')
    st.divider()
    summary = Path(__file__).with_name('STATISTICS_ONE_PAGER.md').read_text(encoding='utf-8')
    st.caption('Fixed summary of the AAPL experiment identified below. For other experiments, use View and download my results.')
    st.download_button('Download statistics one-pager', summary, file_name='STATISTICS_ONE_PAGER.md', mime='text/markdown')
    st.markdown(summary)
    st.stop()
if view == 'Simple example — start here':
    show_simple_example()
    st.stop()
if view == 'Understand the process':
    st.header('One stock, from input to submission')
    st.write('Example: you enter AAPL. The app studies Apple price history, trains models for Apple, and measures their predictions and simulated trades. SPY supplies market context and a comparison investment.')
    st.info('All numbers in this walkthrough are invented teaching examples. Your actual results are under “View and download my results”.')
    st.subheader('Start here')
    st.write('Open port 8502 → select Step 1 → enter AAPL in Stock ticker → check the dates and capital → click Run complete analysis for this ticker. The seven operations below run in order. Keep the experiment ID shown at the end.')
    examples = [
        ('1. Collect prices and save to SQL',
         'Input: your ticker AAPL and collection dates. The app downloads daily AAPL prices and SPY benchmark prices from Yahoo Finance.',
         'An AAPL day might have Open $100, High $105, Low $99, Close $103, Volume 10,000. SQL stores that day as a row identified by experiment, ticker, and date.',
         'Output: saved daily prices and data-quality statistics. This step has not trained any model. Missing downloads stop the process; no generated prices replace them.'),
        ('2. Turn prices into features and targets',
         'Features describe what was already known after each close: recent returns, moving averages, RSI, volatility, volume, and market context.',
         'If price was $100 five sessions ago and is $105 now, the trailing change is +5%. This becomes a clue. Separately, if the next-open entry is $106 and the open five sessions after entry is $109, the forward return is about +2.83% without dividends. The training answer is “up”.',
         'Output: a table of clues (X) and historical answers (y) for 1, 5, and 21 sessions. Future answers never become input clues. These targets use open-to-open returns, not today’s close to a future close.'),
        ('3. Separate learning dates from evaluation dates',
         'The app uses earlier observations to learn, then evaluates predictions on later dates. Your saved research periods determine the actual date boundaries.',
         'As a simplified example, learn from earlier years, validate on later development dates, and evaluate the final test on 2025 onward. A 21-session answer cannot be used for training until its exit has happened.',
         'Output: a chronological evaluation plan. The app refits at scheduled windows using completed outcomes known at the cutoff, including eligible earlier test outcomes. It does not fit once and freeze forever.'),
        ('4A. Train XGBoost',
         'The app gives XGBoost the earlier feature rows and their known answers. It also trains logistic regression and simple comparison methods.',
         'One row might contain momentum +5%, RSI 60, and price above its moving average, followed by an “up” answer. XGBoost learns patterns across many rows using decision trees; it is not a fixed rule that RSI 60 means buy.',
         'Output: separate trained models for 1-, 5-, and 21-session horizons. In an AAPL experiment, AAPL supplies the training examples; SPY supplies context and benchmark prices.'),
        ('4B. Predict and score',
         'Each trained model estimates the probability of a positive forward return on later observations. The app then compares predictions with the actual historical answers.',
         'A probability of 60% is classified as “up” at the 50% accuracy cutoff. It also clears the default 55% trading threshold. A 52% prediction counts as “up” for accuracy, but does not clear that trading threshold.',
         'Output: accuracy, precision, recall, AUC, and a confusion matrix. These describe predictions, not portfolio profit.'),
        ('5. Simulate trades after costs',
         'A qualifying signal after a close can enter at the next open. The simulation applies position and exit rules, credits eligible dividends, and includes adverse slippage.',
         'Suppose $1,000 cash buys at a reference open of $100. With 10 bps buy slippage, the fill is $100.10, buying about 9.9900 units. If the exit reference open is $103, the sell fill is $102.897 after 10 bps sell slippage. Final cash is about $1,027.94: profit about $27.94, with no dividends in this example.',
         'Output: portfolio returns, drawdown, Sharpe, closed trades, and costs. A signal can be skipped when the stock is already held, so prediction counts and trade counts differ.'),
        ('6. Produce the report and submission statistics',
         'The report collects prediction and trading results. The metrics page reads these saved files; it does not rerun the model when you change the display filters.',
         'If AAPL strategy return is +8% and SPY buy-and-hold is +12% over the same evaluation dates and cost assumption, the strategy made money but did not beat SPY on return. Compare drawdown as well.',
         'Output: downloadable CSV statistics, experiment settings, and the HTML report available in the research app. Keep the original test results rather than repeatedly tuning settings to improve them.'),
    ]
    for title, action, example, output in examples:
        with st.container(border=True):
            st.subheader(title)
            st.write(action)
            st.markdown('**Worked example**')
            st.write(example)
            st.markdown('**What you get**')
            st.write(output)
            if title.startswith('2.'):
                show_feature_example()
    st.header('How to understand the statistics you submit')
    st.write('Imagine 100 scored predictions with this result. These are teaching counts, not saved results:')
    st.table([
        {'Actual outcome': 'Up', 'Predicted up': 40, 'Predicted not up': 20},
        {'Actual outcome': 'Not up', 'Predicted up': 10, 'Predicted not up': 30},
    ])
    st.table([
        {'Metric': 'Samples', 'Example calculation': '40 + 20 + 10 + 30 = 100 scored observations'},
        {'Metric': 'Accuracy', 'Example calculation': '(40 + 30) / 100 = 70% correct'},
        {'Metric': 'Precision', 'Example calculation': '40 / (40 + 10) = 80% of predicted rises actually rose'},
        {'Metric': 'Recall', 'Example calculation': '40 / (40 + 20) = 66.67% of actual rises were found'},
        {'Metric': 'Balanced accuracy', 'Example calculation': '(40/60 + 30/40) / 2 = 70.83%'},
        {'Metric': 'AUC', 'Example calculation': 'Ranks probabilities across classes. 0.5 is random ranking; it cannot be calculated from this one confusion matrix alone.'},
        {'Metric': 'Net return', 'Example calculation': '$100,000 becomes $110,000 after costs → +10%'},
        {'Metric': 'CAGR', 'Example calculation': '$100,000 becomes $121,000 in exactly two years → 10% compound growth per year'},
        {'Metric': 'Maximum drawdown', 'Example calculation': 'Portfolio peak $120,000 falls to $90,000 → −25%'},
        {'Metric': 'Annualized volatility', 'Example calculation': 'Daily standard deviation 1% × square root of 252 → about 15.87%'},
        {'Metric': 'Sharpe', 'Example calculation': 'Daily average return 0.05%, daily standard deviation 1% → about 0.79 annualized, with zero risk-free rate'},
        {'Metric': 'Trade win rate', 'Example calculation': '6 profitable closed trades out of 10 → 60%; this is different from prediction accuracy'},
    ])
    st.warning('A 70% prediction accuracy does not mean a 70% investment return. Return also depends on trade sizes, gains versus losses, skipped signals, and costs. Historical results do not guarantee future performance.')
    st.header('Exactly what to download')
    st.write('1. In this page’s sidebar, choose View and download my results.\n\n2. Select the experiment created for your ticker.\n\n3. Choose Final test and the declared trading cost, such as 10 bps per side.\n\n4. Download submission table CSV and experiment settings JSON.\n\n5. Keep all three horizons and baseline rows, including the SPY comparison. Label development results separately if you include them.')
    st.write('Example submission statement: “We evaluated AAPL using 1-, 5-, and 21-session models on the saved final-test dates, with 10 bps slippage per side. We report prediction statistics and trading performance against baseline models and buy-and-hold benchmarks.” Fill in your actual dates and results from the saved experiment.')
    st.caption('These are descriptive statistics. Overlapping targets and shared market exposure mean observations are not independent trials; this report does not establish statistical significance.')
    st.stop()
runs = []
for p in sorted((Path(__file__).resolve().parent / 'artifacts').glob('*/manifest.json'), reverse=True):
    m = json.loads(p.read_text(encoding='utf-8'))
    if not m.get('synthetic', True) and m.get('workflow') == 'multi_horizon_research':
        runs.append(p.parent)
if not runs:
    st.info('Create a historical research experiment in the research workflow first.')
    st.stop()
run = st.selectbox('Experiment', runs, format_func=lambda p: p.name)
m = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
if not all((run/name).exists() for name in ('accuracy.csv','performance.csv')):
    st.info('Complete prediction (Step 4B) and backtest (Step 5) for this experiment first.')
    st.stop()
phase = st.selectbox('Evaluation period', ['test','development'], format_func=lambda p: 'Final test' if p == 'test' else 'Development validation')
perf = pd.read_csv(run/'performance.csv')
accuracy = pd.read_csv(run/'accuracy.csv')
costs = sorted(perf.bps.unique().tolist())
cost = st.selectbox('Trading cost: basis points per side', costs, index=costs.index(10) if 10 in costs else 0)
st.caption(f"Instruments: {', '.join(m['config']['tickers'])} | Initial capital: {m['config']['capital']:,.0f} | {cost} bps = {cost/100:.2f}% each side. Prediction cutoff: 50%; trading threshold: {m['config']['threshold']:.0%}.")
st.subheader('1. Prediction statistics — how well did the models predict?')
pred = accuracy[accuracy.phase.eq(phase)].copy()
columns = ['horizon','model','samples','accuracy','balanced_accuracy','precision','recall','auc','true_negative','false_positive','false_negative','true_positive']
pred = pred[[c for c in columns if c in pred]]
st.dataframe(pred.round(4), hide_index=True)
st.download_button('Download prediction statistics CSV', pred.to_csv(index=False), file_name=f'{run.name}_{phase}_prediction_statistics.csv', mime='text/csv')
st.subheader('2. Trading statistics — what happened after costs?')
portfolio = perf[perf.phase.eq(phase) & perf.bps.eq(cost)].copy()
portfolio = portfolio[['horizon','model','bps','net_return','cagr','annualized_volatility','sharpe','max_drawdown','trades','win_rate','slippage_cost']]
st.dataframe(portfolio.round(4), hide_index=True)
st.caption('Horizon 0 identifies buy-and-hold benchmarks, not a prediction horizon. Fractions use decimal format: 0.10 = 10%; −0.20 drawdown = a 20% fall from a peak.')
st.download_button('Download trading statistics CSV', portfolio.to_csv(index=False), file_name=f'{run.name}_{phase}_{cost}bps_trading_statistics.csv', mime='text/csv')
st.subheader('3. Submission table — XGBoost with comparisons')
merged = portfolio.merge(pred, on=['horizon','model'], how='left')
st.dataframe(merged.round(4), hide_index=True)
st.caption('Blank prediction metrics for SPY/equal_weight are expected: these benchmarks hold investments and do not predict direction.')
metadata = {'experiment': run.name, 'evaluation_period': phase, 'cost_bps_per_side': cost, 'config': m['config'], 'versions': m.get('versions'), 'stages': m.get('stages')}
st.download_button('Download submission table CSV', merged.to_csv(index=False), file_name=f'{run.name}_{phase}_{cost}bps_submission.csv', mime='text/csv')
st.download_button('Download experiment settings JSON', json.dumps(metadata, indent=2), file_name=f'{run.name}_settings.json', mime='application/json')
st.subheader('What should I submit?')
st.write('Submit the comparison table and experiment settings. Keep all three horizons (1, 5, 21 sessions), the baseline models, and SPY/equal-weight benchmarks. State the evaluation dates, instruments, capital, trading threshold, and cost assumption. Use the final test for final evaluation; label development results separately.')
with st.expander('Plain-English metric reference', expanded=True):
    st.table([
        {'Metric':'Samples','Meaning':'Number of scored observations; overlapping dates are not independent samples.'},
        {'Metric':'Accuracy','Meaning':'Fraction of correct direction predictions. Compare with always_up and majority.'},
        {'Metric':'Balanced accuracy','Meaning':'Average recall for down and up classes; useful when one direction is more common.'},
        {'Metric':'Precision / recall','Meaning':'Precision: of predicted rises, how many rose? Recall: of actual rises, how many were found?'},
        {'Metric':'AUC','Meaning':'How well probabilities rank rises above non-rises; 0.5 is random ranking.'},
        {'Metric':'Net return / CAGR','Meaning':'Total return after costs / annualized compound growth.'},
        {'Metric':'Volatility / drawdown','Meaning':'Annualized daily-return variability / largest fall from a portfolio peak.'},
        {'Metric':'Sharpe','Meaning':'Daily average return relative to daily variability, annualized; zero risk-free rate.'},
        {'Metric':'Trades / win rate','Meaning':'Closed trade count / fraction of closed trades with positive realized profit. Different from prediction accuracy.'},
    ])
st.info('These are descriptive historical statistics. No statistical significance is established: instruments share market exposure and horizon labels overlap. Do not report sample counts as independent trials or invent p-values.')
