"""Explain one completed AMZN run using a traceable historical trade."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUN = Path(__file__).parent/'artifacts'/'20261024T205450515026Z'


def show_amzn():
    st.header('AMZN: one full flow with real dates')
    st.info('Real Yahoo Finance historical prices, actual saved model outputs, and simulated trades. No brokerage orders were placed. This example is the first weekly final-test trade by date, not a trade selected for profitability.')
    if not (RUN/'report.html').exists():
        st.info('The AMZN analysis is still running. Refresh shortly to see the completed walkthrough.')
        return
    m = json.loads((RUN/'manifest.json').read_text())
    splits = json.loads((RUN/'splits.json').read_text())
    bars = pd.read_parquet(RUN/'bars.parquet')
    stock = bars[bars.ticker.eq('AMZN')].sort_values('date').reset_index(drop=True)
    features = pd.read_parquet(RUN/'features.parquet')
    predictions = pd.read_parquet(RUN/'predictions.parquet')
    trades = pd.read_parquet(RUN/'backtests/test_h5_xgboost_10bps_trades.parquet')
    buys = trades[trades.side.eq('buy')].reset_index(drop=True)
    if buys.empty:
        st.info('No weekly trades were executed. Inspect probabilities in the saved results instead.')
        return
    chosen = st.selectbox('Follow one actual weekly trade', range(len(buys)),
        format_func=lambda i: f"Signal {buys.iloc[i].signal_date.date()} → buy {buys.iloc[i].date.date()}")
    buy = buys.iloc[chosen]
    sell = trades[trades.side.eq('sell') & trades.date.eq(buy.scheduled_exit)].iloc[0]
    prediction = predictions[predictions.model.eq('xgboost') & predictions.horizon.eq(5)
        & predictions.phase.eq('test') & predictions.date.eq(buy.signal_date)].iloc[0]
    feature = features[features.horizon.eq(5) & features.ticker.eq('AMZN') & features.date.eq(buy.signal_date)].iloc[0]
    number = stock.index[stock.date.eq(buy.signal_date)][0]
    signal_close, past_close = stock.iloc[number].close, stock.iloc[number-5].close
    cutoff = pd.Timestamp(prediction.fit_cutoff)
    st.caption(f"Experiment {RUN.name} | capital ${m['config']['capital']:,.0f} | weekly = 5 trading sessions | cost = 10 bps per side")

    st.subheader('1. Input: what did we ask the app to do?')
    st.write(f"Study AMZN using prices from {m['config']['start']} through the last completed session before {m['config']['end']}. Train daily, weekly and monthly models, and compare them with SPY. The entire collect → features → splits → train → predict → backtest → report flow has completed.")
    st.dataframe(pd.read_csv(RUN/'coverage.csv'), hide_index=True)

    st.subheader('2. Real prices: what was known on the signal date?')
    st.write(f"Stand at the close of **{buy.signal_date.date()}**. Only this date and earlier prices can be used as clues. Later rows shown below are for explaining what happened afterward.")
    sample = stock.iloc[max(0,number-5):number+1][['date','ticker','open','high','low','close','volume']]
    st.dataframe(sample, hide_index=True)
    st.write(f"Five sessions earlier, close was **${past_close:.4f}**. On the signal day it was **${signal_close:.4f}**. The price-only change is ({signal_close:.4f} ÷ {past_close:.4f} − 1) = **{signal_close/past_close-1:.2%}**.")

    st.subheader('3. Features X versus future answer y')
    st.table([
        {'Input clue available at signal close': '5-session trailing return (dividend-inclusive)', 'Saved value': f'{feature.return_5:.4%}'},
        {'Input clue available at signal close': 'Distance from 20-session simple moving average', 'Saved value': f'{feature.sma_distance_20:.4%}'},
        {'Input clue available at signal close': 'RSI', 'Saved value': f'{feature.rsi:.4f}'},
        {'Input clue available at signal close': 'Volatility feature', 'Saved value': f'{feature.volatility:.6f}'},
    ])
    st.write(f"The later entry reference was **${buy.reference_price:.4f} on {buy.date.date()}**; the later exit reference was **${sell.reference_price:.4f} on {sell.date.date()}**. The saved forward return is **{feature.forward_return:.2%}**, so the historical target is **{int(feature.target)} ({'UP' if feature.target == 1 else 'NOT UP'})**. Entry/exit prices and this answer were unknown at the signal close and are excluded from X.")

    st.subheader('4. Training and validation: what could the model learn from?')
    st.json(splits['partitions'])
    st.write(f"For this prediction, the model's training cutoff was **{cutoff.date()}**. Its latest training target exit was **{pd.Timestamp(prediction.max_label_exit).date()}**, which must not exceed that cutoff. It then predicted the later signal date **{buy.signal_date.date()}**. Refits occur every 63 sessions using an expanding history; earlier completed test outcomes can enter later test refits.")

    st.subheader('5. Prediction: did it trigger a signal?')
    st.write(f"XGBoost predicted an UP probability of **{prediction.probability:.2%}**. The trading threshold was **{m['config']['threshold']:.0%}**. It cleared that threshold, and the simulation had no existing AMZN position blocking this entry, so it bought at the next open on **{buy.date.date()}**. Prediction accuracy instead uses a 50% cutoff.")
    st.write(f"For scoring this one prediction: predicted UP, actual outcome **{'UP' if prediction.target == 1 else 'NOT UP'}** → **{'correct' if prediction.target == 1 else 'incorrect'}**. A single prediction is not enough to judge the model.")

    st.subheader('6. Execution: show the actual simulated buy and sell')
    st.dataframe(pd.DataFrame([buy,sell])[['date','side','reference_price','executed_price','normalized_units','notional','slippage_cost','realized_pnl']], hide_index=True)
    st.write(f"Buy fill = {buy.reference_price:.4f} × 1.001 = **${buy.executed_price:.4f}**. Sell fill = {sell.reference_price:.4f} × 0.999 = **${sell.executed_price:.4f}**. Quantity = **{buy.normalized_units:.4f} split-normalized research units**. This closed trade's saved profit/loss, including costs and eligible distributions, was **${sell.realized_pnl:+,.2f}**. These units describe vendor-normalized history, not necessarily contemporaneous exchange share counts.")
    curve = pd.read_parquet(RUN/'backtests/test_h5_xgboost_10bps_equity.parquet')
    period = curve[curve.date.between(buy.date,sell.date)]
    st.dataframe(period[['date','equity','cash','return','drawdown']], hide_index=True)
    st.caption('Daily equity is marked at the close. A scheduled exit can be followed by a new entry on the same day, so the exit-day portfolio row may include the next trade. Realized P&L above belongs to this closed trade.')

    st.subheader('7. Overall evidence: judge all test predictions and trades')
    a, p = pd.read_csv(RUN/'accuracy.csv'), pd.read_csv(RUN/'performance.csv')
    a = a[a.phase.eq('test')]
    test = p[p.phase.eq('test') & p.bps.eq(10)]
    records = []
    for h in (1,5,21):
        model = a[a.horizon.eq(h) & a.model.eq('xgboost')].iloc[0]
        baseline = a[a.horizon.eq(h) & a.model.eq('always_up')].iloc[0]
        strategy = test[test.horizon.eq(h) & test.model.eq('xgboost')].iloc[0]
        records.append({'Horizon':h, 'Scored predictions':int(model.samples), 'Accuracy (%)':model.accuracy*100,
            'Always-up accuracy (%)':baseline.accuracy*100, 'Improvement (pp)':(model.accuracy-baseline.accuracy)*100,
            'Sharpe':strategy.sharpe, 'Max drawdown (%)':strategy.max_drawdown*100,
            'After-cost CAGR (%)':strategy.cagr*100, 'Closed trades':int(strategy.trades)})
    table = pd.DataFrame(records)
    st.dataframe(table.round(2), hide_index=True)
    spy = test[test.model.eq('SPY')].iloc[0]
    st.write(f"SPY comparison at 10 bps: annualized return **{spy.cagr:.2%}**, Sharpe **{spy.sharpe:.2f}**, maximum drawdown **{spy.max_drawdown:.2%}**. Compare higher CAGR/Sharpe and less-deep drawdown; positive strategy returns alone do not demonstrate an advantage.")
    sensitivity = p[p.phase.eq('test') & p.model.eq('xgboost') & p.bps.isin([5,10,20])]
    st.markdown('**Annualized return after different transaction costs (%)**')
    st.dataframe((sensitivity.pivot(index='horizon',columns='bps',values='cagr')*100).round(2))
    st.line_chart(curve.set_index('date')[['equity']])
    st.download_button('Download AMZN five-result summary CSV', table.to_csv(index=False), file_name='AMZN_five_results_10bps.csv', mime='text/csv')
    st.download_button('Download full AMZN HTML report', (RUN/'report.html').read_bytes(), file_name='AMZN_report.html', mime='text/html')
    verification = RUN/'verification.json'
    if verification.exists():
        with st.expander('Saved-output verification checks'):
            st.json(json.loads(verification.read_text()))
    st.warning('These are historical, descriptive results. Overlapping targets are dependent; this run does not establish statistical significance or guarantee future returns.')
