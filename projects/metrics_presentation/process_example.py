"""A short, interactive English explanation; does not execute research."""
import streamlit as st
from feature_example import show_feature_example


def show_simple_example():
    st.header('What happens after I enter a ticker?')
    st.write('You enter AAPL (Apple). The app uses its old prices to learn patterns, tests predictions on later dates, and reports how accurate and profitable those predictions were in a simulation.')
    st.info('Everything below is a made-up teaching example. Changing these controls does not train a model or change your real experiment.')
    st.markdown('**AAPL → old prices → patterns → XGBoost → predictions → simulated trades → statistics**')
    steps = [
        ('1. You enter a ticker', 'You type AAPL and choose the dates and starting capital.', 'Click Run complete analysis for this ticker on port 8502. The process runs for AAPL. SPY is downloaded as a comparison and market context.', 'One saved experiment for AAPL.'),
        ('2. The app collects old prices', 'Yahoo Finance supplies a row for each trading day.', 'Example row: Open $100, High $105, Low $99, Close $103, Volume 10,000. SQL is simply where these rows are stored.', 'A table of historical prices. Nothing has been predicted yet.'),
        ('3. The app creates clues and answers', 'Prices are converted into features: numbers that describe past behavior.', 'If the close was $100 five sessions ago and $105 now, the recent change is +5%. That is a clue. For a different calculation, next-open entry $106 and an exit open five sessions later at $109 gives +2.83% before costs, without dividends: the historical answer is UP.', 'Many rows of clues plus known historical answers. The future answer is kept separate from the clues.'),
        ('4. XGBoost learns from earlier examples', 'Earlier data is used for learning; later dates are used for evaluation.', 'Imagine a practice sheet with clues in one column and UP / NOT UP answers in another. XGBoost learns patterns from the practice rows. For a later date, it receives only the clues and must predict the answer.', 'Trained models for 1, 5 and 21 sessions. At scheduled refits, only outcomes already completed at the cutoff may be used.'),
        ('5. The model makes a prediction', 'XGBoost outputs a probability, not a guaranteed future price.', 'Example: 60% estimated probability of a positive 5-session return. This clears the default 55% trading threshold. A 52% estimate does not. Other position and execution rules also apply.', 'Predictions that can be compared with actual historical outcomes.'),
        ('6. The app simulates a trade', 'A signal observed after the close can enter at the following open.', 'With $1,000, entry reference $100, exit reference $103, and 10 bps per side: buy fill $100.10, units about 9.9900, sell fill $102.897. Final cash is about $1,027.94; profit about $27.94. Assume no dividends and an eligible entry.', 'Trading results after costs. These are historical simulations, not actual broker orders.'),
        ('7. You submit statistics', 'The app summarizes prediction quality and trading performance.', 'If 70 out of 100 direction predictions were correct, accuracy is 70%. Separately, if $100,000 becomes $110,000 after costs, investment return is 10%. These percentages measure different things.', 'On this page choose View and download my results, then download the submission CSV and experiment settings JSON.'),
    ]
    selected = st.selectbox('Read one step at a time', range(len(steps)), format_func=lambda i: steps[i][0])
    title, meaning, example, output = steps[selected]
    with st.container(border=True):
        st.subheader(title)
        st.write(meaning)
        st.markdown('**Clear example**')
        st.write(example)
        st.markdown('**What comes out of this step?**')
        st.success(output)
    if selected == 2:
        show_feature_example()
    st.subheader('Try the prediction example')
    probability = st.slider('Example probability of UP (%)', 0, 100, 60)
    st.write(f'For accuracy scoring at 50%: **{"UP" if probability >= 50 else "NOT UP"}**.')
    st.write(f'At the default 55% trading threshold: **{"clears the probability threshold" if probability >= 55 else "does not clear the probability threshold"}**. Clearing it does not guarantee a trade or a profit.')
    st.subheader('The numbers you actually submit')
    st.table([
        {'Question': 'How often was the direction correct?', 'Metric': 'Accuracy', 'Simple example': '70 correct / 100 predictions = 70%'},
        {'Question': 'Were predicted rises really rises?', 'Metric': 'Precision', 'Simple example': '40 actual rises / 50 predicted rises = 80%'},
        {'Question': 'How many actual rises were found?', 'Metric': 'Recall', 'Simple example': '40 found / 60 actual rises = 66.67%'},
        {'Question': 'Did the simulated investment grow?', 'Metric': 'Net return', 'Simple example': '$100,000 → $110,000 after costs = +10%'},
        {'Question': 'How far did it fall from a peak?', 'Metric': 'Maximum drawdown', 'Simple example': '$120,000 peak → $90,000 = −25%'},
        {'Question': 'Did it beat simply holding the market?', 'Metric': 'SPY comparison', 'Simple example': 'Strategy +8%, SPY +12%: strategy made money but had lower return'},
    ])
    st.write('Submit all three horizons, baseline comparisons, and the settings used. A profitable strategy can still underperform SPY. Accuracy alone is not proof of profitable trading.')
    st.caption('The detailed explanation also covers balanced accuracy, AUC, CAGR, volatility, and Sharpe. Historical results do not guarantee future returns.')
