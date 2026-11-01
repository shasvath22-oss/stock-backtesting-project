"""Worked example separating known features from future training labels."""
import streamlit as st


def show_feature_example():
    st.subheader('Features and targets: follow one row')
    st.caption('Invented prices for teaching only. S0, S1, etc. mean consecutive trading sessions, excluding weekends and holidays. Assume no dividends or missing prices.')
    st.write('We are standing just after the close of S0. The model must use only information available by that moment.')
    st.table([
        {'Session': 'S−5', 'Open': '—', 'Close': '100', 'Known after S0 close?': 'Yes — past'},
        {'Session': 'S−2', 'Open': '—', 'Close': '101', 'Known after S0 close?': 'Yes — past'},
        {'Session': 'S−1', 'Open': '—', 'Close': '103', 'Known after S0 close?': 'Yes — past'},
        {'Session': 'S0 — signal day', 'Open': '104', 'Close': '105', 'Known after S0 close?': 'Yes — now'},
        {'Session': 'S1 — next open / entry', 'Open': '106', 'Close': '107', 'Known after S0 close?': 'No — future'},
        {'Session': 'S2 — 1-session exit', 'Open': '104', 'Close': '105', 'Known after S0 close?': 'No — future'},
        {'Session': 'S6 — 5-session exit', 'Open': '109', 'Close': '110', 'Known after S0 close?': 'No — future'},
        {'Session': 'S22 — 21-session exit', 'Open': '102', 'Close': '103', 'Known after S0 close?': 'No — future'},
    ])
    st.markdown('**A. Build X: clues using only the past and S0**')
    st.code('5-session trailing return = (105 / 100) - 1 = 0.05 = +5%\n3-session simple average = (101 + 103 + 105) / 3 = 103\nDistance above that average = (105 / 103) - 1 ≈ +1.94%', language='text')
    st.write('The 3-session average is a small teaching example; the app uses its configured feature windows. RSI, volatility, volume clues, and SPY context also come from already completed observations. This tiny table is not enough history to calculate every actual feature.')
    st.markdown('**B. Build y: the answer, available only after the future exit**')
    st.write('The simulated entry reference is the S1 open, 106. Count each horizon from that entry: one session later is S2; five sessions later is S6; 21 sessions later is S22.')
    st.table([
        {'Horizon': '1 session', 'Entry': 'S1 open = 106', 'Exit': 'S2 open = 104', 'Forward return': '(104 / 106) − 1 = −1.89%', 'Target y': '0 — not up'},
        {'Horizon': '5 sessions', 'Entry': 'S1 open = 106', 'Exit': 'S6 open = 109', 'Forward return': '(109 / 106) − 1 = +2.83%', 'Target y': '1 — up'},
        {'Horizon': '21 sessions', 'Entry': 'S1 open = 106', 'Exit': 'S22 open = 102', 'Forward return': '(102 / 106) − 1 = −3.77%', 'Target y': '0 — not up'},
    ])
    st.markdown('**C. What a training row looks like**')
    st.table([
        {'Signal session': 'S0', 'Ticker': 'AAPL', 'Horizon': h, 'X: past 5-session return': '+5%', 'y: historical answer': y}
        for h, y in [(1, '0'), (5, '1'), (21, '0')]
    ])
    st.success('Same day, same past clues, three different future questions. The three horizon models learn separately. The horizon identifies the question; it is not an extra future-price clue.')
    st.write('During historical training, X and the completed y are available to teach the model. When making a new prediction after S0 close, it receives only X. Neither the S1 entry price nor any future exit price is an input clue. Training cannot use this row’s answer until the corresponding exit has happened.')
    st.warning('Target 1 means positive forward return before trading slippage, including eligible dividends in the actual app. It does not mean a profitable executed trade after costs. Zero or negative forward return gives target 0; missing future data leaves the target unknown, not 0.')
