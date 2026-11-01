"""Read-only, allowlisted samples from the presentation's saved experiment."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUN = Path(__file__).parent/'artifacts'/'20261024T192312985153Z'
SAMPLES = {
    'input': ('1. Ticker and experiment settings', 'manifest.json', 'This records the selected ticker, capital, dates, model settings, and completed stages.'),
    'prices': ('2. Collected prices', 'bars.parquet', 'Each row is one instrument on one trading day. Prices include AAPL and SPY market context.'),
    'features': ('3. Features and historical targets', 'features.parquet', 'Past-only feature columns are model inputs. Entry/exit dates, forward return and target are future outcomes, kept out of the feature schema.'),
    'validation': ('4. Chronological validation and walk-forward folds', 'splits.json', 'Check fit_cutoff before each evaluation start. Training target exits must be at or before that cutoff.'),
    'training': ('5. Training inputs and model configuration', 'feature_schema.json', 'The columns list defines the actual model inputs. The manifest below records the XGBoost configuration and completed horizon training stages.'),
    'predictions': ('6. Predicted probabilities and signals', 'predictions.parquet', 'Probability is the estimated chance of a positive forward return. Signal uses the configured trading threshold; it is not a guarantee of execution. Target is the later observed answer.'),
    'trades': ('7. Simulated trades, positions and capital', 'backtests/test_h5_xgboost_10bps_trades.parquet', 'These are weekly XGBoost final-test executions at 10 bps per side. Reference prices differ from execution prices because of slippage.'),
    'metrics': ('8. Prediction, risk and return statistics', 'accuracy.csv', 'Accuracy describes direction predictions. Performance describes simulated investment results. Keep the horizon, evaluation period and costs consistent.'),
}


def sample_links():
    st.subheader('Open the saved sample data for each step')
    st.caption('AAPL experiment: 20261024T192312985153Z. Links open read-only data views on port 8503.')
    for key, (title, _, _) in SAMPLES.items():
        st.link_button(title, f'http://localhost:8503/?sample={key}')


def show_sample(key):
    if key not in SAMPLES:
        st.error('Unknown sample. Choose a link from the presentation page.')
        st.link_button('Back to presentation', 'http://localhost:8503')
        return
    title, filename, explanation = SAMPLES[key]
    st.header(title)
    st.link_button('Back to presentation pages', 'http://localhost:8503')
    st.caption(f'Saved historical data, not invented examples | AAPL experiment {RUN.name}')
    st.write(explanation)
    path = RUN/filename
    if not path.exists():
        st.info('This saved sample is unavailable.')
        return

    def display_table(path, label):
        frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        if key in ('features', 'predictions'):
            if 'ticker' in frame:
                frame = frame[frame.ticker.eq('AAPL')]
            if 'horizon' in frame:
                frame = frame[frame.horizon.eq(5)]
            if 'model' in frame:
                frame = frame[frame.model.eq('xgboost')]
            if 'phase' in frame:
                frame = frame[frame.phase.eq('test')]
            if key == 'features':
                inputs = json.loads((RUN/'feature_schema.json').read_text())['columns']
                frame = frame.dropna(subset=inputs)
                frame = frame[['date','ticker','horizon','return_5','sma_distance_20','rsi','volatility','entry_date','exit_date','forward_return','target']]
        st.subheader(label)
        st.caption(f'{len(frame):,} rows in this view; displaying the first 30. Decimal 0.05 means 5%.')
        st.dataframe(frame.head(30), hide_index=True)
        st.download_button(f'Download {label} CSV', frame.to_csv(index=False), file_name=path.stem+'_sample_view.csv', mime='text/csv', key=str(path))

    if path.suffix == '.json':
        st.json(json.loads(path.read_text(encoding='utf-8')))
        st.download_button('Download saved JSON', path.read_bytes(), file_name=path.name, mime='application/json')
    else:
        display_table(path, title)
    if key == 'training':
        metadata = json.loads((RUN/'manifest.json').read_text())
        st.json({'xgboost': metadata['config']['xgboost'], 'training_stages': {k:v for k,v in metadata['stages'].items() if k.startswith('train')}})
    if key == 'trades':
        for kind in ('positions', 'equity'):
            display_table(RUN/'backtests'/f'test_h5_xgboost_10bps_{kind}.parquet', kind.capitalize())
    if key == 'metrics':
        display_table(RUN/'performance.csv', 'Performance across all costs and periods')
    st.divider()
    sample_links()
