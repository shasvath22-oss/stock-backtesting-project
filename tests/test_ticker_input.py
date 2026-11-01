from pathlib import Path

from streamlit.testing.v1 import AppTest

from trend_analyzer.common import manifest


ROOT = Path(__file__).resolve().parents[1]


def test_ticker_input_runs_each_stage_for_selected_stock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config').mkdir()
    for name in ['research.json', 'universe.csv']:
        (tmp_path / 'config' / name).write_bytes((ROOT / 'config' / name).read_bytes())
    import trend_analyzer.__main__ as cli
    calls = []

    def execute(stage, run):
        calls.append((stage, manifest(run)['config']['tickers']))

    monkeypatch.setattr(cli, 'execute', execute)
    app = AppTest.from_file(str(ROOT / 'research_app.py'), default_timeout=30).run()
    next(w for w in app.text_input if w.label == 'Stock ticker').set_value(' msft ')
    next(w for w in app.button if w.label == 'Run complete analysis for this ticker').click().run()
    assert not app.exception
    assert calls == [(s, ['MSFT']) for s in ['collect', 'features', 'splits', 'train', 'predict', 'backtest', 'report']]
    saved = list(Path('artifacts').glob('*/manifest.json'))
    assert len(saved) == 1
    cfg = manifest(saved[0].parent)['config']
    assert cfg['benchmark'] == 'SPY'
    assert cfg['symbols'] == ['MSFT']


def test_invalid_ticker_creates_no_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / 'research_app.py'), default_timeout=30).run()
    next(w for w in app.text_input if w.label == 'Stock ticker').set_value('AAPL,MSFT')
    next(w for w in app.button if w.label == 'Run complete analysis for this ticker').click().run()
    assert not app.exception
    assert any('one valid stock ticker' in w.value for w in app.error)
    assert not list(Path('artifacts').glob('*/manifest.json'))
