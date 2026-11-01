from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_empty_dashboard_shows_create_run(tmp_path, monkeypatch):
    entry = Path(__file__).resolve().parents[1]/"legacy_app.py"
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(entry)).run()
    assert not app.exception
    assert any("Create a run" in item.value for item in app.info)
