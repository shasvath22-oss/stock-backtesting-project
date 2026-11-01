from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from trend_analyzer.common import create_run, manifest
from trend_analyzer.__main__ import execute


ROOT = Path(__file__).resolve().parents[1]


def learning_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    for name in ["research.json", "universe.csv"]:
        (tmp_path / "config" / name).write_bytes((ROOT / "config" / name).read_bytes())
    return create_run(demo=True, learning=True)


def test_one_stock_pipeline_and_guided_views(tmp_path, monkeypatch):
    run = learning_run(tmp_path, monkeypatch)
    cfg = manifest(run)["config"]
    assert cfg["tickers"] == ["AAPL"]
    assert cfg["horizons"] == [5]
    app = AppTest.from_file(str(ROOT / "legacy_app.py")).run()
    assert not app.exception
    assert any("Setup: understand" in w.value for w in app.subheader)
    scenario = next(w for w in app.selectbox if w.label.startswith("Example outcome"))
    scenario.set_value("Buy → loss").run()
    assert not app.exception
    assert any("$-51.90" in w.value for w in app.markdown)
    next(w for w in app.selectbox if w.label.startswith("Example outcome")).set_value("No buy").run()
    assert not app.exception
    assert not manifest(run)["stages"]
    next(w for w in app.button if w.label.startswith("Continue to step 1:")).click().run()
    # No implicit execution, and future lessons cannot run out of order.
    assert not manifest(run)["stages"]
    lesson = next(w for w in app.selectbox if w.label == "Lesson")
    lesson.set_value(3).run()
    assert not app.exception
    assert any("Complete step 1" in item.value for item in app.info)
    for index, stage in enumerate(["collect", "features", "splits", "train", "predict", "backtest", "report"]):
        if stage == "train":
            next(w for w in app.selectbox if w.label == "Lesson").set_value(index).run()
            assert not app.exception
            assert any("Training, operation by operation" in w.value for w in app.markdown)
            next(w for w in app.selectbox if w.label == "Training window to inspect").set_value(1).run()
            assert "train" not in manifest(run)["stages"]
        execute(stage, run)
        app = AppTest.from_file(str(ROOT / "legacy_app.py")).run()
        next(w for w in app.selectbox if w.label == "Lesson").set_value(index).run()
        assert not app.exception, stage
    features = pd.read_parquet(run / "features.parquet")
    assert set(features.ticker) == {"AAPL"}
    predictions = pd.read_parquet(run / "predictions.parquet")
    assert set(predictions.ticker) == {"AAPL"}
    assert set(predictions.horizon) == {5}
    assert (run / "report.html").exists()
    # Revisiting a lesson is read-only; Continue navigates without executing.
    next(w for w in app.selectbox if w.label == "Lesson").set_value(0).run()
    assert any(item.value == "What happens when you run it" or "What happens when you run it" in item.value for item in app.markdown)
    next(w for w in app.button if w.label.startswith("Continue to step 2:")).click().run()
    assert next(w for w in app.selectbox if w.label == "Lesson").value == 1
    assert "report" in manifest(run)["stages"]
    # Re-execution invalidates later results rather than displaying stale output.
    next(w for w in app.selectbox if w.label == "Lesson").set_value(0).run()
    next(w for w in app.button if w.label == "Run again: Download prices").click().run(timeout=30)
    assert not app.exception
    assert set(manifest(run)["stages"]) == {"collect"}


def test_create_learning_run_does_not_execute_stages(tmp_path, monkeypatch):
    learning_run(tmp_path, monkeypatch)
    app = AppTest.from_file(str(ROOT / "legacy_app.py")).run()
    next(w for w in app.button if w.label == "Create run").click().run()
    assert not app.exception
    selected = next(w for w in app.selectbox if w.label == "Run").value
    created = Path("artifacts") / selected
    assert manifest(created)["learning"]
    assert not manifest(created)["stages"]
    next(w for w in app.button if w.label.startswith("Continue to step 1:")).click().run()
    # Execute only collection via the same button a learner uses.
    next(w for w in app.button if w.label.startswith("Run step 1:")).click().run(timeout=30)
    assert not app.exception
    assert set(manifest(created)["stages"]) == {"collect"}
