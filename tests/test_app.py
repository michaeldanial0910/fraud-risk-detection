"""Smoke test: every page of the Streamlit app renders without an exception.
    pytest -q tests/test_app.py      (needs the trained model + reports; skipped otherwise)"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAGES = ["queue.py", "thresholds.py", "results.py", "mistakes.py", "data_story.py"]
ready = (ROOT / "models" / "fraud_model.joblib").exists() and (ROOT / "reports" / "eda_summary.json").exists()


@pytest.mark.skipif(not ready, reason="model or reports missing: run src.train, src.explain, src.eda")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app_pages" / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not ready, reason="model or reports missing")
def test_queue_hindsight_toggle(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app_pages" / "queue.py"), default_timeout=120).run()
    at.toggle(key="show_truth").set_value(True).run()
    assert not at.exception
