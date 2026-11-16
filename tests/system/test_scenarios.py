"""System tests: every scenario in tests/fixtures/scenarios is run end to end in LocalMode.

These tests call a real LLM (Vertex AI / Gemini), so they need valid Google Cloud credentials
(see README). Without a model use `python tests/utils/check_specs.py` instead - it validates the
specs and walks every scenario against the fixtures offline.

Scenarios ending in `_api` use the local mock API (start it first: `python -m Tools.MockApi.server`).
"""
import json
import socket
from pathlib import Path

import pytest

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios"


def _mock_api_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 9000), timeout=0.5):
            return True
    except OSError:
        return False


def _ids():
    out = []
    for p in sorted(SCENARIO_DIR.rglob("*.json")):
        rel = p.relative_to(SCENARIO_DIR).with_suffix("").as_posix()
        meta = json.loads(p.read_text(encoding="utf-8"))
        marks = []
        if meta.get("mock_tools") == 4:
            marks.append(pytest.mark.skipif(not _mock_api_up(), reason="mock API not running on :9000"))
        out.append(pytest.param(rel, id=rel, marks=marks))
    return out


@pytest.mark.parametrize("scenario_id", _ids())
def test_scenario(scenario_runner, scenario_id):
    answer, scenario, token = scenario_runner(scenario_id)
    assert answer.text
