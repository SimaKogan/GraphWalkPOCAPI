import json
from pathlib import Path
from typing import Any, Dict, Optional

from Config.settings import settings

_SCENARIO_CACHE: Dict[str, Dict[str, Any]] = {}
_CALL_COUNTERS: Dict[str, int] = {}


def _project_root() -> Path:
    # Tools/MockTools/scenario_loader.py -> Tools/MockTools -> Tools -> repo
    return Path(__file__).resolve().parents[2]


def get_current_scenario_id() -> str:
    debug_cfg = getattr(settings, "debug", {}) or {}
    scenario_id = debug_cfg.get("scenario_id")
    if not scenario_id:
        raise RuntimeError("settings.debug['scenario_id'] is not set for scenario-based mocks")
    return scenario_id


def load_scenario() -> Dict[str, Any]:
    scenario_id = get_current_scenario_id()
    if scenario_id in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[scenario_id]

    root = _project_root()
    path = root / "tests" / "fixtures" / "scenarios" / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _SCENARIO_CACHE[scenario_id] = data
    # reset per-endpoint call counters when (re)loading scenario
    _CALL_COUNTERS.clear()
    return data


def reset_cache() -> None:
    _SCENARIO_CACHE.clear()
    _CALL_COUNTERS.clear()


def load_api_response(endpoint: str, call_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """
    Load the mock API response for a given endpoint, based on the current scenario.

    In the scenario JSON, 'api_responses' can be either:
      - endpoint: "relative/path/to/file.json"
      - endpoint: ["path1.json", "path2.json", ...]  # for multiple calls

    For now, we ignore call_kwargs; we can extend this later if we want
    arg-based dispatch.
    """
    scenario = load_scenario()
    mapping = scenario.get("api_responses", {})

    if endpoint not in mapping:
        raise KeyError(
            f"No API mock defined for endpoint='{endpoint}' in scenario='{get_current_scenario_id()}'"
        )

    spec = mapping[endpoint]
    if isinstance(spec, str):
        rel_path = spec
    else:
        # assume list of paths for multiple calls to same endpoint
        idx = _CALL_COUNTERS.get(endpoint, 0)
        try:
            rel_path = spec[idx]
        except (IndexError, TypeError):
            raise IndexError(
                f"Too many calls to endpoint='{endpoint}' for scenario='{get_current_scenario_id()}'"
            )
        _CALL_COUNTERS[endpoint] = idx + 1

    root = _project_root()
    full_path = root / rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"API fixture not found: {full_path}")

    with full_path.open("r", encoding="utf-8") as f:
        return json.load(f)
