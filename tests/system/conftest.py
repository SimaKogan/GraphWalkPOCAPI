import asyncio
import json
import os
from pathlib import Path

import pytest

from Config.settings import settings
from Core.routing_orchestrator import Orchestrator
from Tools.MockTools.scenario_loader import reset_cache

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)


def _load_scenario_meta(scenario_id: str) -> dict:
    path = ROOT / "tests" / "fixtures" / "scenarios" / f"{scenario_id}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_parent_and_submain_payloads(mem_store: dict, parent_main_id: str):
    """
    Yield (label, facts, cursor) for the parent main and its submains.

    With the new main/submain design, submains are stored in mem_store["mains"]
    but are NOT necessarily in main_order. Tests that assert on facts/cursor must
    therefore consider submains as well.
    """
    mains = (mem_store or {}).get("mains") or {}
    parent = mains.get(parent_main_id) or {}
    yield ("parent_main", parent.get("facts") or {}, parent.get("cursor") or {})

    for sub_id in parent.get("submain_ids", []) or []:
        sub = mains.get(sub_id) or {}
        yield (f"submain:{sub_id}", sub.get("facts") or {}, sub.get("cursor") or {})


def _find_first_non_null(values):
    for v in values:
        if v is not None:
            return v
    return None


def _resolve_expected_workflow_id(answer, mem_store: dict, parent_main_id: str):
    """
    Find workflow id from answer (preferred), then parent, then submains.

    The old tests assumed workflow_id lives on the "main" facts/cursor.
    With decomposition, workflow selection and execution usually happens on submains.
    """
    candidates = []

    # 1) Answer payload (most direct)
    if getattr(answer, "cursor", None):
        candidates.append(answer.cursor.get("workflow_id"))
    if getattr(answer, "facts", None):
        candidates.append(answer.facts.get("selected_workflow_id"))
        candidates.append(answer.facts.get("workflow_id"))

    # 2) Parent + submains
    for _, facts, cursor in _iter_parent_and_submain_payloads(mem_store, parent_main_id):
        candidates.append(cursor.get("workflow_id"))
        candidates.append(facts.get("selected_workflow_id"))
        candidates.append(facts.get("workflow_id"))

    return _find_first_non_null(candidates)


def _facts_contains_key_anywhere(answer, mem_store: dict, parent_main_id: str, key: str) -> bool:
    """
    Facts assertions should pass if the key exists in:
      - answer.facts
      - parent main facts
      - ANY submain facts
    """
    if getattr(answer, "facts", None) and key in (answer.facts or {}):
        return True

    for _, facts, _ in _iter_parent_and_submain_payloads(mem_store, parent_main_id):
        if key in facts:
            return True

    return False


@pytest.fixture
def run_full_conversation():
    """
    Helper fixture:
      - sets settings from scenario.config_path
      - forces debug + mock_tools=3 + scenario_id
      - runs Orchestrator in LocalMode
      - consumes scripted_user_replies until answer.pending is False
    """

    def _runner(scenario_id: str):
        reset_cache()
        scenario = _load_scenario_meta(scenario_id)

        # Load base config
        config_path = scenario.get("config_path", "Resources/SystemConfigs/exp_1_create_dbs.json")
        settings.update(str(ROOT / config_path))

        # Ensure debug block exists
        if not hasattr(settings, "debug") or settings.debug is None:
            settings.debug = {}  # type: ignore

        settings.debug["active"] = True
        settings.debug["mock_tools"] = scenario.get("mock_tools", 3)
        settings.debug["scenario_id"] = scenario_id
        if scenario.get("external_personal_info_base_url"):
            settings.services["external_personal_info_services"]["base_url"] = scenario.get(
                "external_personal_info_base_url")
        orc = Orchestrator(mode="local")

        user_id = scenario["user_id"]
        token = asyncio.run(orc.create_conversation(user_id=user_id))
        first_query = scenario["first_query"]

        answer = orc.ask(user_id=user_id, token=token, text=first_query)

        replies = list(scenario.get("scripted_user_replies", []))
        while answer.pending or len(replies) > 0:
            if not replies:
                raise RuntimeError(
                    f"Scenario '{scenario_id}' ran out of scripted_user_replies but answer.pending is still True"
                )
            user_reply = replies.pop(0)
            answer = orc.ask(user_id=user_id, token=token, text=user_reply)

        return answer, scenario, token, orc  # TODO: display html? console.log(answer.text)?

    return _runner


@pytest.fixture
def scenario_runner(run_full_conversation):
    """
    High-level helper:
      - runs the scenario via run_full_conversation
      - applies generic JSON expectations ('route', 'workflow_id',
        'facts_contains_keys', 'answer_must_include')
      - returns (answer, scenario, token) for workflow-specific extra asserts.
    """

    def _runner(scenario_id: str):
        answer, scenario, token, orc = run_full_conversation(scenario_id)
        exp = scenario.get("expectations", {})

        # Route
        # expected_route = exp.get("route")
        # if expected_route is not None:
        #     route = (answer.cursor or {}).get("route") or (answer.facts or {}).get("route")
        #     assert route == expected_route, f"Expected route '{expected_route}', got '{route}'"

        # Workflow id (search answer -> parent main -> submains)
        expected_wf = exp.get("workflow_id")
        # mem_store = getattr(getattr(orc, "sessions", None), "store", None).get(token).store  # type: ignore
        store = getattr(orc, "_local_store", None)
        if store is None:
            # Fallback: sessions_async -> AsyncLocalSessionsAdapter -> LocalSessionsAdapter -> store
            sessions_async = getattr(orc, "sessions_async", None)
            local_sessions = getattr(sessions_async, "_s", None)
            store = getattr(local_sessions, "store", None)

        if store is None:
            raise RuntimeError("No local SessionStore found on orchestrator (are you running in RemoteMode?)")

        mem_store = store.get(token).store  # type: ignore
        parent_main_id = mem_store.get("current_main_id")
        if expected_wf is not None:
            wf_id = _resolve_expected_workflow_id(answer, mem_store, parent_main_id)
            assert wf_id == expected_wf, f"Expected workflow_id '{expected_wf}', got '{wf_id}'"

        # Facts contain keys (in answer OR parent main OR any submain)
        for key in exp.get("facts_contains_keys", []):
            assert _facts_contains_key_anywhere(answer, mem_store, parent_main_id, key), (
                f"Expected facts to contain key '{key}' (in answer.facts, parent main facts, or any submain facts)"
            )

        # Answer must contain snippets
        for snippet in exp.get("answer_must_include", []):
            assert snippet in answer.text, f"Expected answer.text to contain '{snippet}'"

        # Basic sanity
        assert answer.pending is False

        return answer, scenario, token

    return _runner
