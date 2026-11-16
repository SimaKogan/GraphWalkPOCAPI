"""Offline consistency check for the specs, fixtures and scenarios (no LLM, no network, no third-party packages).

    python tests/utils/check_specs.py            # check everything, exit code 1 on any error
    python tests/utils/check_specs.py --update   # also rewrite `facts_contains_keys` of the scenarios

What it does
  1. Loads Resources/Specs/transit_*.json and checks the references between them
     (workflow -> blocks -> actions -> tools/endpoints, states, questions).
  2. Runs the real WorkflowValidator over all workflows.
  3. Walks every scenario through its workflow with the real edge-condition code of the executor
     (extracted from Core/WFExecution/executor.py) using the scenario's mock responses, or - for
     the *_api scenarios - the demo rider from Resources/MockApi/riders.json, and checks that the
     walk ends, consumes the scripted selections, and produces the facts the scenario expects.

The LLM-driven parts (routing, workflow selection, answer composition) are NOT exercised here;
those run in tests/system with a real model.
"""
import ast
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Core.Utils.WorkflowsUtils.workflow_validator import WorkflowValidator  # noqa: E402  (pure python)

ERRORS: list = []
WARNINGS: list = []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNINGS.append(msg)


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Reuse the real functions instead of re-implementing them
# ---------------------------------------------------------------------------
def _extract(path: str, names: set, class_name: Optional[str] = None) -> dict:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    picked = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            picked.append(node)
    ns = {"re": re, "datetime": datetime, "timezone": timezone, "Optional": Optional, "Union": Union,
          "staticmethod": staticmethod}
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(module, path, "exec"), ns)
    return ns


_ex = _extract("Core/WFExecution/executor.py", {"lookup_dotted"})
lookup_dotted = _ex["lookup_dotted"]
_ex2 = _extract("Core/WFExecution/executor.py", {"_filter_edges_by_condition"}, class_name="Executor")
_ex2["lookup_dotted"] = lookup_dotted
# re-exec so the method sees lookup_dotted as a global
filter_edges = _extract("Core/WFExecution/executor.py", {"lookup_dotted", "_filter_edges_by_condition"},
                        class_name=None) if False else None

_tt = _extract("Tools/transit_tools.py", {"DATE_LIKE_RE", "_to_datetime", "is_date_in_range", "reduce", "echo_value"})
is_date_in_range, reduce_tool, echo_value = _tt["is_date_in_range"], _tt["reduce"], _tt["echo_value"]


def _build_filter():
    tree = ast.parse((ROOT / "Core/WFExecution/executor.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Executor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_filter_edges_by_condition")
    fn.decorator_list = []
    ld = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lookup_dotted")
    ns = {}
    exec(compile(ast.Module(body=[ld, fn], type_ignores=[]), "executor", "exec"), ns)
    return ns["_filter_edges_by_condition"]


filter_edges = _build_filter()


# ---------------------------------------------------------------------------
# 1. Spec integrity
# ---------------------------------------------------------------------------
specs = {n: load(f"Resources/Specs/transit_{n}.json")[n] for n in
         ("tools", "actions", "endpoints", "states", "blocks", "workflows")}
tools, actions, endpoints = specs["tools"], specs["actions"], specs["endpoints"]
states, blocks, workflows = specs["states"], specs["blocks"], specs["workflows"]

for sid, st in states.items():
    if st["parent_id"] is not None and st["parent_id"] not in states:
        err(f"state '{sid}': unknown parent '{st['parent_id']}'")

for aid, act in actions.items():
    for i, step in enumerate(act["steps"]):
        t = step["tool_name"]
        if t not in tools and t not in actions:
            err(f"action '{aid}' step {i}: unknown tool/action '{t}'")
        if t == "send_request":
            ep = step["params"]["endpoint"]["value"]
            if ep not in endpoints:
                err(f"action '{aid}' step {i}: unknown endpoint '{ep}'")
            else:
                used = set(step["params"]["add_kwargs"]["value"])
                if not used <= set(endpoints[ep]["params_in_url"]):
                    err(f"action '{aid}': kwargs {sorted(used)} not accepted by {ep}")
                missing = set(endpoints[ep]["params_in_url"]) - used
                if missing:
                    warn(f"action '{aid}': endpoint {ep} parameters not provided: {sorted(missing)}")

for bid, b in blocks.items():
    if b["action"]["id"] not in actions:
        err(f"block '{bid}': unknown action '{b['action']['id']}'")
    if b["parent_id"] not in states:
        err(f"block '{bid}': unknown parent state '{b['parent_id']}'")

for wid, wf in workflows.items():
    if wf["parent_id"] not in states:
        err(f"workflow '{wid}': unknown parent state '{wf['parent_id']}'")


class _WF:
    def __init__(self, wid, wf):
        self.id, self.title, self.blocks, self.edges = wid, wf["title"], wf["blocks"], wf["edges"]
        self._adj = {}
        for e in self.edges:
            self._adj.setdefault(e["from"], []).append(e)

    def get_adjacency(self):
        return self._adj

    def start_queue(self):
        return [e["to"] for e in self._adj.get("start", [])] or ["end"]


for rep in WorkflowValidator([_WF(w, d) for w, d in workflows.items()], known_block_ids=set(blocks)).validate_all():
    for e in rep.errors:
        err(f"{rep.workflow_id}: {e}")
    for w in rep.warnings:
        warn(f"{rep.workflow_id}: {w}")

with (ROOT / "Resources/PersonalQuestions/transit_questions.csv").open(encoding="utf-8", newline="") as f:
    qrows = list(csv.DictReader(f))
for r in qrows:
    if r["workflow_id"] not in workflows:
        err(f"question {r['id']}: unknown workflow '{r['workflow_id']}'")
for wid in workflows:
    n = sum(1 for r in qrows if r["workflow_id"] == wid)
    if n < 5:
        warn(f"{wid}: only {n} curated questions")


# ---------------------------------------------------------------------------
# 2. Scenario simulation
# ---------------------------------------------------------------------------
class Sim:
    def __init__(self, user_id, responder, replies):
        self.user_id, self.responder, self.replies = user_id, responder, list(replies)
        self.facts = {}
        self.used_endpoints = []

    # --- parameter resolution (same three arg types as the executor)
    def resolve(self, pdef, scope):
        if not isinstance(pdef, dict) or "arg_type" not in pdef:
            return pdef
        t, val = pdef["arg_type"], pdef["value"]
        if t == "value":
            return val
        if t == "param":
            cur = lookup_dotted(scope, val)
            if isinstance(cur, dict) and "value" in cur and "fieldNameEn" in cur:
                return cur["value"]
            return cur
        if t == "dict":
            return {k: self.resolve(v, scope) for k, v in val.items()}
        raise ValueError(f"unknown arg_type {t}")

    def run_action(self, name, scope, static_params=None):
        out = {}
        for step in actions[name]["steps"]:
            tool = step["tool_name"]
            params = {k: self.resolve(v, scope) for k, v in (step.get("params") or {}).items()}
            if tool in actions:  # nested action: run in its own scope and map its outputs
                nested = self.run_action(tool, {**params}, None)
                rd = step.get("return_data") or {}
                for src, dst in rd.items():
                    scope[dst] = nested[src]
                    out[dst] = nested[src]
                continue
            result = self.call_tool(tool, params)
            rd = step.get("return_data")
            if isinstance(rd, str):
                scope[rd] = result
                out[rd] = result
        return out

    def call_tool(self, tool, params):
        if tool == "get_user_id":
            return self.user_id
        if tool == "send_request":
            ep = params["endpoint"]
            self.used_endpoints.append(ep)
            return self.responder(ep)
        if tool == "get_selection_from_user":
            if not params["options"]:
                return None
            if not self.replies:
                raise RuntimeError("ran out of scripted selections")
            return int(self.replies.pop(0))
        if tool == "select_option":
            if not params["options"] or not params["selected_index"]:
                return None
            return params["options"][int(params["selected_index"]) - 1]
        if tool == "is_date_in_range":
            return is_date_in_range(**{k: v for k, v in params.items() if v is not None})
        if tool == "reduce":
            return reduce_tool(**params)
        if tool == "echo_value":
            return echo_value(**params)
        raise NotImplementedError(tool)

    def walk(self, wid):
        wf = workflows[wid]
        adj = _WF(wid, wf).get_adjacency()
        queue, path = ["start"], []
        while queue:
            node = queue.pop(0)
            if node == "end":
                continue
            if node != "start":
                path.append(node)
                self.facts.update(self.run_action(blocks[node]["action"]["id"], self.facts))
            queue.extend(e["to"] for e in filter_edges(adj.get(node, []), self.facts))
            if len(path) > 50:
                raise RuntimeError("workflow does not terminate")
        return path


def interesting(facts):
    return sorted(k for k in facts if k.endswith("_response") or k.startswith("selected_") and k.endswith("_details")
                  or k in ("is_discount_valid", "balance_after_renewal"))


def fixture_responder(mapping):
    counters = {}

    def _resp(ep):
        spec = mapping.get(ep)
        if spec is None:
            raise KeyError(f"no mock response for endpoint '{ep}'")
        if isinstance(spec, list):
            i = counters.get(ep, 0)
            counters[ep] = i + 1
            spec = spec[min(i, len(spec) - 1)]
        return load(spec)
    return _resp


riders = load("Resources/MockApi/riders.json")["riders"]
update = "--update" in sys.argv
checked = 0
facts_by_group = {}

for path in sorted((ROOT / "tests/fixtures/scenarios").rglob("*.json")):
    scen = json.loads(path.read_text(encoding="utf-8"))
    sid = scen["scenario_id"]
    exp = scen["expectations"]
    plan = scen.get("workflow_plan") or ([{"workflow_id": exp["workflow_id"],
                                           "replies": [r for r in scen["scripted_user_replies"] if isinstance(r, int)]}]
                                         if exp.get("workflow_id") else [])
    if not plan:
        continue

    if scen["mock_tools"] == 4:  # api scenario -> demo rider
        rider = riders.get(scen["user_id"])
        if rider is None:
            err(f"{sid}: rider {scen['user_id']} missing in riders.json")
            continue
        responder = lambda ep, rider=rider: rider[ep] if ep in rider else (_ for _ in ()).throw(
            KeyError(f"rider has no data for '{ep}'"))
    else:
        for ep, spec in scen["api_responses"].items():
            for sp in (spec if isinstance(spec, list) else [spec]):
                if not (ROOT / sp).exists():
                    err(f"{sid}: fixture missing: {sp}")
        responder = fixture_responder(scen["api_responses"])

    sim = Sim(scen["user_id"], responder, [])
    try:
        for step in plan:
            sim.replies = list(step["replies"])
            sim.walk(step["workflow_id"])
            if sim.replies:
                err(f"{sid}: {len(sim.replies)} scripted selection(s) were never asked for")
    except Exception as e:  # noqa: BLE001
        err(f"{sid}: simulation failed: {type(e).__name__}: {e}")
        continue

    got = interesting(sim.facts)
    # an expected key must be produced; produced response keys must be covered by fixtures
    for k in exp.get("facts_contains_keys", []):
        if k not in sim.facts:
            err(f"{sid}: expected fact '{k}' was not produced (got {got})")
    if update and scen["mock_tools"] != 5:
        exp["facts_contains_keys"] = got
        path.write_text(json.dumps(scen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if scen["mock_tools"] == 3:
        unused = set(scen["api_responses"]) - set(sim.used_endpoints)
        if unused:
            warn(f"{sid}: fixtures never requested: {sorted(unused)}")
    facts_by_group.setdefault(sid.removesuffix("_api"), []).append(set(got))
    checked += 1

for base, groups in facts_by_group.items():
    if len(groups) == 2 and groups[0] != groups[1]:
        err(f"{base}: mock and api variants produce different facts")

print(f"workflows: {len(workflows)}  blocks: {len(blocks)}  actions: {len(actions)}  endpoints: {len(endpoints)}  "
      f"questions: {len(qrows)}  scenarios simulated: {checked}")
for w in WARNINGS:
    print("WARN ", w)
for e in ERRORS:
    print("ERROR", e)
print("OK" if not ERRORS else f"{len(ERRORS)} error(s)")
sys.exit(1 if ERRORS else 0)
