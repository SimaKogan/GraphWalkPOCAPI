from copy import deepcopy
from typing import Dict, Any, List, Set, Optional


def _collect_params_literals(params):
    if not isinstance(params, dict):
        return {}
    out = {}
    for k, v in params.items():
        if isinstance(v, dict) and "arg_type" in v:
            if v["arg_type"] == "value":
                out[k] = v.get("value")
            elif v["arg_type"] == "dict":
                out[k] = _collect_params_literals(v.get("value", {}))
        elif isinstance(v, dict):
            out[k] = _collect_params_literals(v)
    return out


def _collect_param_refs(params, acc=None):
    acc = acc or set()
    if not isinstance(params, dict):
        return acc
    for _, v in params.items():
        if isinstance(v, dict) and v.get("arg_type") == "param":
            acc.add(v.get("value"))
        elif isinstance(v, dict) and v.get("arg_type") == "dict":
            _collect_param_refs(v.get("value", {}), acc)
        elif isinstance(v, dict):
            _collect_param_refs(v, acc)
    return acc


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# =========================
# Block / Action metadata
# =========================
def _return_data_targets(return_data) -> List[str]:
    if return_data is None:
        return []
    if isinstance(return_data, str):
        return [return_data]
    if isinstance(return_data, dict):
        return [v if isinstance(v, str) else str(v) for v in return_data.values()]
    if isinstance(return_data, (list, tuple)):
        out: List[str] = []
        for item in return_data:
            out.extend(_return_data_targets(item))
        return out
    return [str(return_data)]


def build_block_metadata(block, action_registry):
    action_id = (block.get("action") or {}).get("id")
    expanded_steps = action_registry.get_action_steps(action_id, expanded=True) if action_id else []
    tools, endpoints, produces, needs = set(), set(), set(), set()

    for step in expanded_steps:
        name = step.get("name")
        tools.add(name)

        for rd in _return_data_targets(step.get("return_data")):
            produces.add(rd)

        if name == "send_request":
            ep = step.get("params", {}).get("endpoint", {})
            if isinstance(ep, dict) and ep.get("arg_type") == "value":
                endpoints.add(ep.get("value"))

        if name in {"query_user", "ask_user_input"}:
            needs.update(_return_data_targets(step.get("return_data")))

        needs.update(_collect_param_refs(step.get("params", {})))

    static_params = _collect_params_literals((block.get("action") or {}).get("params", {}) or {})

    title = block.get("title", "").strip()
    desc = (block.get("description") or "").strip()
    human = title if title else block.get("id", "")
    if desc:
        human += f" — {desc}"
    if endpoints:
        human += f" [endpoints: {', '.join(sorted(endpoints))}]"
    if static_params:
        human += f" [static: {static_params}]"

    return {
        "block_id": block.get("id"),
        "title": title,
        "description": desc,
        "action_id": action_id,
        "static_params": static_params,
        "tools_used": sorted(tools),
        "endpoints_called": sorted(endpoints),
        "facts_produced": sorted(produces),
        "needs_inputs": sorted(needs),
        "summary": human,
    }


# =========================
# Workflow metadata
# =========================

def build_workflow_metadata(wf, blocks, states, action_registry):
    blocks_meta = []
    facts_all, endpoints_all, tools_all, needs_all = set(), set(), set(), set()

    for node in wf.get("blocks", []):
        bid = node.get("id")
        base_block = blocks.get(bid, {})
        bmeta = build_block_metadata(base_block, action_registry)

        node.update({
            "title": bmeta["title"],
            "description": bmeta["description"],
            "action_id": bmeta["action_id"],
            "metadata": {
                "tools_used": bmeta["tools_used"],
                "endpoints_called": bmeta["endpoints_called"],
                "facts_produced": bmeta["facts_produced"],
                "static_params": bmeta["static_params"],
                "needs_inputs": bmeta["needs_inputs"],
            },
        })
        blocks_meta.append(bmeta)
        facts_all.update(bmeta["facts_produced"])
        endpoints_all.update(bmeta["endpoints_called"])
        tools_all.update(bmeta["tools_used"])
        needs_all.update(bmeta["needs_inputs"])

    decision_keys = []
    for e in wf.get("edges", []):
        cond = e.get("condition") or {}
        on = cond.get("on")
        if on:
            decision_keys.append(on)

    # Contextual path (topic/subtopic) if available
    parent_state_id = wf.get("parent_id")
    parent_state = ""
    if states and parent_state_id in states:
        state = states[parent_state_id]
        parent_state = "Title:" + state.get("title") + ": " + state.get("description", parent_state_id)

    # concise human_summary
    steps_preview = " → ".join([b["title"] or b["block_id"] for b in blocks_meta]) or "start → end"
    human_summary = f"Steps: {steps_preview}."
    if endpoints_all:
        human_summary += f" Touches endpoints: {', '.join(sorted(endpoints_all))}."
    if decision_keys:
        human_summary += f" Decisions on: {', '.join(decision_keys)}."

    # detailed selection_summary
    lines: List[str] = []
    lines.append(f"Workflow: {wf.get('title', '')}".strip())
    lines.append(human_summary)
    lines.append("Blocks:")
    for b in blocks_meta:
        line = f"- {b['title'] or b['block_id']}"
        if b["description"]:
            line += f": {b['description']}"
        lines.append(line)
    lines.append(f"Parent Topic: {parent_state}")
    selection_summary = "\n".join(lines).strip()

    return {
        "blocks_meta": blocks_meta,
        "tools_all": sorted(tools_all),
        "endpoints_all": sorted(endpoints_all),
        "facts_all": sorted(facts_all),
        "needs_all": sorted(needs_all),
        "decision_keys": decision_keys,
        "human_summary": human_summary,
        "selection_summary": selection_summary,
    }


# =========================
# Top-level enricher
# =========================

def enrich_data_specs_with_metadata(data_specs, action_registry, append_summaries_to_description=True):
    ds = deepcopy(data_specs)
    blocks = ds.get("blocks", [])
    workflows = ds.get("workflows", [])
    states = ds.get("states", {})

    for wf_id, wf in workflows.items():
        meta = build_workflow_metadata(wf, blocks, states, action_registry)
        wf["metadata"] = meta

        if append_summaries_to_description:
            parts: List[str] = []
            base_desc = (wf.get("description") or "").strip()
            if base_desc:
                parts.append(base_desc)
            # parts.append(f"[summary] {meta['human_summary']}".strip())
            parts.append(f"\nsummary:\n{meta['selection_summary']}".strip())
            wf["description"] = "\n\n".join(parts).strip()

    return ds
