import os
from datetime import datetime
import pandas as pd
import re
import json

from DataLayer.data_module import load_dict


# ---------------- Helpers ----------------

def safe_id(name: str) -> str:
    """Return a Mermaid-safe node ID (ASCII only)."""
    return "id_" + re.sub(r'[^a-zA-Z0-9_]', "_", str(name))


def safe_label(text: str) -> str:
    """Clean labels for Mermaid: remove problematic quotes."""
    if text is None:
        return ""
    # Replace double quotes with single quotes to avoid parse errors
    return text.replace('"', "'")


def extract_endpoints_from_action(actions, action_name, seen=None):
    """Recursively collect endpoints an action calls via send_request."""
    if seen is None:
        seen = set()
    if action_name in seen or action_name not in actions:
        return set()
    seen.add(action_name)

    eps = set()
    for step in actions[action_name].get("steps", []):
        tool = step.get("tool_name")
        params = step.get("params", {}) or {}
        if tool == "send_request":
            ep_param = params.get("endpoint", {})
            val = ep_param.get("value") if isinstance(ep_param, dict) else None
            if isinstance(val, str):
                eps.add(val)
        elif tool in actions:  # nested action
            eps |= extract_endpoints_from_action(actions, tool, seen)
    return eps


def normalize_notes(v):
    if isinstance(v, list):
        return "; ".join(v)
    return str(v) if v else None


# ---------------- Traceability ----------------

def build_traceability(data):
    """Build traceability matrix and all catalogs."""
    actions = data.get("actions", {})
    blocks = data.get("blocks", {})
    workflows = data.get("workflows", {})
    endpoints = data.get("endpoints", {})
    states = data.get("states", {})

    # Map action -> endpoints
    action_to_eps = {a: sorted(extract_endpoints_from_action(actions, a)) for a in actions}

    rows = []
    # Include every workflow
    for wf_id, wf in workflows.items():
        for blk in wf.get("blocks", []):
            blk_id = blk["id"]
            blk_def = blocks.get(blk_id, {})
            action_id = blk_def.get("action", {}).get("id") if blk_def else None
            action_params = blk_def.get("action", {}).get("params", {}) if blk_def else {}
            eps_used = action_to_eps.get(action_id, [])

            if not eps_used:  # no endpoint
                rows.append({
                    "workflow_id": wf_id,
                    "workflow_title": wf.get("title"),
                    "block_id": blk_id,
                    "block_title": blk_def.get("title") if blk_def else None,
                    "action_id": action_id,
                    "action_params": json.dumps(action_params, ensure_ascii=False),
                    "endpoint_name": None,
                    "endpoint_method": None,
                    "endpoint_url": None,
                    "endpoint_keys": None,
                    "endpoint_returns": None,
                    "endpoint_notes": None
                })
            else:
                for ep in eps_used:
                    ep_def = endpoints.get(ep, {})
                    rows.append({
                        "workflow_id": wf_id,
                        "workflow_title": wf.get("title"),
                        "block_id": blk_id,
                        "block_title": blk_def.get("title") if blk_def else None,
                        "action_id": action_id,
                        "action_params": json.dumps(action_params, ensure_ascii=False),
                        "endpoint_name": ep,
                        "endpoint_method": ep_def.get("method"),
                        "endpoint_url": ep_def.get("end_point"),
                        "endpoint_keys": ", ".join(ep_def.get("json_requestBody", [])),
                        "endpoint_returns": ", ".join(ep_def.get("main_return_keys", [])),
                        "endpoint_notes": normalize_notes(ep_def.get("notes")),
                    })

    wiring_df = pd.DataFrame(rows)

    # Unused blocks
    used_blocks = set(wiring_df["block_id"].dropna())
    all_blocks = set(blocks.keys())
    unused_blocks = all_blocks - used_blocks
    unused_blocks_df = pd.DataFrame([{
        "block_id": b,
        "title": blocks[b].get("title"),
        "action_id": blocks[b].get("action", {}).get("id")
    } for b in sorted(unused_blocks)])

    # Unused endpoints
    used_eps = set(wiring_df["endpoint_name"].dropna().unique())
    all_eps = set(endpoints.keys())
    unused_eps = all_eps - used_eps
    unused_eps_df = pd.DataFrame([{
        "endpoint_name": ep,
        "method": endpoints[ep].get("method"),
        "url": endpoints[ep].get("end_point"),
        "keys": ", ".join(endpoints[ep].get("json_requestBody", [])),
        "returns": ", ".join(endpoints[ep].get("main_return_keys", [])),
        "notes": normalize_notes(endpoints[ep].get("notes")),
    } for ep in sorted(unused_eps)])

    # Actions catalog
    actions_df = pd.DataFrame([{
        "action": a,
        "endpoints": ", ".join(eps),
        "n_endpoints": len(eps)
    } for a, eps in action_to_eps.items()])

    # Endpoints catalog (all endpoints)
    endpoints_df = pd.DataFrame([{
        "endpoint_name": ep,
        "method": endpoints[ep].get("method"),
        "url": endpoints[ep].get("end_point"),
        "keys": ", ".join(endpoints[ep].get("json_requestBody", [])),
        "returns": ", ".join(endpoints[ep].get("main_return_keys", [])),
        "notes": normalize_notes(endpoints[ep].get("notes")),
    } for ep in sorted(endpoints.keys())])

    # States catalog (all states)
    states_df = pd.DataFrame([{
        "state_id": sid,
        "title": sdef.get("title"),
        "parent": sdef.get("parent_id"),
        "description": sdef.get("description"),
        "kind": sdef.get("kind"),
        "level": sdef.get("level"),
    } for sid, sdef in states.items()])

    # Blocks catalog (all blocks)
    blocks_df = pd.DataFrame([{
        "block_id": bid,
        "title": bdef.get("title"),
        "parent": bdef.get("parent_id"),
        "action_id": bdef.get("action", {}).get("id"),
        "description": bdef.get("description"),
    } for bid, bdef in blocks.items()])

    return wiring_df, unused_blocks_df, unused_eps_df, actions_df, endpoints_df, states_df, blocks_df


def save_traceability(wiring_df, unused_blocks_df, unused_eps_df,
                      actions_df, endpoints_df, states_df, blocks_df,
                      xlsx_path="traceability.xlsx", csv_path="traceability.csv"):
    with pd.ExcelWriter(xlsx_path) as writer:
        wiring_df.to_excel(writer, index=False, sheet_name="wiring_matrix")
        unused_blocks_df.to_excel(writer, index=False, sheet_name="unused_blocks")
        unused_eps_df.to_excel(writer, index=False, sheet_name="unused_endpoints")
        actions_df.to_excel(writer, index=False, sheet_name="actions_catalog")
        endpoints_df.to_excel(writer, index=False, sheet_name="endpoints_catalog")
        states_df.to_excel(writer, index=False, sheet_name="states_catalog")
        blocks_df.to_excel(writer, index=False, sheet_name="blocks_catalog")
    wiring_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✔ Excel saved: {xlsx_path}")
    print(f"✔ CSV saved:   {csv_path}")


# ---------------- Mermaid Hierarchy ----------------

def build_mermaid_hierarchy(data: dict) -> str:
    """Generate Mermaid diagram with states and blocks (all included)."""
    states = data["states"]
    blocks = data["blocks"]

    lines = ["flowchart TD", '    root[root : root]']

    # Map parent->children states
    children = {}
    for sid, sdef in states.items():
        parent = sdef.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(sid)

    # Add state nodes
    def add_state_node(state_id: str):
        node_id = safe_id(state_id)
        title = states[state_id].get("title", state_id)
        label = safe_label(f"{state_id} : {title}")
        lines.append(f'    {node_id}["{label}"]')
        return node_id

    # Add first-level children of root
    for sid in children.get("root", []):
        node_id = add_state_node(sid)
        lines.append(f'    root --> {node_id}')

        def recurse(parent_id, parent_node):
            for child_id in children.get(parent_id, []):
                child_node = add_state_node(child_id)
                lines.append(f'    {parent_node} --> {child_node}')
                recurse(child_id, child_node)

        recurse(sid, node_id)

    # Add blocks under their parent_id using subgraphs
    parent_to_blocks = {}
    for bid, bdef in blocks.items():
        parent = bdef.get("parent_id")
        parent_to_blocks.setdefault(parent, []).append((bid, bdef))

    for parent, b_list in parent_to_blocks.items():
        if not b_list:
            continue
        sg_name = f'sg_{safe_id(parent)}'
        lines.append(f'    subgraph {sg_name} [ ]')
        lines.append('      direction TB')
        for bid, bdef in b_list:
            node_id = safe_id(bid)
            title = bdef.get("title", "")
            label = safe_label(f"{bid} : {title}")
            lines.append(f'      {node_id}["{label}"]')
        lines.append("    end")
        parent_node = safe_id(parent)
        lines.append(f'    {parent_node} --> {sg_name}')

    return "\n".join(lines)


# ---
def esc(s: str) -> str:
    """Escape backslashes then double quotes for DOT labels."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def safe_name(s: str) -> str:
    """Create ASCII-only ID (letters, numbers, underscore)."""
    return re.sub(r'[^0-9A-Za-z_]', '_', s)


def workflows_to_dot(workflows, dot_file):
    dot = []
    dot.append("digraph Workflows {")
    dot.append('  rankdir=LR;')
    dot.append('  graph [fontname="Arial"];')
    dot.append('  node  [shape=box, fontname="Arial"];')
    dot.append('  edge  [fontname="Arial"];')

    def format_single_cond(cond):
        """
        Human-readable condition text matching executor semantics.
        Supports: equals, null, not_null, empty, not_empty.
        """
        # With explicit "on" field (preferred format)
        if "on" in cond:
            field = str(cond.get("on", ""))

            if "equals" in cond:
                val = cond.get("equals")
                return f"{field} == {val}"

            if cond.get("null") is True:
                return f"{field} is null"

            if cond.get("not_null") is True:
                return f"{field} is not null"

            if cond.get("empty") is True:
                return f"{field} is empty"

            if cond.get("not_empty") is True:
                return f"{field} is not empty"

            # Fallback: show remaining keys
            others = {k: v for k, v in cond.items() if k != "on"}
            if others:
                parts = [f"{field}.{k} = {v}" for k, v in others.items()]
                return ", ".join(parts)
            return field

        # Legacy / generic key-value format
        parts = [f"{k} = {v}" for k, v in cond.items()]
        return ", ".join(parts)

    def build_condition_label(or_conds, and_conds):
        or_labels = [format_single_cond(c) for c in or_conds] if or_conds else []
        and_labels = [format_single_cond(c) for c in and_conds] if and_conds else []

        if not or_labels and not and_labels:
            return None

        if or_labels and and_labels:
            return f"({' OR '.join(or_labels)}) AND ({' AND '.join(and_labels)})"
        if or_labels:
            return " OR ".join(or_labels)
        return " AND ".join(and_labels)

    for wf_id, wf in workflows.items():
        title = wf.get("title", wf_id)
        cluster_id = safe_name(wf_id)
        label_text = esc(f"{wf_id}: {title}")

        dot.append(f'  subgraph "cluster_{cluster_id}" {{')
        dot.append(f'    label="{label_text}";')
        dot.append('    labelloc="t";')
        dot.append('    fontname="Arial";')

        # Start / End nodes
        start_id = f"{cluster_id}_start"
        end_id = f"{cluster_id}_end"
        dot.append(f'    "{start_id}" [label="Start", shape=circle, style=filled, fillcolor=lightgreen];')
        dot.append(f'    "{end_id}" [label="End", shape=doublecircle, style=filled, fillcolor=lightblue];')

        # Map original block IDs to safe IDs
        id_map = {}
        for block in wf.get("blocks", []):
            orig = block["id"]
            node_id = f"{cluster_id}_{safe_name(orig)}"
            id_map[orig] = node_id
            dot.append(f'    "{node_id}" [label="{esc(orig)}", shape=box];')

        # Helper for resolving edge endpoints
        def resolve(name: str) -> str:
            if name == "start":
                return start_id
            if name == "end":
                return end_id
            return id_map.get(name, f"{cluster_id}_{safe_name(name)}")

        # Add edges
        for edge in wf.get("edges", []):
            src = resolve(edge["from"])
            dst = resolve(edge["to"])

            or_conds = edge.get("or_conditions") or []
            and_conds = edge.get("and_conditions") or []
            lbl = build_condition_label(or_conds, and_conds)

            if lbl:
                dot.append(f'    "{src}" -> "{dst}" [label="{esc(lbl)}"];')
            else:
                dot.append(f'    "{src}" -> "{dst}";')

        dot.append("  }")  # end cluster

    dot.append("}")

    with open(dot_file, "w", encoding="utf-8") as f:
        f.write("\n".join(dot))

    print(f"DOT file written to {dot_file}")


def workflows_to_mermaid(workflows, mermaid_file, *, direction="LR", compact_ids=True, strip_quotes=True,
                         safe_name_fn=None, esc_fn=None):
    """
    Convert workflows dict -> Mermaid flowchart text and write to file.
    """

    # --- helpers -------------------------------------------------------------
    def _default_safe_name(s):
        out = re.sub(r'\W+', '_', str(s), flags=re.UNICODE)
        out = re.sub(r'_+', '_', out).strip('_')
        return out or 'id'

    def _normalize_label_text(s):
        text = str(s)
        if strip_quotes:
            text = text.replace('"', '')  # remove straight quotes
            text = text.replace('\\"', '')  # remove escaped form
        text = text.replace('`', "'")
        text = text.replace('\r', ' ').replace('\n', ' ')
        return text

    def _default_esc(s):
        return _normalize_label_text(s).replace('\\', '\\\\')

    safe_name = safe_name_fn or _default_safe_name
    esc = esc_fn or _default_esc

    def _short_id(cluster_id, orig_id):
        if not compact_ids:
            return f"{cluster_id}_{safe_name(orig_id)}"
        m = re.search(r'\d+', str(orig_id))
        if m:
            return f"{cluster_id}_{m.group(0)}"
        return f"{cluster_id}_{safe_name(orig_id)}"

    def _wf_header(wf_id, title):
        cid = safe_name(wf_id)
        label = esc(f"{wf_id}: {title}")
        return f'subgraph {cid}["{label}"]', cid

    def _format_single_cond(cond):
        """
        Human-readable condition text matching executor semantics.
        Supports: equals, null, not_null, empty, not_empty.
        """
        if "on" in cond:
            field = str(cond.get("on", ""))

            if "equals" in cond:
                val = cond.get("equals")
                return f"{field} == {val}"

            if cond.get("null") is True:
                return f"{field} is null"

            if cond.get("not_null") is True:
                return f"{field} is not null"

            if cond.get("empty") is True:
                return f"{field} is empty"

            if cond.get("not_empty") is True:
                return f"{field} is not empty"

            others = {k: v for k, v in cond.items() if k != "on"}
            if others:
                parts = [f"{field}.{k} = {v}" for k, v in others.items()]
                return ", ".join(parts)
            return field

        parts = [f"{k} = {v}" for k, v in cond.items()]
        return ", ".join(parts)

    def _build_condition_label(or_conds, and_conds):
        or_labels = [_format_single_cond(c) for c in or_conds] if or_conds else []
        and_labels = [_format_single_cond(c) for c in and_conds] if and_conds else []

        if not or_labels and not and_labels:
            return None

        if or_labels and and_labels:
            return f"({' OR '.join(or_labels)}) AND ({' AND '.join(and_labels)})"
        if or_labels:
            return " OR ".join(or_labels)
        return " AND ".join(and_labels)

    def _edge(src, dst, or_conds=None, and_conds=None):
        label = _build_condition_label(or_conds or [], and_conds or [])
        if not label:
            return f"  {src} --> {dst}"
        # Mermaid edge with label
        return f"  {src} -->|{esc(label)}| {dst}"

    # --- build lines ---------------------------------------------------------
    lines = []
    lines.append(f"flowchart {direction}")

    for wf_id, wf in workflows.items():
        title = wf.get("title", wf_id)
        header, cluster_id = _wf_header(wf_id, title)

        lines.append("")
        lines.append(f"%% {cluster_id.replace('_', ' ')}")
        lines.append(header)

        # Start / End
        start_id = f"{cluster_id}_start"
        end_id = f"{cluster_id}_end"
        lines.append(f'  {start_id}(("Start"))')
        lines.append(f'  {end_id}((("End")))')

        # Nodes
        id_map = {}
        for block in wf.get("blocks", []):
            orig = block["id"]
            node_id = _short_id(cluster_id, orig)
            id_map[orig] = node_id
            lines.append(f'  {node_id}["{esc(orig)}"]')

        # Resolve endpoints
        def resolve(name):
            if name == "start":
                return start_id
            if name == "end":
                return end_id
            return id_map.get(name, _short_id(cluster_id, name))

        # Edges
        for edge in wf.get("edges", []):
            src = resolve(edge["from"])
            dst = resolve(edge["to"])
            or_conds = edge.get("or_conditions") or []
            and_conds = edge.get("and_conditions") or []
            lines.append(_edge(src, dst, or_conds, and_conds))

        lines.append("end")

    with open(mermaid_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Mermaid file written to {mermaid_file}")


# ---
base_path = "../Resources"
data_specs = {
    "states": load_dict(f"{base_path}/Specs/transit_states.json",
                        encoding="utf-8")["states"],
    "blocks": load_dict(f"{base_path}/Specs/transit_blocks.json",
                        encoding="utf-8")["blocks"],
    "workflows": load_dict(f"{base_path}/Specs/transit_workflows.json",
                           encoding="utf-8")["workflows"],
    "actions": load_dict(f"{base_path}/Specs/transit_actions.json",
                         encoding="utf-8")["actions"],
    "endpoints": load_dict(f"{base_path}/Specs/transit_endpoints.json",
                           encoding="utf-8")["endpoints"],
}

output_path = base_path + f"/Traceability/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_path, exist_ok=True)

# Step 2: Build traceability + catalogs
(wiring_df, unused_blocks_df, unused_eps_df,
 actions_df, endpoints_df, states_df, blocks_df) = build_traceability(data_specs)
save_traceability(wiring_df, unused_blocks_df, unused_eps_df,
                  actions_df, endpoints_df, states_df, blocks_df,
                  xlsx_path=f"{output_path}\\traceability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                  csv_path=f"{output_path}\\traceability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Step 3: Build Mermaid diagram
mermaid_text = build_mermaid_hierarchy(data_specs)
with open(
        f"{output_path}\\mermaid_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mmd",
        "w") as f:
    f.write(mermaid_text)
print("Mermaid diagram saved to file 'mermaid_diagram.mmd'")

# Step 4: Build DOT file
workflows_to_dot(data_specs["workflows"],
                 f"{output_path}\\workflows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dot")
workflows_to_mermaid(workflows=data_specs["workflows"],
                     mermaid_file=f"{output_path}\\workflows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mmd")
