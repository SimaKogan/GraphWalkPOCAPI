import os
import re
import json
import glob
from datetime import datetime

import pandas as pd


# ---------------- Helpers ----------------

def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_str(v):
    if v is None:
        return None
    return str(v)


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_case_id_from_scenario_id(scenario_id: str):
    """
    Try to parse:
      - wf_10_10000
      - wf_2/wf_2_2000_success
      - wf_7/wf_7_7000_happy_path
    Returns (wf_id:int|None, idx:int|None, case_name:str)
    """
    s = scenario_id.replace("\\", "/").strip()

    # Prefer last segment as the "case name"
    case_name = s.split("/")[-1]

    # Try to find wf + idx anywhere in the string
    m = re.search(r"wf[_\-\s]?(\d+)[^\d]+(\d{3,})", case_name, flags=re.IGNORECASE)
    if not m:
        # fallback: try the full scenario string
        m = re.search(r"wf[_\-\s]?(\d+)[^\d]+(\d{3,})", s, flags=re.IGNORECASE)

    wf_id = int(m.group(1)) if m else None
    idx = int(m.group(2)) if m else None
    return wf_id, idx, case_name


def _find_latest_traceability_xlsx(traceability_root: str) -> str | None:
    """
    Find latest traceability_*.xlsx under traceability_root (recursive).
    """
    candidates = glob.glob(os.path.join(traceability_root, "**", "traceability_*.xlsx"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_endpoint_to_block_maps(traceability_xlsx_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Read traceability wiring matrix and map:
      - endpoint_name -> block_title
      - endpoint_name -> block_id
    If multiple rows exist for same endpoint, keep the first non-empty value.
    """
    wiring = pd.read_excel(traceability_xlsx_path, sheet_name="wiring_matrix")

    ep_to_block_title: dict[str, str] = {}
    ep_to_block_id: dict[str, str] = {}

    for _, row in wiring.iterrows():
        ep = row.get("endpoint_name")
        bt = row.get("block_title")
        bid = row.get("block_id")

        if pd.isna(ep) or ep is None:
            continue
        ep = str(ep)

        if ep not in ep_to_block_title and not pd.isna(bt) and bt:
            ep_to_block_title[ep] = str(bt)

        if ep not in ep_to_block_id and not pd.isna(bid) and bid:
            ep_to_block_id[ep] = str(bid)

    return ep_to_block_title, ep_to_block_id


def _discover_scenarios(scenarios_dir: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(scenarios_dir, "**", "*.json"), recursive=True):
        try:
            data = _read_json(path)
        except Exception as e:
            rows.append({
                "scenario_json_path": _norm_path(path),
                "scenario_id": None,
                "error": f"Failed to read JSON: {e}",
            })
            continue

        scenario_id = data.get("scenario_id") or os.path.splitext(os.path.basename(path))[0]
        wf_id, idx, case_name = _extract_case_id_from_scenario_id(scenario_id)

        rows.append({
            "scenario_id": scenario_id,
            "case_name": case_name,
            "wf_id": wf_id,
            "idx": idx,
            "description": data.get("description") or data.get("title") or data.get("name"),
            "first_query": data.get("first_query"),
            "final_answer": data.get("expectations").get("gt_answer") or None,  # optional field if you add later
            "answer_must_include": "; ".join((data.get("expectations") or {}).get("answer_must_include", []) or []),
            "answer_must_exclude": "; ".join((data.get("expectations") or {}).get("answer_must_exclude", []) or []),
            "facts_contains_keys": "; ".join((data.get("expectations") or {}).get("facts_contains_keys", []) or []),
            "route_expected": (data.get("expectations") or {}).get("route"),
            "workflow_id_expected": (data.get("expectations") or {}).get("workflow_id"),
            "api_responses": data.get("api_responses") or {},
            "scripted_user_replies": "; ".join([str(x) for x in data.get("scripted_user_replies") or []]),
            "scenario_json_path": _norm_path(path),
            "error": None,
        })
    return rows


def _discover_test_files(tests_root: str) -> list[dict]:
    """
    Find pytest files and extract scenario ids referenced, e.g.:
        scenario_runner("wf_2/wf_2_2000_success")
    """
    out = []
    py_files = glob.glob(os.path.join(tests_root, "**", "test_*.py"), recursive=True)
    for p in py_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception:
            continue

        # capture scenario_runner("...") and scenario_runner('...')
        matches = re.findall(r"scenario_runner\(\s*[rRuU]?[\"']([^\"']+)[\"']\s*\)", txt)
        for raw_id in matches:
            norm_id = raw_id.replace("\\", "/").split("/")[-1]
            out.append({
                "scenario_id": norm_id,
                "test_case_path": _norm_path(p),
            })

    # If same scenario appears multiple times, keep the first path
    uniq = {}
    for r in out:
        uniq.setdefault(r["scenario_id"], r["test_case_path"])
    return [{"scenario_id": k, "test_case_path": v} for k, v in uniq.items()]


def _build_execution_order_blocks(
        api_responses: dict,
        ep_to_block_title: dict[str, str],
        ep_to_block_id: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Derive execution order from the order of keys in the scenario's api_responses mapping.
    Return:
      - block_titles: [..]
      - block_ids:    [..]
      - endpoints:    [..]
    """
    if not isinstance(api_responses, dict) or not api_responses:
        return [], [], []

    endpoints_order = list(api_responses.keys())

    block_titles: list[str] = []
    block_ids: list[str] = []

    for ep in endpoints_order:
        bt = ep_to_block_title.get(ep)
        bid = ep_to_block_id.get(ep)

        block_titles.append(bt if bt else f"[UNKNOWN BLOCK for {ep}]")
        block_ids.append(bid if bid else f"[UNKNOWN BLOCK_ID for {ep}]")

    return block_titles, block_ids, endpoints_order


def build_tests_traceability_df(
        scenarios_dir: str,
        tests_root: str,
        traceability_xlsx_path: str | None
) -> pd.DataFrame:
    scenario_rows = _discover_scenarios(scenarios_dir)
    tests_rows = _discover_test_files(tests_root)

    scen_df = pd.DataFrame(scenario_rows)
    tests_df = pd.DataFrame(tests_rows)

    ep_to_block_title = {}
    ep_to_block_id = {}
    if traceability_xlsx_path and os.path.exists(traceability_xlsx_path):
        ep_to_block_title, ep_to_block_id = _load_endpoint_to_block_maps(traceability_xlsx_path)

    # Build execution order columns (as lists)
    blocks_titles_list = []
    blocks_ids_list = []
    endpoints_list = []

    for _, row in scen_df.iterrows():
        titles, ids, endpoints = _build_execution_order_blocks(
            row.get("api_responses") or {},
            ep_to_block_title,
            ep_to_block_id,
        )
        blocks_titles_list.append(titles)
        blocks_ids_list.append(ids)
        endpoints_list.append(endpoints)

    scen_df["wf_execution_order_blocks"] = blocks_titles_list
    scen_df["wf_execution_order_block_ids"] = blocks_ids_list
    scen_df["api_endpoints_order"] = endpoints_list

    # Fill final_answer if not explicitly provided: fall back to expectations snippets
    scen_df["final_answer"] = scen_df["final_answer"].fillna(scen_df["answer_must_include"])

    # Recommended core columns + extra useful metadata
    col_order = [
        "idx",
        "case_name",
        "wf_id",
        "scenario_id",
        "description",
        "first_query",
        "final_answer",
        "wf_execution_order_blocks",
        "wf_execution_order_block_ids",
        "test_case_path",
        "scenario_json_path",
        "route_expected",
        "workflow_id_expected",
        "facts_contains_keys",
        "answer_must_include",
        "answer_must_exclude",
        "api_endpoints_order",
        "scripted_user_replies",
        "error",
    ]

    # Make sure idx / wf_id first exist
    if "idx" not in scen_df.columns:
        scen_df["idx"] = None
    if "wf_id" not in scen_df.columns:
        scen_df["wf_id"] = None

    for c in col_order:
        if c not in scen_df.columns:
            scen_df[c] = None

    scen_df = scen_df[col_order].sort_values(by=["wf_id", "idx", "case_name"], na_position="last")
    return scen_df


def save_tests_traceability(df: pd.DataFrame, out_xlsx: str) -> None:
    _ensure_dir(os.path.dirname(out_xlsx))
    with pd.ExcelWriter(out_xlsx) as writer:
        df.to_excel(writer, index=False, sheet_name="tests_traceability")
    print(f"✔ Excel saved: {out_xlsx}")


# ---------------- Run (PyCharm style) ----------------
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

resources_path = os.path.join(repo_root, "Resources")
traceability_root = os.path.join(resources_path, "Traceability")
tests_root = os.path.join(repo_root, "tests")
scenarios_dir = os.path.join(tests_root, "fixtures", "scenarios")

# Pick latest traceability_*.xlsx automatically (you can override by setting traceability_xlsx_path directly)
traceability_xlsx_path = _find_latest_traceability_xlsx(traceability_root)

# Output: timestamped folder under Resources/Traceability/
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(traceability_root, f"tests/{ts}")
_ensure_dir(output_dir)
output_xlsx = os.path.join(output_dir, f"tests_traceability_{ts}.xlsx")

df = build_tests_traceability_df(
    scenarios_dir=scenarios_dir,
    tests_root=tests_root,
    traceability_xlsx_path=traceability_xlsx_path,
)

save_tests_traceability(df, output_xlsx)
