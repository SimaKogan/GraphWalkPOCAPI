from dataclasses import dataclass, field
from typing import Dict, List, Any, Iterable, Optional, Tuple, Set, Callable


# ---------- Public report types ----------

@dataclass
class WFReport:
    workflow_id: str
    title: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors


@dataclass
class ValidatorPolicy:
    """
    Policy knobs:
    - allow_multi_uncond_from_start: allow multiple unconditional edges out of 'start' (serialized execution)
    - allow_multi_uncond_elsewhere: if False, emit ERROR for >1 unconditional out-edges on non-start nodes;
      if True, only WARN (executor will serialize, but it may reflect unintended "parallel")
    """
    allow_multi_uncond_from_start: bool = True
    allow_multi_uncond_elsewhere: bool = False


# ---------- Workflow-like protocol (duck typing) ----------

class WorkflowLike:
    id: str
    title: Optional[str]
    blocks: List[Dict[str, Any]]

    def get_adjacency(self) -> Dict[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    def start_queue(self) -> List[str]:
        raise NotImplementedError


# ---------- Core validator ----------

class WorkflowValidator:
    """
    Validate a collection of workflows against:
      - structural topology (start/end reachability, acyclicity)
      - edge correctness (references, duplicates)
      - unconditional branching policy (no parallelism)
      - block existence (edges reference declared blocks; blocks exist in system)
      - basic condition sanity (optional)
    """

    def __init__(
        self,
        workflows: Iterable[WorkflowLike],
        known_block_ids: Optional[Set[str]] = None,
        policy: Optional[ValidatorPolicy] = None,
        condition_root_whitelist: Optional[Set[str]] = None,  # e.g., {"response"}
    ):
        self.workflows = list(workflows)
        self.known_block_ids = set(known_block_ids or set())
        self.policy = policy or ValidatorPolicy()
        self.condition_root_whitelist = condition_root_whitelist or {"response"}

    # --- Entry point ---

    def validate_all(self) -> List[WFReport]:
        reports: List[WFReport] = []
        for wf in self.workflows:
            rep = self._validate_one(wf)
            reports.append(rep)
        return reports

    # --- Single workflow ---

    def _validate_one(self, wf: WorkflowLike) -> WFReport:
        rep = WFReport(workflow_id=wf.id, title=getattr(wf, "title", None))

        node_ids = {b["id"] for b in (wf.blocks or [])}
        adj = wf.get_adjacency()

        # 1) Start/end presence
        if "start" not in adj:
            rep.errors.append("No edges from 'start'.")
            return rep  # further checks depend on a start
        if not self._has_edge_to_end(adj):
            rep.errors.append("No edge leading to 'end'.")
        # Derive all referenced node ids from edges
        edge_node_ids = self._referenced_nodes(adj)

        # 2) Edge references & block membership
        for frm, edges in adj.items():
            if frm not in ("start", "end") and frm not in node_ids:
                rep.errors.append(f"Edge 'from' references unknown block: '{frm}'.")
            for e in edges:
                to = e.get("to")
                if to not in ("start", "end") and to not in node_ids:
                    rep.errors.append(f"Edge 'to' references unknown block: '{to}'.")

        # 3) Blocks exist in the system (if a global set was provided)
        if self.known_block_ids:
            missing = node_ids - self.known_block_ids
            if missing:
                rep.errors.append(f"Blocks listed in workflow not found in system: {sorted(missing)}")

        # 4) Unused blocks (declared but not touched by any edge)
        used_blocks = {n for n in edge_node_ids if n not in ("start", "end")}
        unused = node_ids - used_blocks
        if unused:
            rep.warnings.append(f"Unused blocks (declared but not referenced by edges): {sorted(unused)}")

        # 5) Duplicate edges
        dups = self._find_duplicate_edges(adj)
        if dups:
            rep.warnings.append(f"Duplicate edges detected: {sorted(list(dups))}")

        # 6) Unconditional branching policy (no parallelism)
        for frm, outs in adj.items():
            uncond = [e for e in outs if not self._is_conditional(e)]
            if frm == "start":
                if len(uncond) > 1 and not self.policy.allow_multi_uncond_from_start:
                    rep.errors.append("Multiple unconditional edges out of 'start' not allowed by policy.")
            else:
                if len(uncond) > 1:
                    if self.policy.allow_multi_uncond_elsewhere:
                        rep.warnings.append(
                            f"Node '{frm}' has {len(uncond)} unconditional out-edges (serialized, but double-check intent)."
                        )
                    else:
                        rep.errors.append(
                            f"Node '{frm}' has {len(uncond)} unconditional out-edges (violates no-parallel policy)."
                        )

        # 7) Condition sanity (basic)
        for frm, outs in adj.items():
            for e in outs:
                cond = e.get("condition")
                if not cond:
                    continue
                on = cond.get("on")
                if not on or not isinstance(on, str) or "." not in on:
                    rep.warnings.append(
                        f"Edge {frm}→{e.get('to')} has odd condition 'on': {on!r} (expected dotted path)."
                    )
                else:
                    root = on.split(".", 1)[0]
                    if self.condition_root_whitelist and root not in self.condition_root_whitelist:
                        rep.warnings.append(
                            f"Condition on '{on}' uses root '{root}', not in whitelist {self.condition_root_whitelist}."
                        )
                # equals is free-form; normalize booleans happens at runtime

        # 8) Reachability + cycles
        reach_ok, unreachable = self._reachability(node_ids, adj)
        if unreachable:
            rep.warnings.append(f"Unreachable blocks from 'start': {sorted(unreachable)}")
        if not reach_ok:
            rep.errors.append("Graph contains a cycle.")

        return rep

    # --- Helpers ---

    @staticmethod
    def _is_conditional(edge: Dict[str, Any]) -> bool:
        """An edge is conditional if it has a legacy 'condition' or or_/and_ conditions."""
        return bool(edge.get("condition") or edge.get("or_conditions") or edge.get("and_conditions"))

    @staticmethod
    def _has_edge_to_end(adj: Dict[str, List[Dict[str, Any]]]) -> bool:
        for edges in adj.values():
            for e in edges:
                if e.get("to") == "end":
                    return True
        return False

    @staticmethod
    def _referenced_nodes(adj: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
        ref: Set[str] = set()
        for frm, edges in adj.items():
            ref.add(frm)
            for e in edges:
                ref.add(e.get("to"))
        return ref

    @staticmethod
    def _find_duplicate_edges(adj: Dict[str, List[Dict[str, Any]]]) -> Set[Tuple[str, str, str]]:
        seen: Set[Tuple[str, str, str]] = set()
        dups: Set[Tuple[str, str, str]] = set()
        for frm, edges in adj.items():
            for e in edges:
                to = e.get("to")
                cond_sig = WorkflowValidator._condition_signature(
                    e.get("condition") or {k: e[k] for k in ("or_conditions", "and_conditions") if e.get(k)})
                sig = (frm, to, cond_sig)
                if sig in seen:
                    dups.add(sig)
                else:
                    seen.add(sig)
        return dups

    @staticmethod
    def _condition_signature(cond: Optional[Dict[str, Any]]) -> str:
        if not cond:
            return "UNCOND"
        # deterministic string signature (order-independent)
        items = sorted((k, str(v)) for k, v in cond.items())
        return "|".join(f"{k}={v}" for k, v in items)

    @staticmethod
    def _reachability(node_ids: Set[str], adj: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, Set[str]]:
        """
        Returns (acyclic, unreachable_nodes)
        Treats 'end' as sink and does not traverse past it.
        """
        # DFS for cycle detection
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def dfs(n: str) -> bool:
            if n == "end":
                return True
            if n in visiting:
                return False  # cycle
            if n in visited:
                return True
            visiting.add(n)
            ok = True
            for e in adj.get(n, []):
                if not dfs(e.get("to")):
                    ok = False
                    break
            visiting.remove(n)
            visited.add(n)
            return ok

        acyclic = dfs("start")

        # Reachability from 'start' to blocks
        seen: Set[str] = set()
        def walk(n: str):
            if n in seen or n == "end":
                return
            seen.add(n)
            for e in adj.get(n, []):
                walk(e.get("to"))

        walk("start")
        unreachable = {b for b in node_ids if b not in seen}
        return acyclic, unreachable


# ---------- Convenience adapters ----------

def validate_from_entities(
    iter_workflows: Callable[[], Iterable[WorkflowLike]],
    known_block_ids: Optional[Set[str]] = None,
    policy: Optional[ValidatorPolicy] = None,
    condition_root_whitelist: Optional[Set[str]] = None,
) -> List[WFReport]:
    """
    Use this after the DB is populated. Pass a function that yields Workflow entities
    (e.g., EntitiesManager.get_all_workflows or a small wrapper over graph_manager).
    """
    validator = WorkflowValidator(
        workflows=iter_workflows(),
        known_block_ids=known_block_ids,
        policy=policy,
        condition_root_whitelist=condition_root_whitelist,
    )
    return validator.validate_all()


def validate_from_data_specs(
    workflows_json: List[Dict[str, Any]],
    known_block_ids: Optional[Set[str]] = None,
    policy: Optional[ValidatorPolicy] = None,
    condition_root_whitelist: Optional[Set[str]] = None,
) -> List[WFReport]:
    """
    Use this before DB load (pure JSON). We wrap dicts with a tiny shim to look like WorkflowLike.
    """

    class _WFShim(WorkflowLike):
        def __init__(self, d: Dict[str, Any]):
            self.id = d["id"]
            self.title = d.get("title")
            self.blocks = d.get("blocks", [])
            self._edges = d.get("edges", [])

        def get_adjacency(self) -> Dict[str, List[Dict[str, Any]]]:
            adj: Dict[str, List[Dict[str, Any]]] = {}
            for e in self._edges:
                adj.setdefault(e["from"], []).append(e)
            return adj

        def start_queue(self) -> List[str]:
            adj = self.get_adjacency()
            return [e["to"] for e in adj.get("start", [])] or ["end"]

    validator = WorkflowValidator(
        workflows=[_WFShim(w) for w in workflows_json],
        known_block_ids=known_block_ids,
        policy=policy,
        condition_root_whitelist=condition_root_whitelist,
    )
    return validator.validate_all()


# How to use it
# A) Validate after DB population (recommended)
#
# Add this once EntitiesManager finished creating/populating the DB:
# # In Orchestrator._init_traversal_framework (AFTER entities_manager.init_graph_db)
# from Core.validator.workflow_validator import validate_from_entities, ValidatorPolicy
#
# def _iter_all_workflows():
#     # Implement a tiny iterator using your graph manager
#     # e.g., EntitiesManager.get_workflows(state=None) → or add a dedicated get_all_workflows()
#     return (wf for wf in entities_manager.get_workflows(state=None))  # adjust to your API
#
# reports = validate_from_entities(
#     iter_workflows=_iter_all_workflows,
#     known_block_ids={b["id"] for b in data_specs["blocks"]},
#     policy=ValidatorPolicy(
#         allow_multi_uncond_from_start=True,   # multiple start edges serialized
#         allow_multi_uncond_elsewhere=False,   # forbid unintended parallelism
#     ),
#     condition_root_whitelist={"response"},    # edge conditions should typically reference response.*
# )
#
# for r in reports:
#     if not r.ok():
#         print(f"[WF INVALID] {r.workflow_id} - {r.title}:")
#         for e in r.errors: print("  ERROR:", e)
#         for w in r.warnings: print("  WARN :", w)
#     else:
#         for w in r.warnings:
#             print(f"[WF WARN] {r.workflow_id}: {w}")

#---
#B) Validate before DB (pure JSON)
# from Core.validator.workflow_validator import validate_from_data_specs, ValidatorPolicy
#
# reports = validate_from_data_specs(
#     workflows_json=data_specs["workflows"],
#     known_block_ids={b["id"] for b in data_specs["blocks"]},
#     policy=ValidatorPolicy(allow_multi_uncond_from_start=True, allow_multi_uncond_elsewhere=False),
#     condition_root_whitelist={"response"},
# )
