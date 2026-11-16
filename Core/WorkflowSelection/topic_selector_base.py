import json
import time
from typing import Any, Dict, List, Optional, Tuple

from Core.Utils.llm_utils import _cosine_rank_one_to_many


class _TTLCache:
    """
    TTL cache for terminal topic states and their JSON representations.
    We cache:
      - terminal_topics: list[dict] (rich rows)
      - flat_json / nested_json: pre-rendered JSON strings for prompt usage
    """

    def __init__(self, ttl_sec: int = 3600):
        self.ttl = int(ttl_sec)
        self._built_at = 0.0
        self.terminal_topics: List[Dict[str, Any]] = []
        self._flat: Optional[Dict[str, Any]] = None
        self._nested: Optional[Dict[str, Any]] = None

    def fresh(self) -> bool:
        return (time.time() - self._built_at) < self.ttl

    def set(self, terminal_topics: List[Dict[str, Any]], root_id: str):
        self.terminal_topics = terminal_topics
        self._flat = self._to_flat_json(terminal_topics)
        self._nested = self._to_nested_json(terminal_topics, root_id=root_id)
        self._built_at = time.time()

    @staticmethod
    def _to_flat_json(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Flat dict keyed by state_id with minimal metadata used in prompts.
        """
        flat: Dict[str, Any] = {}
        for row in rows:
            path_ids = row.get("path_ids", [])
            path_ids_str = " -> ".join(path_ids[1:]) if len(path_ids) > 1 else ""
            flat[row["state_id"]] = {
                "description": row.get("state_description", ""),
                "path": path_ids_str,
                "workflows": row.get("wf_titles", []),
            }
        return flat

    @staticmethod
    def _to_nested_json(rows: List[Dict[str, Any]], root_id: str = "root") -> Dict[str, Any]:
        """
        Build a nested dict (no artificial root).

        Each path defines a chain of parent→child nodes.
        Terminal nodes (last in each path) include: description, workflows.
        Non-terminal nodes only have: id, children.
        """
        tree: Dict[str, Any] = {}

        for row in rows:
            node = tree
            path_ids: List[str] = row.get("path_ids", [])
            for i, node_id in enumerate(path_ids):
                if node_id not in node:
                    node[node_id] = {"children": {}}

                if i == len(path_ids) - 1:
                    node[node_id].update(
                        {
                            "description": row.get("state_description", ""),
                            "workflows": row.get("wf_titles", []),
                        }
                    )
                    node[node_id].pop("children", None)
                else:
                    node = node[node_id]["children"]

        # remove artificial root wrapper
        return tree.get(root_id, {}).get("children", {})

    def get_json_str(self, fmt: str) -> str:
        if fmt == "flat":
            payload = self._flat or {}
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2)
        payload = self._nested or {}
        return json.dumps(payload, ensure_ascii=False, indent=2)


class TopicSelectorBase:
    """
    Shared base for workflow selection methods that rely on the Topic .

    Responsibilities:
      - BFS traversal over the topic to collect terminal topics (topics with workflows)
      - TTL caching of:
            * terminal topic rows
            * states_json representations (flat|nested)
      - Shared helper methods:
            * _load_states()
            * _collect_wf_candidates() (cosine rank within each topic)
    """

    def __init__(self, get_root, get_children, get_states, get_workflows_of_state,
                 states_representation: str = "nested", cache_ttl_minutes: int = 60,
                 include_state_embeddings: bool = False):
        self.get_root = get_root
        self.get_children = get_children
        self.get_states = get_states
        self.get_workflows_of_state = get_workflows_of_state

        self.repr_kind = states_representation
        self.cache = _TTLCache(ttl_sec=int(cache_ttl_minutes) * 60)
        self.include_state_embeddings = include_state_embeddings

    # -------- cache / topic graph --------

    def _ensure_terminal_topics_cached(self) -> None:
        if self.cache.fresh() and self.cache.terminal_topics:
            return
        root = self.get_root()
        terminal_rows = self._collect_terminal_topics(root)
        self.cache.set(terminal_rows, root_id=root.get_id())

    def _get_states_json(self) -> str:
        self._ensure_terminal_topics_cached()
        return self.cache.get_json_str(self.repr_kind)

    def _collect_terminal_topics(self, root_state) -> List[Dict[str, Any]]:
        """
        BFS over topic graph. Terminal topic := a state that has workflows (immediate=True).
        Each collected row includes path metadata and wf_titles.
        """
        out: List[Dict[str, Any]] = []
        q: List[Tuple[Any, List[str], List[str]]] = [(root_state, [root_state.get_id()], [root_state.get_title()])]
        seen = {root_state.get_id()}

        while q:
            curr_state, curr_path_ids, curr_path_titles = q.pop(0)

            # enqueue children
            for child in (self.get_children(state=curr_state) or []):
                cid = child.get_id()
                if cid in seen:
                    continue
                seen.add(cid)
                q.append((child, curr_path_ids + [cid], curr_path_titles + [child.get_title()]))

            # terminal?
            wfs = self.get_workflows_of_state(state=curr_state, immediate=True) or []
            if wfs:
                row: Dict[str, Any] = {
                    "state_id": curr_state.get_id(),
                    "state_title": curr_state.get_title(),
                    # "state_description": curr_state.get_description(),
                    "path_ids": curr_path_ids,
                    "path_titles": curr_path_titles,
                    "wf_titles": [wf.get_title() for wf in wfs],
                }
                if self.include_state_embeddings:
                    # defensive: some state objects may not have embeddings
                    try:
                        row["state_embedding"] = curr_state.get_embedding()
                    except Exception:
                        row["state_embedding"] = []
                out.append(row)

        return out

    # -------- shared helpers --------

    def _load_states(self, ids: List[str]) -> Dict[str, Any]:
        """
        Maps state_id -> state entity (first result from get_states()).
        """
        id_state_dict: Dict[str, Any] = {}
        for state_id in ids:
            states = self.get_states(state_id) or []
            if states:
                id_state_dict[state_id] = states[0]
        return id_state_dict

    def _collect_wf_candidates(
            self,
            *,
            state_ids: List[str],
            query_embed,
            id_state_dict: Dict[str, Any],
            topic_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect workflows under each topic state and compute cosine similarity between
        query embedding and workflow embedding. Also attaches optional topic metadata.

        Returns wf_candidates as list of dicts (compatible with existing ranking agents).
        """
        wf_candidates: List[Dict[str, Any]] = []
        topic_meta = topic_meta or {}

        for state_id in state_ids:
            topic_state = id_state_dict.get(state_id)
            if not topic_state:
                continue

            wfs = self.get_workflows_of_state(state=topic_state, immediate=True) or []
            if not wfs:
                continue

            order, sims = _cosine_rank_one_to_many(query_embed, [wf.get_embedding() for wf in wfs])
            ranked = [(wfs[i], float(sims[i])) for i in order]

            meta = topic_meta.get(state_id, {})
            for wf, sim in ranked:
                wf_candidates.append(
                    {
                        "id": wf.get_id(),
                        "title": wf.get_title(),
                        "description": wf.get_description(),
                        "cos_sim": sim,
                        # Optional topic context for unified selector prompts
                        "topic_id": state_id,
                        "topic_path": meta.get("path", ""),
                        "topic_description": meta.get("description", ""),
                        "topic_sim": float(meta.get("score", 0.0) or 0.0),
                    }
                )

        return wf_candidates
