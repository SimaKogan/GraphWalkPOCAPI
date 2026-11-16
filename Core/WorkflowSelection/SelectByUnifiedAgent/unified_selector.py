from typing import Any, Dict, List, Optional

from Core.Agents.unified_workflow_selection_agent import UnifiedWorkflowSelectionAgent
from Core.Utils.llm_utils import _cosine_rank_one_to_many
from Core.WorkflowSelection.topic_selector_base import TopicSelectorBase


class UnifiedWorkflowSelector(TopicSelectorBase):
    """
    Unified selection method (single LLM call for workflow ranking).
    Uses topic graph only to:
      - build states_json context
      - optionally prefilter topics by embedding similarity

    If topics_top_k is None (or <=0), no topic filtering is performed (all terminal topics included).
    """

    def __init__(
            self,
            get_root,
            get_children,
            get_states,
            get_workflows_of_state,
            topics_top_k: Optional[int] = None,
            wf_top_k: int = 3,
            states_representation: str = "nested",
            cache_ttl_minutes: int = 60,
    ):
        super().__init__(
            get_root=get_root,
            get_children=get_children,
            get_states=get_states,
            get_workflows_of_state=get_workflows_of_state,
            states_representation=states_representation,
            cache_ttl_minutes=cache_ttl_minutes,
            include_state_embeddings=True,  # needed for optional topic embedding ranking
        )

        # When None or <=0 => include all terminal topics
        self.state_top_k = None if topics_top_k is None else int(topics_top_k)
        if self.state_top_k is not None and self.state_top_k <= 0:
            self.state_top_k = None

        self.wf_top_k = int(wf_top_k)
        self.wf_ranker = UnifiedWorkflowSelectionAgent()

    def select(self, user_query, user_query_embed, conv_history=None):
        # Ensure cache and build states_json
        states_json = self._get_states_json()

        # Determine candidate topic states
        ranked_topics = self._rank_terminal_topics_by_embedding(user_query_embed, self.state_top_k)

        # If we didn't rank (edge cases), fall back to "all"
        if not ranked_topics:
            self._ensure_terminal_topics_cached()
            ranked_topics = [
                {
                    "topic_id": r["state_id"],
                    "score": 0.0,
                    "reason": "fallback_all_topics",
                    "path": " -> ".join((r.get("path_ids") or [])[1:]),
                    "description": r.get("state_description", ""),
                }
                for r in (self.cache.terminal_topics or [])
            ]

        beam_state_ids = [t["topic_id"] for t in ranked_topics]
        id_state_dict = self._load_states(beam_state_ids)

        topic_meta = {t["topic_id"]: t for t in ranked_topics}

        wf_candidates = self._collect_wf_candidates(
            state_ids=beam_state_ids,
            query_embed=user_query_embed,
            id_state_dict=id_state_dict,
            topic_meta=topic_meta,
        )
        if not wf_candidates:
            return [], ""

        # Deterministic ordering for prompt stability: highest cos_sim first, then title
        wf_candidates_sorted = sorted(
            wf_candidates, key=lambda x: (-float(x.get("cos_sim", 0.0)), str(x.get("title", "")))
        )

        wf_rank_resp = self.wf_ranker.rank(
            user_query=user_query,
            states_json=states_json,
            wf_candidates=wf_candidates_sorted,
            top_k=self.wf_top_k,
            conv_history=conv_history,
        )

        wf_beam = [
            {
                "id": wf["workflow_id"],
                "score": float(wf.get("score", 0.0)),
                "reason": wf.get("reason", "unified selector"),
            }
            for wf in wf_rank_resp.get("ranked_workflows", [])
        ]

        pick_id = wf_rank_resp.get("pick_id", "")
        return wf_beam, pick_id

    # -------- internals --------

    def _rank_terminal_topics_by_embedding(self, query_embed, top_k: Optional[int]) -> List[Dict[str, Any]]:
        """
        Rank terminal topics by cosine similarity between query embedding and state embedding.

        If top_k is None => include all (still ranked if embeddings exist).
        If embeddings are missing => return in cached order (up to k).
        """
        self._ensure_terminal_topics_cached()
        rows = self.cache.terminal_topics or []
        if not rows:
            return []

        k = len(rows) if top_k is None else min(max(int(top_k), 1), len(rows))

        embs = [r.get("state_embedding") or [] for r in rows]
        if not any(embs):
            # Embeddings missing; preserve deterministic ordering
            ranked_rows = rows[:k]
            return [
                {
                    "topic_id": r["state_id"],
                    "score": 0.0,
                    "reason": "state_embedding_missing",
                    "path": " -> ".join((r.get("path_ids") or [])[1:]),
                    "description": r.get("state_description", ""),
                }
                for r in ranked_rows
            ]

        order, sims = _cosine_rank_one_to_many(query_embed, embs)

        ranked: List[Dict[str, Any]] = []
        for i in order[:k]:
            r = rows[i]
            ranked.append(
                {
                    "topic_id": r["state_id"],
                    "score": float(sims[i]),
                    "reason": "topic_embedding_similarity",
                    "path": " -> ".join((r.get("path_ids") or [])[1:]),
                    "description": r.get("state_description", ""),
                }
            )
        return ranked
