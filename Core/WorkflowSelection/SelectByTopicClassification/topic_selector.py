from Core.Agents.topic_ranking_agent import TopicRankingAgent
from Core.Agents.workflow_ranking_agent import WorkflowRankingAgent
from Core.WorkflowSelection.topic_selector_base import TopicSelectorBase


class TopicSelector(TopicSelectorBase):
    """
    Existing selection-by-topic classification method (two LLM calls):
      1) TopicRankingAgent ranks topic states
      2) WorkflowRankingAgent ranks workflows within shortlisted topics
    """

    def __init__(self, get_root, get_children, get_states, get_workflows_of_state, topics_top_k: int = 5,
                 wf_top_k: int = 5, states_representation: str = "nested", cache_ttl_minutes: int = 60):
        super().__init__(
            get_root=get_root,
            get_children=get_children,
            get_states=get_states,
            get_workflows_of_state=get_workflows_of_state,
            states_representation=states_representation,
            cache_ttl_minutes=cache_ttl_minutes,
            include_state_embeddings=False,
        )

        self.state_top_k = int(topics_top_k)
        self.wf_top_k = int(wf_top_k)

        self.topic_ranker = TopicRankingAgent()
        self.wf_ranker = WorkflowRankingAgent()

    def select(self, user_query, user_query_embed, conv_history=None):
        # 1) terminal states JSON
        states_json = self._get_states_json()

        # 2) LLM #1 — rank topics
        ranked_topics = self.topic_ranker.rank(
            user_query=user_query,
            states_json=states_json,
            top_k=self.state_top_k,
            conv_history=conv_history,
        )
        if not ranked_topics:
            # Keep original semantics (caller decides fallback)
            return [], ""

        # 3) Load topic entities; collect workflows under each
        beam_state_ids = [t["topic_id"] for t in ranked_topics]
        id_state_dict = self._load_states(beam_state_ids)

        wf_candidates = self._collect_wf_candidates(
            state_ids=beam_state_ids,
            query_embed=user_query_embed,
            id_state_dict=id_state_dict,
            topic_meta=None,  # topic selector prompt does not need this
        )

        # 4) LLM #2 — rank workflows
        wf_rank_resp = self.wf_ranker.rank(
            user_query=user_query,
            wf_candidates=wf_candidates,
            top_k=self.wf_top_k,
            conv_history=conv_history,
        )

        wf_beam = [
            {
                "id": wf["workflow_id"],
                "score": float(wf.get("score", 0.0)),
                "reason": wf.get("reason", "topic_classification shortlist"),
            }
            for wf in wf_rank_resp.get("ranked_workflows", [])
        ]

        pick_id = wf_rank_resp.get("pick_id", "")
        return wf_beam, pick_id
