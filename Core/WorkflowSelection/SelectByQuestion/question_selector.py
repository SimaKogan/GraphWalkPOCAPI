from typing import List, Dict, Any

import numpy as np


class QuestionSelector:
    """
    Thin wrapper over EntitiesManager/GraphManager to:
      1) embed the query (already in sessions.facts["query_embed"])
      2) run ANN on CuratedQuestion index
      3) return normalized hits
    """

    def __init__(self, search_top_k_fn, k: int = 8,
                 vector_index_name="question_embedding_index"):
        self.search_top_k_fn = search_top_k_fn
        self.k = max(1, int(k))
        self.index = vector_index_name  # optional override

    def retrieve(self, query_embed, norm_scores_fn=None):
        retrieved_quests = self.search_top_k_fn(query_embed, k=self.k, vector_index_name=self.index)
        # normalize scores to [0,1] for fusion (Min_max normalization)
        if norm_scores_fn:
            norm_scores_fn(retrieved_quests)
        return retrieved_quests

    @staticmethod
    def summarize_questions_by_workflow(retrieved_quests, top_n_per_wf: int = 5) -> Dict[str, Dict[str, Any]]:
        by_wf = {}
        for retrieved_quest in retrieved_quests:
            by_wf.setdefault(retrieved_quest["question"].get_handler_workflow_id(), []).append(retrieved_quest)

        summary = {}
        for wf_id, quest_lst in by_wf.items():
            quest_lst = sorted(quest_lst, key=lambda x: -x["score"])
            summary[wf_id] = {
                "max_sim_score": max([q["score"] for q in quest_lst], default=0.0),
                "avg_sim_score": (np.mean([q["score"] for q in quest_lst]) if quest_lst else 0.0),
                "connected_questions_with_sim_scores": quest_lst[:top_n_per_wf],
                "count": len(quest_lst),
            }
        return summary

    def retrieve_and_summarize(self, query_embed, norm_scores_fn=None, per_wf_limit=5):
        retrieved_quests = self.retrieve(query_embed, norm_scores_fn)
        retrieved_workflows = self.summarize_questions_by_workflow(retrieved_quests, top_n_per_wf=per_wf_limit)
        quest_cand_wf_id = (max(retrieved_workflows,
                                key=lambda x: retrieved_workflows[x]["max_sim_score"]) if retrieved_workflows else "")
        return retrieved_workflows, quest_cand_wf_id


def min_max_norm_scores(dict_with_scores):
    if not dict_with_scores:
        return
    scores = [element.get("score", 0.0) for element in dict_with_scores]
    min_score, max_score = min(scores), max(scores)
    denom = float(max_score - min_score)
    if denom == 0.0:
        for element in dict_with_scores:
            element["score"] = 1.0
        return
    for element in dict_with_scores:
        element["score"] = (element.get("score", 0.0) - min_score) / denom


def z_score_norm_scores(dict_with_scores):
    if not dict_with_scores:
        return
    scores = np.array([elem.get("score", 0.0) for elem in dict_with_scores], dtype=float)
    mean = scores.mean()
    std = scores.std()

    if std == 0.0:
        for elem in dict_with_scores:
            elem["score"] = 1.0
        return

    z_scores = (scores - mean) / std

    for elem, z in zip(dict_with_scores, z_scores):
        elem["score"] = float(z)
