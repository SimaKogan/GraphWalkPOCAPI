from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def aggregate_candidates(topic_select_wf_beam, retrieved_wf_by_quests, fetch_connected_questions, get_workflow_fn,
                         get_parent_state_fn, query_embed: List[float] | None, per_wf_limit: int = 5):
    """
    Aggregate two already-sorted candidate sources by workflow, without re-ranking.
    For workflows present in retrieved_by_wf, reuse those retrieved curated questions as the
    `connected_questions` block (with their vector-search similarity scores).
    Only for Topic-based selection only workflows, fetch connected curated questions from the graph and
    compute cosine(query, question.embedding).
    """
    if not topic_select_wf_beam and not (retrieved_wf_by_quests or {}):
        return []
    if query_embed is None:
        raise ValueError("query_embed is None")
    q_vec = np.asarray(query_embed, dtype=np.float32)
    cand_wfs = {}

    # 1) seed with topic selector order
    for rank, item in enumerate(topic_select_wf_beam, start=1):
        wf_id = item.get("id")
        if not wf_id:
            continue
        cand_wfs[wf_id] = {
            "id": wf_id,
            "source": "Topic Classification",
            "topic_class": {
                "rank": rank,
                "score": item.get("score"),
                "reason": item.get("reason", "") #TODO: get from wf selector more fields.
            },
            "questions_similarity": None,
        }
    #
    # 2) merge question-similarity block and *reuse* retrieved questions as connected examples
    for wf_id, qsim_block in (retrieved_wf_by_quests or {}).items():
        if wf_id in cand_wfs:
            cand_wfs[wf_id]["source"] = "Topic-based + Question Similarity"
            cand_wfs[wf_id]["questions_similarity"] = qsim_block
        else:
            cand_wfs[wf_id] = {
                "id": wf_id,
                "source": "Question Similarity",
                "topic_class": None,
                "questions_similarity": qsim_block,
            }

    # 3) for Topic-based selection only workflows, fetch connected curated questions and compute cosine(query, q.emb)
    for wf_id, row in cand_wfs.items():
        if row.get("questions_similarity") is not None:
            continue  # already filled from retrieved vector hits
        quests = fetch_connected_questions(wf_id, limit=per_wf_limit) or []
        quest_lst = []
        for quest in quests:
            sim = float(cosine_similarity([q_vec], [quest.get_embedding(as_np=True)]).squeeze())
            quest_lst.append({"question": quest, "score": sim})

        quest_lst = sorted(quest_lst, key=lambda x: -x["score"])

        row["questions_similarity"] = {
            "connected_questions_with_sim_scores": quest_lst[:per_wf_limit],
            "max_sim_score": max([q["score"] for q in quest_lst], default=0.0),
            "avg_sim_score": (np.mean([q["score"] for q in quest_lst]) if quest_lst else 0.0),
            "count": len(quest_lst)
        }

    # 4) add Objects - Workflow & Parent State
    for wf_id, row in cand_wfs.items():
        row["workflow_obj"] = get_workflow_fn(wf_id)[0]
        row["parent_state_obj"] = get_parent_state_fn(wf_id)[0]

    # 5) output order: Topic-based selection first (by rank), then question-only by their max_sim_score desc
    topic_class_ids = [it["id"] for it in topic_select_wf_beam if it.get("id")]
    question_only = [wf for wf in cand_wfs.values() if wf["id"] not in topic_class_ids]
    question_only.sort(key=lambda r: -((r["questions_similarity"] or {}).get("max_sim_score", 0.0)))

    ordered = [cand_wfs[wid] for wid in topic_class_ids] + question_only
    return ordered


def format_candidates(wfs_cand):
    wfs_cand_format = []
    for wf_cand_dict in wfs_cand:
        format_quests = []
        qs = wf_cand_dict.get("questions_similarity") or {}
        quests_dicts = qs.get("connected_questions_with_sim_scores", [])
        for quest_dict in quests_dicts:
            format_quest = quest_dict["question"].format_string()
            format_quest += f"\nSimilarity: {quest_dict['score']}"
            format_quests.append(format_quest)

        orig_qs = wf_cand_dict.get("questions_similarity") or {}

        formatted_qs = {
            **orig_qs,
            "connected_questions_with_sim_scores": format_quests
        }

        wf_cand_format = {
            "workflow": wf_cand_dict["workflow_obj"].format_string(),
            "selection_source": wf_cand_dict["source"],
            "questions_similarity": formatted_qs,
        }

        if wf_cand_dict.get("topic_class") is not None:
            wf_cand_format["topic_class"] = wf_cand_dict["topic_class"]

        wfs_cand_format.append(wf_cand_format)
    return wfs_cand_format
