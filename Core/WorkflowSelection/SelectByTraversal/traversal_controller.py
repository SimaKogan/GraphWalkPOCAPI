import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from Core.Agents.traversal_selector_agent import TraversalSelectionAgent


def _cosine_rank_one_to_many(query_embed, embeddings: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not embeddings:
        return np.array([], dtype=int), np.array([], dtype=float)
    sims = cosine_similarity([query_embed], embeddings)[0]
    order = np.argsort(-sims)
    return order, sims


def wf_preview_for_state(state, get_workflows_of_state, top_n=5, immediate=True, seen_wf=None):
    """
    Collect workflow titles under a state for look-ahead previews.
    Deduplicates against workflows already previewed in ancestor states.

    :param state: State object
    :param get_workflows_of_state: function to retrieve workflows
    :param top_n: max number of workflow titles to return
    :param immediate: True = only direct workflows; False = recursive (may overlap)
    :param seen_wf: set of workflow ids already previewed along this path
    """
    seen_wf = seen_wf or set()
    wfs = get_workflows_of_state(state=state, immediate=immediate) or []  # NOTE: Add check if state or workflow?

    titles = []
    for wf in wfs:
        if wf.get_id() in seen_wf:
            continue
        seen_wf.add(wf.get_id())
        titles.append(wf.get_title())  # TODO: enrich with more fields - description (from metadata enrichment)
        if len(titles) >= top_n:
            break
    return titles


def _pack_state_candidate(state, sim, rel_depth, path_ids, wf_preview) -> Dict[str, Any]:
    return {
        "id": state.get_id(),
        "title": state.get_title(),
        "description": state.get_description(),
        "sim": float(sim),
        "rel_depth": int(rel_depth),
        "path_ids": path_ids,
        "workflows_preview": wf_preview or []
    }


def _pack_workflow_candidate(wf, sim) -> Dict[str, Any]:
    return {
        "id": wf.get_id(),
        "title": wf.get_title(),
        "description": wf.get_description(),
        "sim": float(sim),
    }


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class TraversalController:
    """
      - Agentic traversal (beam + look-ahead + hop).
      - SelectByTraversal returns a workflow CANDIDATE + BEAM and sets phase 'critic_workflow'.
      - CriticAgent owns all clarifications and final approval.

    Executor contract preserved: only starts when cursor.phase == 'selected_workflow'.
    """

    def __init__(self, selector_agent=None, beam_width: int = 5, lookahead_depth: int = 2,
                 wf_preview_top_n: int = 5, immediate_wf_preview: bool = True):
        self.selector = selector_agent or TraversalSelectionAgent()
        self.beam_width = beam_width
        self.lookahead_depth = lookahead_depth
        self.preview_n = wf_preview_top_n
        self.immediate_wf_preview = immediate_wf_preview

    # -------- public (resumable) --------
    def route(self, get_root, get_children, get_states, get_workflows_of_state, facts, cursor, conv_history=None):
        """
        Run the traversal loop until completion (done flag set).
        Calls `step` repeatedly until traversal ends in a terminal phase.

        :param get_root: function to get the root state
        :param get_children: function to get the children of a state
        :param get_states: function to retrieve states by id
        :param get_workflows_of_state: function to get workflows of a state
        :param facts: Context memory dict (facts accumulated along the conversation)
        :param cursor: Cursor dict (current traversal pointer and phase)
        :param conv_history: Optional bounded conversation history for LLM context.
        :return: A dict containing {"done": bool, "facts": dict, "cursor": dict}
        """
        while True:
            out = self.step(get_root, get_children, get_states, get_workflows_of_state, facts, cursor, conv_history)
            if out.get("done"):
                return out

    def step(self, get_root, get_children, get_states, get_workflows_of_state, facts, cursor, conv_history=None):
        """
        Step the traversal agent forward given the current state of the traversal.
        Handles different phases:
          * None or 'choose_child' → expand children/frontier, pick next state
          * 'choose_workflow' or 'critic_workflow' → select candidate workflow and hand off to critic
          * Other phases → reset to 'choose_child'

        :param get_root: function to get the root state
        :param get_children: function to get the children of a state
        :param get_states: function to retrieve states by id
        :param get_workflows_of_state: function to get workflows of a state
        :param facts: Context memory dict
        :param cursor: Cursor dict
        :param conv_history: Optional bounded conversation history for LLM context.
        :return: A dict containing facts and cursor; may include {"done": True} when traversal ends
        """
        if cursor.get("state_id") is None:
            return self._begin(get_root, facts, cursor)

        curr_state = get_states(cursor["state_id"])[0]
        phase = cursor.get("phase")

        match phase:
            case None | "choose_child":
                return self._choose_child(curr_state, get_children, get_workflows_of_state, facts, cursor, conv_history)
            case "choose_workflow" | "critic_workflow":
                return self._choose_workflow(curr_state, get_workflows_of_state, facts, cursor, conv_history)
            case _:
                return self._reset_phase(cursor, facts)

    # -------- phases --------
    def _begin(self, get_root, facts, cursor):
        """
        Initialize traversal from the root state.

        :param get_root: function to get the root state
        :param facts: Context memory dict
        :param cursor: Cursor dict (will be updated with root state and phase)
        :return: dict with updated facts and cursor
        """
        root = get_root()
        cursor.update({
            "state_id": root.get_id(),
            "path": [root.get_id()],
            "phase": "choose_child",
            "phase_idx": 0
        })
        return {"facts": facts, "cursor": cursor}

    def _reset_phase(self, cursor, facts):
        """
       Reset traversal phase to 'choose_child' when cursor is in an unexpected state.

       :param cursor: Cursor dict
       :param facts: Context memory dict
       :return: dict with updated facts and cursor
       """
        cursor["phase"] = "choose_child"
        cursor["phase_idx"] = 0
        return {"facts": facts, "cursor": cursor}

    def _enumerate_frontier(self, curr_state, get_children, get_workflows_of_state):
        """
        Build the frontier of candidate states for beam search, up to depth lookahead_depth.

        Uses BFS to collect children, grandchildren, etc., along with metadata
        (path_ids, relative depth, workflow previews). Deduplicates previews so
        a workflow is not shown in both an ancestor and descendant preview

        :param curr_state: The current state object
        :param get_children: function to get the children of a state
        :param get_workflows_of_state: function to get workflows of a state (for look-ahead previews)
        :return: (frontier, lookahead_map)
                 frontier = list of {"obj": state entity, "path_ids": list, "rel_depth": int}
                 lookahead_map = dict id -> {rel_depth, path_ids, wf_preview}
        """
        frontier: List[Dict[str, Any]] = []
        lookahead_map: Dict[str, Dict[str, Any]] = {}

        queue: List[Tuple[Any, List[str], int, set]] = []

        lvl1 = get_children(state=curr_state) or []
        for child in lvl1:
            queue.append((child, [curr_state.get_id(), child.get_id()], 1, set()))

        seen_nodes = set()
        while queue:
            entity, path_ids, depth, seen_wf = queue.pop(0)
            if entity.get_id() in seen_nodes:
                continue
            seen_nodes.add(entity.get_id())

            if entity.get_embedding() is not None:
                wf_preview = wf_preview_for_state(
                    entity,
                    get_workflows_of_state,
                    top_n=self.preview_n,
                    immediate=self.immediate_wf_preview,
                    seen_wf=seen_wf
                )  # TODO: what if workflow? filters out before?
                frontier.append({
                    "obj": entity,
                    "path_ids": path_ids,
                    "rel_depth": depth
                })
                lookahead_map[entity.get_id()] = {
                    "rel_depth": depth,
                    "path_ids": path_ids,
                    "wf_preview": wf_preview,
                }

            if depth < self.lookahead_depth:
                for child in (get_children(state=entity) or []):
                    # propagate a copy of seen_wf so descendants inherit ancestor previews
                    queue.append((child, path_ids + [child.get_id()], depth + 1, set(seen_wf)))
        return frontier, lookahead_map

    def _choose_child(self, curr_state, get_children, get_workflows_of_state, facts, cursor, conv_history=None):
        """
        Choose the next state (child or deeper descendant) using beam search and LLM selector.

        1. Enumerate frontier up to depth D (look-ahead).
        2. Rank by cosine similarity to query embedding.
        3. Shortlist top-k candidates (beam).
        4. Pass shortlist to LLM selector to re-rank and pick.
        5. Update cursor to chosen state (supports hopping multiple layers).

        :param curr_state: The current state object
        :param get_children: function to get children of the current state
        :param get_workflows_of_state: function to get workflows (for previews)
        :param facts: Context memory dict (must include "query_embed")
        :param cursor: Cursor dict (updated with new state_id and path)
        :param conv_history: Conversation history (for LLM selector)
        :return: dict with updated facts and cursor
        """
        frontier, lookahead_map = self._enumerate_frontier(curr_state, get_children, get_workflows_of_state)

        if not frontier:
            cursor["phase"] = "choose_workflow"
            cursor["phase_idx"] = 0
            return {"facts": facts, "cursor": cursor, "note": "leaf: choose_workflow"}

        order, sims = _cosine_rank_one_to_many(facts["query_embed"],
                                               [child["obj"].get_embedding() for child in frontier])
        ranked = [(frontier[i], float(sims[i])) for i in order]
        shortlist = ranked[: self.beam_width]

        cand_lst = []
        for (child, sim) in shortlist:
            cand_lst.append(_pack_state_candidate(child["obj"], sim, child["rel_depth"], child["path_ids"],
                                                  lookahead_map[child["obj"].get_id()]["wf_preview"]))

        cand_json = _j(cand_lst)

        look_dict = {}
        for child_id, look_map in lookahead_map.items():
            look_dict[child_id] = {"rel_depth": look_map["rel_depth"], "path_ids": look_map["path_ids"],
                                   "wf_preview": look_map["wf_preview"]}
        look_json = _j(look_dict)

        sel = self.selector.select(
            query=facts.get("query_text") or "",
            stage="child",
            path_text=" / ".join(cursor.get("path", [])),
            candidates_json=cand_json,
            lookahead_json=look_json,
            conv_history=conv_history,
        ) or {}

        pick_id = sel.get("pick_id") or shortlist[0][0]["obj"].get_id()
        meta = lookahead_map.get(pick_id)
        if meta:
            # hop merge
            cur_path = cursor.get("path") or []
            tail = meta["path_ids"][1:]  # skip the first because it repeats current
            cursor.update({"state_id": pick_id, "path": cur_path + tail, "phase": "choose_child", "phase_idx": 0})
        else:
            cursor.update({"state_id": pick_id, "path": (cursor.get("path") or []) + [pick_id],
                           "phase": "choose_child", "phase_idx": 0})

        # If next state has no children, next step will switch to choose_workflow
        return {"facts": facts, "cursor": cursor}

    def _choose_workflow(self, curr_state, get_workflows_of_state, facts, cursor, conv_history=None):
        """
        Select a candidate workflow from the current state.

        1. Retrieve workflows of the state.
        2. Rank by cosine similarity, keep top-k.
        3. Pass shortlist to LLM selector for re-ranking and pick.
        4. Update cursor to phase 'critic_workflow' with candidate workflow id.
        5. Store the beam in facts["wf_beam"] for the critic.

        :param curr_state: Current state object
        :param get_workflows_of_state: function to get workflows of the state
        :param facts: Context memory dict (must include "query_embed")
        :param cursor: Cursor dict
        :param conv_history: Conversation history (for LLM selector)
        :return: dict with facts and cursor; includes "done": True to hand off to critic
        """
        wfs = get_workflows_of_state(state=curr_state) or []
        if not wfs:
            return {"done": True, "facts": facts, "cursor": cursor, "note": "no workflows at state"}

        order, sims = _cosine_rank_one_to_many(facts["query_embed"], [wf.get_embedding() for wf in wfs])
        ranked = [(wfs[i], float(sims[i])) for i in order]
        shortlist = ranked[: self.beam_width]

        cand_json = _j([_pack_workflow_candidate(wf, sim) for (wf, sim) in shortlist])
        sel = self.selector.select(
            query=facts.get("query_text") or "",
            stage="workflow",
            path_text=" / ".join(cursor.get("path", [])),
            candidates_json=cand_json,
            lookahead_json="",
            conv_history=conv_history
        ) or {}

        pick_id = sel.get("pick_id") or shortlist[0][0].get_id()
        beam = sel.get("beam") or [{"id": wf.get_id(), "score": sim, "reason": "cosine shortlist"} for (wf, sim) in
                                   shortlist]

        # ---- hand off to critic ----
        cursor.update({
            "traversal_cand_wf_id": pick_id,
            "phase": "critic_workflow",  # critic owns clarifications and final approval
            "phase_idx": 0
        })
        facts["wf_beam"] = beam
        return {"done": True, "facts": facts, "cursor": cursor}
