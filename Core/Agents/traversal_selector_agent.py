from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions


class TraversalSelectionAgent(PromptBasedAgent):
    """
    LLM selector for traversal stages. It re-ranks a small candidate beam and returns:
      {
        "stage": "child"|"workflow",
        "pick_id": str,
        "beam": [{"id": str, "score": float, "reason": str}]
      }
    """

    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="TraversalSelectionAgent", prompt_type="traversal_selector", llm_options=llm_options)

    def select(self, query: str, stage: str, path_text: str,
               candidates_json: str, lookahead_json: str | None = None, conv_history=None):
        """
        Call the LLM to re-rank traversal candidates.

        :param query: User's natural language query.
        :param stage: SelectByTraversal stage, either "child" (select next state) or "workflow".
        :param path_text: Short breadcrumb of the current path (for context).
        :param candidates_json: JSON string containing candidate items:
                                - for "child": {id, title, description, sim, rel_depth, path_ids, workflows_preview}
                                - for "workflow": {id, title, description, sim}
        :param lookahead_json: Optional JSON with extra look-ahead hints (for "child" stage).
        :param conv_history: Optional conversation history for LLM context.
        :return: dict with keys:
                 - stage: Echo of input stage
                 - pick_id: ID of the chosen candidate
                 - beam: Ranked list of {id, score, reason}
        """
        fmt, parser = self.get_output_parser()
        out = self.get_response_and_parse(
            output_parser=parser, conv_history=conv_history, query=query, stage=stage,
            path_text=path_text, candidates_json=self.obj_to_json(candidates_json),
            lookahead_json=self.obj_to_json(lookahead_json) or "",
            traversal_selector_format_instructions=fmt) or {}
        return out
