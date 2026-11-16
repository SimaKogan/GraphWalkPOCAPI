from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions


class UnifiedWorkflowSelectionAgent(PromptBasedAgent):
    """Single-call workflow selection.

    This agent replaces the TopicSelector's former two-step LLM pipeline
    (topic ranking + workflow ranking) with a single call that returns a
    sorted workflow beam.
    """

    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(
            name="UnifiedWorkflowSelectionAgent",
            prompt_type="unified_workflow_selection",
            llm_options=llm_options,
        )

    def rank(
        self,
        user_query: str,
        wf_candidates: list[dict],
        top_k: int,
        states_json: str | None = None,
        conv_history=None,
    ):
        fmt_ins, output_parser = self.get_output_parser()
        return self.get_response_and_parse(
            output_parser=output_parser,
            conv_history=conv_history,
            user_query=user_query,
            wf_candidates=wf_candidates,
            top_k=top_k,
            states_json=states_json or "",
            format_decimals=4,
            format_instructions=fmt_ins,
        ) or {}
