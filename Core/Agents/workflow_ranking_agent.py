from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions


class WorkflowRankingAgent(PromptBasedAgent):
    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="WorkflowRankingAgent", prompt_type="workflow_ranking", llm_options=llm_options)

    def rank(self, user_query: str, wf_candidates: list[dict], top_k: int, conv_history=None):
        fmt_ins, output_parser = self.get_output_parser()
        resp = self.get_response_and_parse(
            output_parser=output_parser, conv_history=conv_history, user_query=user_query, wf_candidates=wf_candidates,
            top_k=top_k, format_decimals=4, format_instructions=fmt_ins
        )
        return resp
