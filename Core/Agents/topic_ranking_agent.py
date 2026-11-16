from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions


class TopicRankingAgent(PromptBasedAgent):
    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="TopicRankingAgent", prompt_type="topic_state_ranking", llm_options=llm_options)

    def rank(self, user_query: str, states_json: str, top_k: int, conv_history=None):
        fmt_ins, output_parser = self.get_output_parser()
        resp = self.get_response_and_parse(
            output_parser=output_parser, conv_history=conv_history, user_query=user_query,
            terminal_states_json=states_json, top_k=top_k, format_instructions=fmt_ins
        )
        return resp.get("ranked_states", []) or []
