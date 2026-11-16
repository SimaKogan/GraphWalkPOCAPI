from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions
from DataLayer.data_module import load_dict


class RouterAgent(PromptBasedAgent):
    def __init__(self, restrictions_path, llm_options: LLMOptions | None = None):
        super().__init__(name="RouterAgent", prompt_type="router_decomposer", llm_options=llm_options)
        self.restriction = load_dict(restrictions_path, encoding="utf-8")

    def decompose_and_route(self, query: str, conv_history=None):
        """Decompose a query into atomic sub-queries and route each.

        Uses the prompt `router_decomposer`.
        """
        router_format_instructions, router_output_parser = self.get_output_parser()
        resp = self.get_response_and_parse(
            output_parser=router_output_parser, query=query, restriction=self.restriction, conv_history=conv_history,
            router_format_instructions=router_format_instructions,
        )
        return resp
