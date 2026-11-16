from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions


class RouteAnswerAgent(PromptBasedAgent):
    """
    Agent that generates responses for blocked or casual routes.
    It takes the user query, the chosen route, and the router's reasoning
    and produces a short, courteous Hebrew answer.
    """

    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="RouteAnswerAgent", prompt_type="route_answer", llm_options=llm_options)

    def answer(self, query: str, route: str, reasoning: str, conv_history=None):
        answer_format_instructions, answer_output_parser = self.get_output_parser()
        response_dict = self.get_response_and_parse(
            output_parser=answer_output_parser, query=query, route=route, reasoning=reasoning,
            conv_history=conv_history, answer_format_instructions=answer_format_instructions,
        )
        return response_dict
