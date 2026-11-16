from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions, format_custom_response


class GeneralQAAgent(PromptBasedAgent):
    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="GeneralQAAgent", prompt_type="general_qa", llm_options=llm_options)

    def answer(self, query: str, conv_history=None):
        answer_format_instructions, answer_output_parser = self.get_output_parser()

        response_dict = self.get_response_and_parse(
            output_parser=answer_output_parser, query=query,
            conv_history=conv_history, answer_format_instructions=answer_format_instructions)
        return response_dict
