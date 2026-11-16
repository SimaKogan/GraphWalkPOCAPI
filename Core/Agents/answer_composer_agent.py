from typing import Any, Dict, List

from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions
from DataLayer.data_module import load_dict


class AnswerComposerAgent(PromptBasedAgent):
    """Compose a final answer from multiple sub-answers."""

    def __init__(self, restrictions_path, llm_options: LLMOptions | None = None):
        super().__init__(name="AnswerComposerAgent", prompt_type="answer_composer", llm_options=llm_options)
        self.restriction = load_dict(restrictions_path, encoding="utf-8")

    def compose(self, query: str, sub_qa: List[Dict[str, Any]], conv_history=None):
        format_instructions, output_parser = self.get_output_parser()
        resp = self.get_response_and_parse(
            output_parser=output_parser, query=query, sub_qa_json=self.obj_to_json(sub_qa),
            answer_composer_format_instructions=format_instructions, restriction=self.restriction,
            conv_history=conv_history)
        return resp
