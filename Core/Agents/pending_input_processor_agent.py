import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, List

from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions


# DecisionKind = Literal["answer", "halt", "new_main", "retry"]
#
#
# @dataclass
# class PendingInputDecision:
#     """LLM output contract for a pending input response."""
#     kind: DecisionKind
#     normalized_inputs: Dict[str, Any]
#     assistant_message: str = ""
#     new_main_question: str = ""
#     retry_kind: Optional[str] = None  # engaged_not_answering / partial_unclear / out_of_scope
#     retry_prefix: Optional[str] = None
#     debug: Optional[Dict[str, Any]] = None


class PendingInputProcessorAgent(PromptBasedAgent):
    """
    Handles user replies while an InputNeeded prompt is active.
    Delegates classification and normalization to an LLM prompt.
    """

    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="PendingInputProcessor", prompt_type="pending_input_processor", llm_options=llm_options)

    def process(self, pending, user_response, conv_history=None):
        format_instructions, output_parser = self.get_output_parser()
        resp = self.get_response_and_parse(
            output_parser=output_parser, pending_prompt_text=pending.get("prompt_text") or "",
            pending_return_keys=json.dumps(pending.get("return_keys") or [], ensure_ascii=False),
            pending_return_types=json.dumps(pending.get("return_types") or [], ensure_ascii=False),
            pending_source=pending.get("source") or "executor", user_response=user_response,
            format_instructions=format_instructions, conv_history=conv_history
        )

        return resp
