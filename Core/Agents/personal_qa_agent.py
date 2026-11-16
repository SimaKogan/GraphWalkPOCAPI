from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions, format_facts_for_prompt


class PersonalQAAgent(PromptBasedAgent):
    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="PersonalQAAgent", prompt_type="personal_qa", llm_options=llm_options)

    def answer(self, query: str, state, workflow, facts,
               conv_history=None):
        answer_format_instructions, answer_output_parser = self.get_output_parser()
        if state:
            formatted_state = state.format_string()
        else:
            formatted_state = ""

        if workflow:
            formatted_workflow = workflow.format_string()
        else:
            formatted_workflow = ""
        if facts:
            formatted_response = format_facts_for_prompt(facts)
        else:
            formatted_response = ""
        response_dict = self.get_response_and_parse(
            output_parser=answer_output_parser, query=query, format_decimals=4, topic_state_data=formatted_state,
            formatted_workflow=formatted_workflow, formatted_response=formatted_response, conv_history=conv_history,
            answer_format_instructions=answer_format_instructions)
        return response_dict
