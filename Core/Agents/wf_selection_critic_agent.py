from Core.Agents.prompt_based_agent import PromptBasedAgent
from Core.Utils.llm_utils import LLMOptions
from Resources.PromptConfigs.prompts import no_workflow_match_default_message


class WorkflowSelectionCriticAgent(PromptBasedAgent):
    def __init__(self, llm_options: LLMOptions | None = None):
        super().__init__(name="WorkflowSelectionCriticAgent", prompt_type="wf_selection_critic",
                         llm_options=llm_options)

    def judge(self, query: str, topic_selector_cand_wf_id, quest_cand_wf_id, agg_cands, conv_history=None):
        wf_select_critic_format_ins, wf_select_critic_output_parser = self.get_output_parser()

        resp = self.get_response_and_parse(
            output_parser=wf_select_critic_output_parser,
            conv_history=conv_history,
            format_decimals=4,
            query=query,
            topic_selector_cand_wf_id=topic_selector_cand_wf_id,
            quest_cand_wf_id=quest_cand_wf_id,
            no_workflow_match_default_message=no_workflow_match_default_message,
            formatted_aggregated_candidates=self.obj_to_json(agg_cands),
            wf_select_critic_format_instructions=wf_select_critic_format_ins
        )
        return resp
