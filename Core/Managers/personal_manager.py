from Core.Embeddings.embeddings import Embedder
from Core.Utils.llm_utils import LLMOptions
from Core.WorkflowSelection.SelectByQuestion.question_selector import QuestionSelector
from Core.WorkflowSelection.SelectByTopicClassification.topic_selector import TopicSelector
from Core.WorkflowSelection.SelectByUnifiedAgent.unified_selector import UnifiedWorkflowSelector
from Core.WorkflowSelection.selection_fusion import aggregate_candidates, format_candidates
from Tools.session_tools import Answer, InputNeeded
from Core.Agents.wf_selection_critic_agent import WorkflowSelectionCriticAgent
from Core.Agents.personal_qa_agent import PersonalQAAgent
from Core.WFExecution.executor import Executor
from Core.Utils.WorkflowsUtils.workflow_metadata import enrich_data_specs_with_metadata
from DataLayer.Entities.entities_manager import EntitiesManager
from DataLayer.data_module import load_dict
from Config.settings import settings
from Core.Utils.orc_utils import compose_topic_select_query
from Utils.async_utils import to_thread
from Resources.PromptConfigs.prompts import no_workflow_match_default_message


class PersonalManager:
    """
    Handles the 'personal' route
    """

    def __init__(self, embedder):

        # load all files
        self.data_specs = {"states": load_dict(settings.specs["states_path"], encoding="utf-8")["states"],
                           "blocks": load_dict(settings.specs["blocks_path"], encoding="utf-8")["blocks"],
                           "workflows": load_dict(settings.specs["workflows_path"], encoding="utf-8")["workflows"],
                           "actions": load_dict(settings.specs["actions_path"], encoding="utf-8")["actions"],
                           "endpoints": load_dict(settings.specs["endpoints_path"], encoding="utf-8")["endpoints"],
                           "personal_questions": settings.workflow_selection["personal_questions"],
                           "constants": settings.constants}
        # create executor
        self.executor = Executor(
            data_specs=self.data_specs,
            external_personal_info_services=dict(settings.services["external_personal_info_services"]),
            debug_active=settings.debug["active"],
            mock_tools=settings.debug["mock_tools"],
        )
        # Process Workflows: tools -> actions -> blocks -> workflows
        # and propagating descriptions to workflows from actions through blocks
        self.data_specs = enrich_data_specs_with_metadata(self.data_specs, self.executor.action_registry)

        # create graph db
        if embedder is None:
            embedder = Embedder(model_name=settings.general["embedding_model_name"])
        self.embedder = embedder
        self.entities_manager = EntitiesManager(embedder=self.embedder)

        # Prefer the new "entities_store" section; fall back to legacy "graph_db" if still present
        entities_store_cfg = getattr(settings, "entities_store", None)
        add_embeddings = entities_store_cfg.get("add_embeddings", True)
        root_id = entities_store_cfg.get("root_id", "root")

        self.entities_manager.init_entities_store(
            data_specs=self.data_specs,
            add_embeddings=add_embeddings,
            root_id=root_id,
        )

        # History
        self.conv_history_last_mains = settings.memory["conv_history"]["n_last_mains"]
        # Agents
        self.qa_agent = PersonalQAAgent(LLMOptions(model_name=settings.agents["personal_qa"]))

        # Workflow selection method:
        # - "topic": existing TopicSelector (may internally build a state_beam)
        # - "unified": new single-LLM selector that returns wf_beam only
        wf_sel_method = (settings.workflow_selection or {}).get("selection_method", "topic")

        if wf_sel_method == "unified":
            self.wf_selector = UnifiedWorkflowSelector(
                get_root=self.entities_manager.get_root,
                get_children=self.entities_manager.get_children,
                get_states=self.entities_manager.get_states,
                get_workflows_of_state=self.entities_manager.get_workflows_of_state,
                topics_top_k=settings.workflow_selection["selector_hyperparams"]["state_top_k"],
                wf_top_k=settings.workflow_selection["selector_hyperparams"]["wf_top_k"],
                states_representation=settings.workflow_selection["selector_hyperparams"]["states_representation"],
                cache_ttl_minutes=settings.workflow_selection["selector_hyperparams"]["cache_ttl_minutes"],
            )
        elif wf_sel_method == "topic":
            self.wf_selector = TopicSelector(
                get_root=self.entities_manager.get_root,
                get_children=self.entities_manager.get_children,
                get_states=self.entities_manager.get_states,
                get_workflows_of_state=self.entities_manager.get_workflows_of_state,
                topics_top_k=settings.workflow_selection["selector_hyperparams"]["state_top_k"],
                wf_top_k=settings.workflow_selection["selector_hyperparams"]["wf_top_k"],
                states_representation=settings.workflow_selection["selector_hyperparams"]["states_representation"],
                cache_ttl_minutes=settings.workflow_selection["selector_hyperparams"]["cache_ttl_minutes"],
            )
        else:
            raise ValueError(f"Unknown workflow selection method: {wf_sel_method}")

        self.critic = WorkflowSelectionCriticAgent(LLMOptions(model_name=settings.agents["wf_selection_critic"]))
        self.critic_max_clarifications = settings.workflow_selection["critic_max_clarifications"]

        self.question_selector = QuestionSelector(
            search_top_k_fn=self.entities_manager.search_questions_top_k,
            k=self.data_specs["personal_questions"]["top_k"],
            vector_index_name=self.data_specs["personal_questions"]["question_vector_index_name"]
        )

        # ---------------- public pipeline ----------------

    def handle(self, token: str, main_id: str, sessions) -> Answer:
        """
        Advance the personal pipeline. Raises InputNeeded when required.
        1. If workflow not yet chosen → run selection.
        2. If workflow chosen but not yet approved → run wf_selection_critic gate.
        3. If approved or already past wf_selection_critic → run executor deterministically.
        4. If executor completed → run personal QAAgent and answer.
        Each branch persists facts/cursor into session store.
        """
        facts = sessions.get_facts(token, main_id) or {}
        cursor = sessions.get_cursor(token, main_id) or {}
        # Phase 1: SelectByTopicClassification → candidate workflow (critic_workflow)
        if cursor.get("phase") in (None, "select_workflow"):
            facts["query_text"] = compose_topic_select_query(sessions, token, main_id)
            facts["query_embed"] = self.embedder.get_text_embeddings(facts["query_text"])

            cursor["phase"] = "select_workflow"
            cursor["phase_idx"] = 0
            wf_beam, topic_selector_cand_wf_id = self.wf_selector.select(
                user_query=facts.get("query_text"),
                user_query_embed=facts.get("query_embed"),
                conv_history=sessions.build_bounded_conv_history(
                    token, main_id, n_last_mains=self.conv_history_last_mains.get("wf_selection"),
                    latest_query_text=facts.get("query_text"))
            )
            wf_selection = {
                "wf_beam": wf_beam,
                "topic_selector_cand_wf_id": topic_selector_cand_wf_id
            }
            facts.update({"wf_selection": wf_selection})
            cursor["phase_idx"] = 1
            sessions.set_facts(token, main_id, facts)
            sessions.set_cursor(token, main_id, cursor)

            # Question-based Workflow selection
            # Retrieve top-k curated question hits (vector index search)
            retrieved_workflows, quest_cand_wf_id = self.question_selector.retrieve_and_summarize(
                facts.get("query_embed"), per_wf_limit=self.data_specs["personal_questions"]["per_workflow_limit"])
            facts.get("wf_selection").update({
                "quest_retrieved_workflows": retrieved_workflows,
                "quest_cand_wf_id": quest_cand_wf_id
            })
            cursor["phase_idx"] = 3

            # Aggregate candidates
            agg_cands = aggregate_candidates(
                topic_select_wf_beam=facts.get("wf_selection").get("wf_beam") or [],
                retrieved_wf_by_quests=retrieved_workflows or {},
                fetch_connected_questions=self.entities_manager.get_questions_for_workflow,
                get_workflow_fn=self.entities_manager.get_workflows,
                get_parent_state_fn=self.entities_manager.get_workflow_parent_state,
                query_embed=facts.get("query_embed"),  # same one used by vector search
                per_wf_limit=self.data_specs["personal_questions"]["per_workflow_limit"]
            )
            facts.get("wf_selection").update({"agg_cands": agg_cands})
            if not agg_cands:
                return self.get_no_match_answer(cursor, facts, None, main_id, sessions, token)
            cursor["phase"] = "critic_workflow"  # advance to critic
            cursor["phase_idx"] = 0

            sessions.set_facts(token, main_id, facts)
            sessions.set_cursor(token, main_id, cursor)

        # Phase 2: Critic → approve or clarify
        if cursor.get("phase") == "critic_workflow":
            facts.setdefault("critic_clarify_count", 0)
            # refresh question embeddings and text
            updated_query = compose_topic_select_query(sessions, token, main_id)
            if updated_query != facts["query_text"]:
                facts["query_text"] = updated_query
                facts["query_embed"] = self.embedder.get_text_embeddings(facts["query_text"])

            topic_selector_cand_wf_id = facts.get("wf_selection").get("topic_selector_cand_wf_id")
            quest_cand_wf_id = facts.get("wf_selection").get("quest_cand_wf_id")
            format_agg_cands = format_candidates(wfs_cand=facts.get("wf_selection").get("agg_cands"))

            judge = self.critic.judge(
                query=facts["query_text"],
                topic_selector_cand_wf_id=topic_selector_cand_wf_id,
                quest_cand_wf_id=quest_cand_wf_id,
                agg_cands=format_agg_cands,
                conv_history=sessions.build_bounded_conv_history(
                    token, main_id, n_last_mains=self.conv_history_last_mains["default"],
                    latest_query_text=facts["query_text"])
            )
            facts.get("wf_selection").update({"critic_reason": judge.get("Reasoning steps")})

            if (judge or {}).get("decision") == "accept":
                chosen_wf_id = ((judge or {}).get("accept_id") or topic_selector_cand_wf_id or quest_cand_wf_id)
                if not chosen_wf_id:
                    return self.get_no_match_answer(cursor, facts, judge, main_id, sessions, token)
                else:
                    cursor.update({
                        "workflow_id": chosen_wf_id,
                        "phase": "selected_workflow",
                        "phase_idx": 0
                    })
                    sessions.set_facts(token, main_id, facts)
                    sessions.set_cursor(token, main_id, cursor)
            elif (judge or {}).get("decision") == "clarify":  # clarify
                if facts["critic_clarify_count"] >= self.critic_max_clarifications - 1:
                    return self.get_no_match_answer(cursor, facts, judge, main_id, sessions, token,
                                                    no_match_text_prefix="לא ניתן להשלים את הבקשה, התהליך הגיעה למספר ההבהרות המותר. ")
                else:
                    facts["critic_clarify_count"] += 1
                    # # Decide where to resume after clarification -
                    # Reset phase so next round can generate a fresh candidate/beam after user clarification
                    retry_phase = (judge or {}).get("retry_phase")
                    if retry_phase == "select_workflow":
                        cursor.update({"phase": "select_workflow", "phase_idx": 0})
                    elif retry_phase == "critic_workflow":
                        cursor.update({"phase": "critic_workflow", "phase_idx": 0})
                    else:  # default it to re-try all
                        cursor.update({"phase": "select_workflow", "phase_idx": 0})

                    sessions.set_facts(token, main_id, facts)
                    sessions.set_cursor(token, main_id, cursor)
                    raise InputNeeded(
                        prompt_text=judge.get("clarify_question") or "צריך הבהרה",
                        return_keys=["clarification"],
                        return_types=["string"],
                        source="wf_selection_critic",
                    )
            elif (judge or {}).get("decision") == "no_match":
                return self.get_no_match_answer(cursor, facts, judge, main_id, sessions, token)
        # Phase 3: Deterministic executor
        if cursor.get("phase") in ("selected_workflow", "run_block_action"):
            try:
                wf_result = self.executor.run_workflow(get_workflows=self.entities_manager.get_workflows, facts=facts,
                                                       cursor=cursor)
                facts.update(wf_result.get("facts", {}))
                cursor.update(wf_result.get("cursor", {}))
                sessions.set_facts(token, main_id, facts)
                sessions.set_cursor(token, main_id, cursor)
            except InputNeeded as exec_need:
                sessions.set_facts(token, main_id, facts)
                sessions.set_cursor(token, main_id, cursor)
                raise InputNeeded(prompt_text=exec_need.text, return_keys=exec_need.return_keys,
                                  return_types=exec_need.return_types, source="executor")

            # Phase 4: QAAgent answer
            wf_obj = self.entities_manager.get_workflows(cursor.get("workflow_id"))[0]
            state_obj = self.entities_manager.get_workflow_parent_state(cursor.get("workflow_id"))[0]
            out = self.qa_agent.answer(
                query=facts["query_text"],
                state=state_obj,
                workflow=wf_obj,
                facts=facts,
                conv_history=sessions.build_bounded_conv_history(
                    token, main_id, n_last_mains=self.conv_history_last_mains["default"],
                    latest_query_text=facts["query_text"]),
            )
            # final_text = append_used_info_section(out["answer"], out.get("used_info"))  # TODO: Maybe remove this addition not necessary if the model receives it togather
            final_text = out["answer"]
            sessions.append_answer(token, main_id, final_text, used_info=out.get("used_info"), source="personal_qa")
            cursor["phase"] = "answered"
            cursor["phase_idx"] = 0
            sessions.set_cursor(token, main_id, cursor)
            return Answer(text=final_text, token=token, facts=facts, cursor=cursor, used_info=out.get("used_info"))

        # Fallback
        return Answer(text="ממשיך בעיבוד…", token=token, facts=sessions.get_facts(token, main_id),
                      cursor=sessions.get_cursor(token, main_id))

    async def handle_async(self, token: str, main_id: str, sessions) -> Answer:
        """Async wrapper around :meth:`handle`.

        We avoid duplicating the PersonalManager state-machine logic by executing
        the existing synchronous implementation in a worker thread.

        External transit API HTTP tool calls are async-first (httpx). When invoked
        from the synchronous workflow runner, awaitables are bridged via
        :func:`Utils.async_utils.run_coro_sync`.
        """
        return await to_thread(self.handle, token, main_id, sessions)

    def get_no_match_answer(self, cursor, facts, judge, main_id, sessions, token, no_match_text_prefix=""):
        no_match_text = no_workflow_match_default_message
        no_match_text = no_match_text_prefix + no_match_text
        facts.get("wf_selection").update({
            "no_match": True,
            "no_match_reason": (judge or {}).get("no_match_reason", no_match_text)
        })
        sessions.set_facts(token, main_id, facts)
        sessions.append_answer(token, main_id, no_match_text, used_info=None, source="personal_qa")

        cursor.update({
            "workflow_id": None,
            "phase": "answered",
            "phase_idx": 0
        })
        sessions.set_cursor(token, main_id, cursor)
        return Answer(text=no_match_text, token=token, facts=facts, cursor=cursor)
