import asyncio
from typing import Any, Dict, List, Optional

from Config.settings import settings
from Core.Agents.answer_composer_agent import AnswerComposerAgent
from Core.Agents.pending_input_processor_agent import PendingInputProcessorAgent
from Core.Agents.route_answer_agent import RouteAnswerAgent
from Core.Agents.router_agent import RouterAgent
from Core.Clients.general_service_client import AsyncGeneralServiceClient
from Core.Clients.personal_service_client import AsyncPersonalServiceClient, AsyncSessionsClient
from Core.Utils.orc_utils import compose_router_query_async
from Tools.session_tools import Answer, InputNeeded
from Utils.async_utils import to_thread


class Orchestrator:
    """Central orchestrator for routing and conversation management.

    Target model (Option A):
    - LocalMode remains in-process via adapters.
    - RemoteMode is async-only at the orchestrator boundary (FastAPI endpoints use async).

    Behavior summary:
    - Router performs decomposition + routing (personal/general) per sub-query.
    - Personal sub-queries run sequentially.
    - General sub-queries run in parallel after all personal sub-queries complete.
    - When a personal sub-query first raises InputNeeded, we prefix a short split-status message.
    - A final AnswerComposerAgent composes the final answer for the parent main.
    """

    def __init__(self, mode: str = "local", personal_base_url: Optional[str] = None,
                 general_base_url: Optional[str] = None):
        self.mode = mode

        self.route_answer_agent = RouteAnswerAgent()
        self.router = RouterAgent(restrictions_path=settings.general["restrictions_path"])
        self.answer_composer = AnswerComposerAgent(restrictions_path=settings.general["restrictions_path"])
        self.pending_input_processor_agent = PendingInputProcessorAgent()
        self.conv_history_last_mains = settings.memory["conv_history"]["n_last_mains"]

        if mode == "local":
            from Tools.session_store import SessionStore
            from Core.Clients.local_adapters import LocalSessionsAdapter, LocalPersonalAdapter, LocalGeneralAdapter
            from Core.Clients.async_local_adapters import (
                AsyncLocalSessionsAdapter,
                AsyncLocalPersonalAdapter,
                AsyncLocalGeneralAdapter,
            )

            store = SessionStore()
            self._local_store = store
            self.sessions_async = AsyncLocalSessionsAdapter(LocalSessionsAdapter(store))
            self.personal_client_async = AsyncLocalPersonalAdapter(LocalPersonalAdapter(store))
            self.general_client_async = AsyncLocalGeneralAdapter(LocalGeneralAdapter())

        elif mode == "remote":
            if not personal_base_url or not general_base_url:
                raise ValueError("In remote mode, personal_base_url and general_base_url must be provided.")

            # RemoteMode is async-only at the boundary.
            self.sessions_async = AsyncSessionsClient(base_url=personal_base_url)
            self.personal_client_async = AsyncPersonalServiceClient(base_url=personal_base_url)
            self.general_client_async = AsyncGeneralServiceClient(base_url=general_base_url)

        else:
            raise ValueError("mode must be 'local' or 'remote'")

    async def aclose(self) -> None:
        """Close any internally-managed async HTTP clients."""
        for obj in (getattr(self, "sessions_async", None), getattr(self, "personal_client_async", None),
                    getattr(self, "general_client_async", None)):
            aclose = getattr(obj, "aclose", None)
            if callable(aclose):
                await aclose()

    # ================= Conversations API (async) =================

    async def create_conversation(self, user_id: Optional[str]) -> str:
        return await self.sessions_async.create_conversation(user_id=user_id)

    async def list_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        return await self.sessions_async.list_conversations_for_user(user_id)

    async def get_conversation_history(self, token: str) -> List[Dict[str, Any]]:
        return await self.sessions_async.get_conversation_history(token)

    # ================= Main entrypoint =================

    async def ask_async(self, user_id: str, token: str, text: str) -> Answer:
        """Unified async entrypoint (LocalMode + RemoteMode).

        - If there's a pending InputNeeded for the current sub/main → continue flow.
        - Else → start a new main (decompose+route).
        """

        meta = await self.sessions_async.get_session_meta(token)
        if meta.get("user_id") and meta.get("user_id") != user_id:
            raise Exception("Forbidden: token does not belong to user")

        # Continue pending input (sub first, then main)
        active_id = meta.get("current_sub_id") or meta.get("current_main_id")
        if active_id:
            pending = await self.sessions_async.pop_pending_input(token, active_id)
            if pending:
                ans = await self._handle_query_continue(token, active_id, pending, text)
                await self.sessions_async.touch(token)
                return ans

        # No pending input → new main
        ans = await self._handle_query_start(token, text)
        await self.sessions_async.touch(token)
        return ans

    def ask(self, user_id: str, token: str, text: str) -> Answer:
        """LocalMode-only synchronous wrapper.

        RemoteMode is async-only. Use :meth:`ask_async`.
        """

        if self.mode != "local":
            raise RuntimeError("RemoteMode is async-only; use await Orchestrator.ask_async(...)")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ask_async(user_id, token, text))

        raise RuntimeError("ask() cannot be called while an event loop is running; use await ask_async().")

    # ================= Internal helpers =================

    async def _handle_query_start(self, token: str, question: str) -> Answer:
        parent_main_id = await self.sessions_async.start_main(token, question)
        return await self._handle_decomposition_block(token, parent_main_id, question)

    async def _handle_query_continue(self, token: str, active_main_id: str, pending: Dict[str, Any],
                                     text: str) -> Answer:
        """Resume after InputNeeded for either parent main or a submain."""

        processed = await to_thread(
            self.pending_input_processor_agent.process,
            pending=pending,
            user_response=text,
            conv_history=await self.sessions_async.build_bounded_conv_history(
                token, active_main_id, n_last_mains=self.conv_history_last_mains["default"], latest_query_text=text),
        )
        source = pending.get("source") or "executor"
        decision = (processed.get("decision") or "").strip().lower()

        if decision == "answer":
            await self.sessions_async.append_followup(token, active_main_id, text, source="user")
            facts = await self.sessions_async.get_facts(token, active_main_id)
            facts.update(processed.get("normalized_value") or {})
            await self.sessions_async.set_facts(token, active_main_id, facts)

            # Router clarification → re-run decomposition
            if source == "router":
                parent_main_id = await self.sessions_async.resolve_parent_main_id(token, active_main_id)
                query = await compose_router_query_async(self.sessions_async, token, parent_main_id)
                return await self._handle_decomposition_block(token, parent_main_id, query)

            # Personal pipeline continuation
            if source in ("wf_selection_critic", "executor"):
                parent = await self.sessions_async.resolve_parent_main_id(token, active_main_id)
                return await self._run_decomposed(token, parent)

            raise Exception(f"Unknown source of input: {source}")

        if decision == "halt":
            halt_msg = processed.get("halt_message", "") or "הבנתי, עצרתי. איך אוכל לעזור?"
            await self.sessions_async.append_followup(token, active_main_id, text, source="user")
            await self.sessions_async.append_system(token, active_main_id, halt_msg, source="pending_input_processor")
            return Answer(text=halt_msg, token=token, facts=await self.sessions_async.get_facts(token, active_main_id),
                          cursor=await self.sessions_async.get_cursor(token, active_main_id), pending=False)

        if decision == "new_main":
            await self.sessions_async.append_followup(token, active_main_id, text, source="user")
            return await self._handle_query_start(token, processed.get("new_main_question", "") or text)

        if decision == "retry":
            await self.sessions_async.set_pending_input(token, active_main_id, pending, log_prompt=False)
            retry_prefix = processed.get("retry_prefix", "") or "רק כדי שאוכל להמשיך — "
            retry_text = f"{retry_prefix}\n{pending.get('prompt_text', '')}"
            await self.sessions_async.append_followup(token, active_main_id, text, source="user")
            await self.sessions_async.append_system(token, active_main_id, retry_text, source="pending_input_processor")
            return Answer(text=retry_text, token=token,
                          facts=await self.sessions_async.get_facts(token, active_main_id),
                          cursor=await self.sessions_async.get_cursor(token, active_main_id), pending=True,
                          need=pending)

        # Safety net
        await self.sessions_async.set_pending_input(token, active_main_id, pending, log_prompt=False)
        return Answer(text=pending.get("prompt_text", ""), token=token,
                      facts=await self.sessions_async.get_facts(token, active_main_id),
                      cursor=await self.sessions_async.get_cursor(token, active_main_id), pending=True, need=pending)

    async def _handle_decomposition_block(self, token: str, parent_main_id: str, question: str) -> Answer:
        """Router-Decomposer block."""

        conv_history = await self.sessions_async.build_bounded_conv_history(
            token,
            parent_main_id,
            n_last_mains=self.conv_history_last_mains["router"],
            latest_query_text=question,
        )
        out = await to_thread(self.router.decompose_and_route, query=question, conv_history=conv_history)
        decision = (out or {}).get("decision", "accept")
        mode = (out or {}).get("mode", "decomposed")

        # Clarification
        if decision == "clarify":
            clarify_q = out.get("clarification_prompt") or "אפשר לחדד את הבקשה?"
            need = InputNeeded(prompt_text=clarify_q, return_keys=["clarification"], return_types=["string"],
                               source="router")
            await self.sessions_async.set_pending_input(token, parent_main_id, need.to_dict(), log_prompt=True)
            return Answer(text=need.text, token=token, facts=await self.sessions_async.get_facts(token, parent_main_id),
                          cursor=await self.sessions_async.get_cursor(token, parent_main_id), pending=True,
                          need=need.to_dict())

        # Blocked / Casual (whole query)
        if mode in ("blocked", "casual"):
            ans = await to_thread(
                self.route_answer_agent.answer,
                query=question,
                route=mode,
                reasoning=out.get("blocked_reason", ""),
            )
            await self.sessions_async.append_answer(token, parent_main_id, ans["answer"], source="route_answer_agent")
            return Answer(text=ans["answer"], token=token,
                          facts=await self.sessions_async.get_facts(token, parent_main_id),
                          cursor=await self.sessions_async.get_cursor(token, parent_main_id))

        # Decomposed
        sub_queries = (out or {}).get("sub_queries") or []
        if not isinstance(sub_queries, list) or not sub_queries:
            ans = await to_thread(self.route_answer_agent.answer, query=question, route="casual",
                                  reasoning=out.get("blocked_reason", ""))
            await self.sessions_async.append_answer(token, parent_main_id, ans["answer"], source="route_answer_agent")
            return Answer(text=ans["answer"], token=token,
                          facts=await self.sessions_async.get_facts(token, parent_main_id),
                          cursor=await self.sessions_async.get_cursor(token, parent_main_id))

        sub_ids: List[str] = []
        personal_ids: List[str] = []
        general_ids: List[str] = []
        for i, sub_query in enumerate(sub_queries, start=1):
            sub_query_text = (sub_query or {}).get("query") or ""
            route = (sub_query or {}).get("route") or "general"
            intro = (sub_query or {}).get("intro") or None
            idx = (sub_query or {}).get("idx") or i
            sub_id = await self.sessions_async.start_submain(
                token=token, parent_main_id=parent_main_id, question=sub_query_text, route=route, idx=idx, intro=intro
            )
            sub_ids.append(sub_id)
            if route == "personal":
                personal_ids.append(sub_id)
            else:
                general_ids.append(sub_id)

        facts = await self.sessions_async.get_facts(token, parent_main_id)
        facts["decomposition"] = {
            "sub_ids": sub_ids,
            "personal_ids": personal_ids,
            "general_ids": general_ids,
            "n": len(sub_ids),
            "next_personal": 0,
            "stage": "personal" if personal_ids else "general",
            "prefixed": {},
        }
        await self.sessions_async.set_facts(token, parent_main_id, facts)
        return await self._run_decomposed(token, parent_main_id)

    async def _run_decomposed(self, token: str, parent_main_id: str) -> Answer:
        """Run decomposed pipeline for a parent main."""

        await self.sessions_async.set_current_sub(token, None)

        parent_facts = await self.sessions_async.get_facts(token, parent_main_id)
        decomp = parent_facts.get("decomposition") or {}

        personal_ids: List[str] = decomp.get("personal_ids") or []
        general_ids: List[str] = decomp.get("general_ids") or []
        n_total = int(decomp.get("n") or (len(personal_ids) + len(general_ids)))
        stage = decomp.get("stage") or ("personal" if personal_ids else "general")
        next_personal = int(decomp.get("next_personal") or 0)

        if stage == "personal":
            while next_personal < len(personal_ids):
                sub_id = personal_ids[next_personal]
                resp = await self.personal_client_async.handle_personal(token=token, main_id=sub_id)
                ans = Answer(text=resp["text"], token=resp.get("token") or token, facts=resp.get("facts"),
                             cursor=resp.get("cursor"), pending=resp.get("pending", False), need=resp.get("need"))

                if ans.pending:
                    need = ans.need or {}
                    prefixed_prompt, _ = await self._maybe_prefix_breakdown(
                        token=token, parent_main_id=parent_main_id, sub_id=sub_id, n_total=n_total,
                        need_prompt=need.get("prompt_text") or ans.text)
                    await self.sessions_async.set_pending_input(token, sub_id, need, log_prompt=True)
                    await self.sessions_async.set_current_sub(token, sub_id)
                    return Answer(text=prefixed_prompt, token=token,
                                  facts=await self.sessions_async.get_facts(token, sub_id),
                                  cursor=await self.sessions_async.get_cursor(token, sub_id), pending=True, need=need)

                next_personal += 1
                decomp["next_personal"] = next_personal
                parent_facts["decomposition"] = decomp
                await self.sessions_async.set_facts(token, parent_main_id, parent_facts)

            decomp["stage"] = "general"
            parent_facts["decomposition"] = decomp
            await self.sessions_async.set_facts(token, parent_main_id, parent_facts)
            stage = "general"

        if stage == "general":
            if general_ids:
                max_workers = int(settings.general.get("max_workers", 5) or 5)
                sem = asyncio.Semaphore(max(1, min(max_workers, len(general_ids))))

                async def _run_one(sid: str) -> Dict[str, Any]:
                    async with sem:
                        q = await self.sessions_async.get_question_text(token, sid)
                        return await self._ask_general(token, sid, q)

                tasks = [asyncio.create_task(_run_one(sid)) for sid in general_ids]
                # Harden fan-out:
                # - One failing sub-call should not fail the whole request.
                # - Preserve per-sid recording even when some calls error.
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for sid, out in zip(general_ids, results):
                    if isinstance(out, asyncio.CancelledError):
                        # Respect cancellation promptly (e.g. client disconnect).
                        raise out
                    if isinstance(out, Exception):
                        # Record a per-sub failure; keep parent flow alive.
                        # Avoid leaking internal exception details to the user.
                        await self.sessions_async.append_answer(
                            token, sid, "משהו השתבש בעת ביצוע החלק הזה. נסו שוב מאוחר יותר.",
                            source="general_service_error", )
                        continue

                    answer_text = (out.get("answer") or "")
                    used_info = out.get("used_info")
                    await self.sessions_async.append_answer(token, sid, answer_text, used_info=used_info)

            # Compose final answer
            submains = await self.sessions_async.list_submains(token, parent_main_id)
            sub_qa: list[dict[str, Any]] = []
            all_used_info: list[Any] = []
            for sub_main in sorted(submains, key=lambda x: (x.get("idx") or 0)):
                used = sub_main.get("used_info") or []
                all_used_info.extend(list(used))
                sub_qa.append(
                    {
                        "idx": sub_main.get("idx"),
                        "route": sub_main.get("route"),
                        "query": sub_main.get("question"),
                        "answer": sub_main.get("answer") or "",
                        "used_info": used,
                    }
                )

            composed = await to_thread(
                self.answer_composer.compose,
                query=await self.sessions_async.get_question_text(token, parent_main_id),
                sub_qa=sub_qa,
                conv_history=await self.sessions_async.build_bounded_conv_history(
                    token, parent_main_id, n_last_mains=settings.memory["conv_history"]["n_last_mains"]["default"],
                    latest_query_text=None),
            )

            if (composed or {}).get("is_restricted"):
                final_answer = (composed or {}).get("restricted_message") or "לא ניתן להשיב על כך."
                await self.sessions_async.append_answer(token, parent_main_id, final_answer, source="answer_composer")
            else:
                final_answer = (composed or {}).get("final_answer") or ""
                await self.sessions_async.append_answer(token, parent_main_id, final_answer, used_info=all_used_info,
                                                        source="answer_composer")

            parent_facts = await self.sessions_async.get_facts(token, parent_main_id)
            decomp = (parent_facts.get("decomposition") or {})
            decomp["stage"] = "done"
            parent_facts["decomposition"] = decomp
            await self.sessions_async.set_facts(token, parent_main_id, parent_facts)

            return Answer(text=final_answer, token=token,
                          facts=await self.sessions_async.get_facts(token, parent_main_id),
                          cursor=await self.sessions_async.get_cursor(token, parent_main_id), pending=False, )

        return Answer(text="משהו השתבש. נסו שוב.", token=token,
                      facts=await self.sessions_async.get_facts(token, parent_main_id),
                      cursor=await self.sessions_async.get_cursor(token, parent_main_id), pending=False, )

    async def _ask_general(self, token: str, sub_id: str, question: str) -> Dict[str, Any]:
        conv_history = await self.sessions_async.build_bounded_conv_history(
            token, sub_id, n_last_mains=self.conv_history_last_mains["default"], latest_query_text=question)
        return await self.general_client_async.ask(
            question=question,
            conv_history=conv_history,
        )

    async def _maybe_prefix_breakdown(self, token: str, parent_main_id: str, sub_id: str, n_total: int,
                                      need_prompt: str, ) -> tuple[str, bool]:
        """Prefix a split explanation only once per personal sub-query."""

        parent_facts = await self.sessions_async.get_facts(token, parent_main_id)
        decomp = (parent_facts.get("decomposition") or {})
        prefixed = (decomp.get("prefixed") or {})
        if prefixed.get(sub_id):
            return need_prompt, False

        submains = await self.sessions_async.list_submains(token, parent_main_id)
        meta = next((x for x in submains if x.get("id") == sub_id), {})

        intro = (meta.get("intro") or "").strip()
        if intro:
            prefix = f"{intro}\n"
        else:
            idx = meta.get("idx")
            q = (meta.get("question") or "").strip()
            q_short = (q[:80] + "…") if len(q) > 80 else q
            prefix = f"הבקשה פוצלה ל-{n_total} חלקים. מטפל כעת בחלק {idx}/{n_total}: \"{q_short}\"\n"

        prefixed[sub_id] = True
        decomp["prefixed"] = prefixed
        parent_facts["decomposition"] = decomp
        await self.sessions_async.set_facts(token, parent_main_id, parent_facts)

        return f"{prefix}{need_prompt}", True
