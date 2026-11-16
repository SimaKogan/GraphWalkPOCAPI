from __future__ import annotations

from typing import Any, Dict, List, Optional

from Utils.async_utils import to_thread

from Core.Clients.local_adapters import LocalSessionsAdapter, LocalPersonalAdapter, LocalGeneralAdapter


class AsyncLocalSessionsAdapter:
    """Async facade over LocalSessionsAdapter.

    LocalMode runs in-process and is inherently synchronous. We expose an async interface
    so *Orchestrator can be async-only* while LocalMode remains functional.

    All methods are lightweight (in-memory) and run on the event loop.
    """

    def __init__(self, sessions: LocalSessionsAdapter) -> None:
        self._s = sessions

    async def aclose(self) -> None:
        return None

    async def create_conversation(self, user_id: Optional[str]) -> str:
        return self._s.create_conversation(user_id)

    async def list_conversations_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        return self._s.list_conversations_for_user(user_id)

    async def get_conversation_history(self, token: str) -> List[Dict[str, Any]]:
        return self._s.get_conversation_history(token)

    async def get_session_meta(self, token: str) -> Dict[str, Any]:
        return self._s.get_session_meta(token)

    async def touch(self, token: str) -> None:
        self._s.touch(token)

    async def start_main(self, token: str, question: str) -> str:
        return self._s.start_main(token, question)

    async def get_current_main(self, token: str) -> Optional[str]:
        return self._s.get_current_main(token)

    async def get_current_sub(self, token: str) -> Optional[str]:
        return self._s.get_current_sub(token)

    async def set_current_sub(self, token: str, sub_id: Optional[str]) -> None:
        self._s.set_current_sub(token, sub_id)

    async def start_submain(self, token: str, parent_main_id: str, question: str, route: Optional[str] = None,
                            idx: Optional[int] = None, intro: Optional[str] = None) -> str:
        return self._s.start_submain(token, parent_main_id, question, route=route, idx=idx, intro=intro)

    async def list_submains(self, token: str, parent_main_id: str) -> List[Dict[str, Any]]:
        return self._s.list_submains(token, parent_main_id)

    async def resolve_parent_main_id(self, token: str, main_id: str) -> str:
        return self._s.resolve_parent_main_id(token, main_id)

    async def get_question_text(self, token: str, main_id: str) -> str:
        return self._s.get_question_text(token, main_id)

    async def get_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        return self._s.get_pending_input(token, main_id)

    async def pop_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        return self._s.pop_pending_input(token, main_id)

    async def set_pending_input(self, token: str, main_id: str, pending: Dict[str, Any],
                                log_prompt: bool = True) -> None:
        self._s.set_pending_input(token, main_id, pending, log_prompt)

    async def append_followup(self, token: str, main_id: str, text: str, source: str) -> None:
        self._s.append_followup(token, main_id, text, source=source)

    async def append_system(self, token: str, main_id: str, text: str, source: str) -> None:
        self._s.append_system(token, main_id, text, source=source)

    async def append_answer(self, token: str, main_id: str, text: str, used_info: Optional[List[str]] = None,
                            source: Optional[str] = None) -> None:
        self._s.append_answer(token, main_id, text, used_info=used_info, source=source)

    async def set_route(self, token: str, main_id: str, route: str) -> None:
        self._s.set_route(token, main_id, route)

    async def get_facts(self, token: str, main_id: str) -> Dict[str, Any]:
        return self._s.get_facts(token, main_id)

    async def set_facts(self, token: str, main_id: str, facts: Dict[str, Any]) -> None:
        self._s.set_facts(token, main_id, facts)

    async def get_cursor(self, token: str, main_id: str) -> Dict[str, Any]:
        return self._s.get_cursor(token, main_id)

    async def build_bounded_conv_history(self, token: str, main_id: str, n_last_mains: int,
                                         latest_query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._s.build_bounded_conv_history(token, main_id, n_last_mains=n_last_mains,
                                                  latest_query_text=latest_query_text)


class AsyncLocalPersonalAdapter:
    """Async facade over LocalPersonalAdapter.

    PersonalManager execution may call LLMs/tools and should not run on the FastAPI event loop.
    """

    def __init__(self, personal: LocalPersonalAdapter) -> None:
        self._p = personal

    async def aclose(self) -> None:
        return None

    async def handle_personal(self, token: str, main_id: str) -> Dict[str, Any]:
        return await to_thread(self._p.handle_personal, token, main_id)


class AsyncLocalGeneralAdapter:
    """Async facade over LocalGeneralAdapter.

    General QA calls the LLM stack and should not block the event loop.
    """

    def __init__(self, general: LocalGeneralAdapter) -> None:
        self._g = general

    async def aclose(self) -> None:
        return None

    async def ask(self, question: str, conv_history: List[Dict[str, Any]]) -> \
            Dict[str, Any]:
        return await to_thread(self._g.ask, question, conv_history)
