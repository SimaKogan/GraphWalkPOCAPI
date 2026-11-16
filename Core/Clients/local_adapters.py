from typing import Any, Dict, List, Optional

from Tools.session_store import SessionStore
from Tools.session_tools import Answer, InputNeeded
from Core.Managers.personal_manager import PersonalManager
from Core.Managers.general_manager import GeneralManager


class LocalSessionsAdapter:
    """In-process facade over SessionStore.

    Exposes the minimal API that Orchestrator uses, mirroring SessionsClient.
    """

    def __init__(self, store: Optional[SessionStore] = None) -> None:
        self.store = store or SessionStore()

    # ---- conversation-level ----
    def create_conversation(self, user_id: Optional[str]) -> str:
        return self.store.new(user_id=user_id)

    def list_conversations_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        return self.store.list_conversations_for_user(user_id)

    def get_conversation_history(self, token: str) -> List[Dict[str, Any]]:
        return self.store.get_conversation_history(token)

    def get_session_meta(self, token: str) -> Dict[str, Any]:
        mem = self.store.get(token)
        return {
            "user_id": mem.get("user_id"),
            "conversation_id": mem.get("conversation_id"),
            "current_main_id": mem.get("current_main_id"),
            "current_sub_id": mem.get("current_sub_id"),
        }

    def touch(self, token: str) -> None:
        self.store.touch_updated_at(token)

    # ---- per-main ----
    def start_main(self, token: str, question: str) -> str:
        return self.store.start_main(token, question)

    def get_current_main(self, token: str) -> Optional[str]:
        return self.store.get_current_main(token)

    def get_current_sub(self, token: str) -> Optional[str]:
        return self.store.get_current_sub(token)

    def set_current_sub(self, token: str, sub_id: Optional[str]):
        self.store.set_current_sub(token, sub_id)

    def start_submain(self, token: str, parent_main_id: str, question: str,
                      route: Optional[str] = None, idx: Optional[int] = None,
                      intro: Optional[str] = None) -> str:
        return self.store.start_submain(token, parent_main_id, question, route=route, idx=idx, intro=intro)

    def list_submains(self, token: str, parent_main_id: str) -> List[Dict[str, Any]]:
        return self.store.list_submains(token, parent_main_id)

    def resolve_parent_main_id(self, token: str, main_id: str) -> str:
        return self.store.resolve_parent_main_id(token, main_id)

    def get_question_text(self, token: str, main_id: str) -> str:
        return self.store.get_question_text(token, main_id)

    def get_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_pending_input(token, main_id)

    def pop_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        return self.store.pop_pending_input(token, main_id)

    def set_pending_input(self, token: str, main_id: str, pending: Dict[str, Any], log_prompt: bool) -> None:
        self.store.set_pending_input(token, main_id, pending, log_prompt)

    def append_followup(self, token: str, main_id: str, text: str, source) -> None:
        self.store.append_followup(token, main_id, text, source=source)

    def append_system(self, token: str, main_id: str, text: str, source) -> None:
        self.store.append_system(token, main_id, text, source=source)

    def append_answer(self, token: str, main_id: str, text: str, used_info: Optional[List[str]] = None, source: Optional[str] = None) -> None:
        self.store.append_answer(token, main_id, text, used_info=used_info, source=source)

    def set_route(self, token: str, main_id: str, route: str) -> None:
        self.store.set_route(token, main_id, route)

    def get_facts(self, token: str, main_id: str) -> Dict[str, Any]:
        return self.store.get_facts(token, main_id)

    def set_facts(self, token: str, main_id: str, facts: Dict[str, Any]) -> None:
        self.store.set_facts(token, main_id, facts)

    def get_cursor(self, token: str, main_id: str) -> Dict[str, Any]:
        return self.store.get_cursor(token, main_id)

    def build_bounded_conv_history(self, token: str, main_id: str, n_last_mains: int,
                                   latest_query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.build_bounded_conv_history(
            token, main_id, n_last_mains=n_last_mains, latest_query_text=latest_query_text
        )


class LocalPersonalAdapter:
    """In-process adapter for PersonalManager.

    Exposes handle_personal(token, main_id) returning a dict shaped like the
    HTTP /personal/handle response.
    """

    def __init__(self, store: SessionStore, personal_manager=None):
        self.store = store
        self.personal_manager = personal_manager or PersonalManager(embedder=None)

    def handle_personal(self, token: str, main_id: str) -> Dict[str, Any]:
        try:
            ans: Answer = self.personal_manager.handle(token, main_id, self.store)
            return {
                "text": ans.text,
                "token": ans.token,
                "facts": ans.facts,
                "cursor": ans.cursor,
                "pending": ans.pending,
                "need": ans.need,
            }
        except InputNeeded as need:
            need_dict = need.to_dict()
            # Orchestrator is responsible for storing pending_input, so it can
            # decorate / prefix the prompt_text (e.g., for decomposed sub-queries).
            return {
                "text": need.text,
                "token": token,
                "facts": self.store.get_facts(token, main_id),
                "cursor": self.store.get_cursor(token, main_id),
                "pending": True,
                "need": need_dict,
            }


class LocalGeneralAdapter:
    """In-process adapter for the integrated General pipeline.

    Exposes ask(question, conv_history) -> {answer, used_info}, matching the
    HTTP :class:`Core.Clients.general_service_client.AsyncGeneralServiceClient` contract.
    """

    def __init__(self, manager: Optional[GeneralManager] = None) -> None:
        self.manager = manager or GeneralManager()

    def ask(self, question: str, conv_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # conv_history is currently unused by the General pipeline's /ask function.
        # We keep it in the signature for parity with RemoteMode.
        out = self.manager.ask(question=question, conv_history=conv_history)
        return {
            "answer": out.get("answer"),
            "used_info": out.get("used_info") or out.get("citations"),
        }
