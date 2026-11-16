from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from Tools.session_tools import Answer


class GeneralManager:
    """General pipeline wrapper.

    The Personal POC orchestrator expects a "general" capability that returns:
      {"answer": <str>, "used_info": <list>}

    The integrated General pipeline (vendored under :mod:`General`) returns:
      {"answer": <str>, "citations": [...], "reasoning": <str>}

    We map `citations` -> `used_info` for compatibility.
    """

    async def async_ask(self, question: str) -> Dict[str, Any]:
        # Import lazily to keep startup light and avoid importing heavy deps
        # when "general" route isn't used.
        try:
            from General.api.main import aiTopics_classify_for_eval
        except ImportError as e:
            raise RuntimeError(
                "General pipeline package is not installed. "
                "Install it from git (pinned) to use LocalMode/general-service execution."
            ) from e

        out = await aiTopics_classify_for_eval(userQuery=question)
        return self._normalize_out(out)

    def ask(self, question: str, conv_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Synchronous wrapper.

        LocalMode adapters run this in a worker thread (see :func:`Utils.async_utils.to_thread`).
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_ask(question))
        raise RuntimeError("GeneralManager.ask() cannot run inside an active event loop; use await aask().")

    def handle(self, token: str, main_id: str, question: str, sessions) -> Answer:  # TODO: NOT used, perhaps remove
        """Legacy-style handler (best-effort).

        Not used by the orchestrator flow, but kept as a convenient hook for
        any older code paths that may call a manager directly.
        """
        # conv_hist = sessions.build_conv_history(token, main_id)
        out = self.ask(question)
        final_text = out.get("answer") or ""
        used_info = out.get("used_info")
        sessions.append_answer(token, main_id, final_text, used_info=used_info, source="general_pipeline")
        return Answer(text=final_text, token=token, facts=None, cursor=None, used_info=used_info)

    @staticmethod
    def _normalize_out(out: Any) -> Dict[str, Any]:
        """Normalize the General pipeline output to {answer, used_info}."""

        if isinstance(out, dict):
            answer = out.get("answer")
            citations = out.get("citations")
            used_info = out.get("used_info")
            return {
                "answer": answer,
                "used_info": used_info if used_info is not None else citations,
            }

        # Defensive fallback: the upstream code sometimes returns raw search results.
        return {
            "answer": "לא הצלחתי לאתר תשובה במאגר המידע עבור השאלה הזו.",
            "used_info": [],
        }
