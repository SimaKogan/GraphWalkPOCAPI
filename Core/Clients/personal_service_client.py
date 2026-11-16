from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class AsyncPersonalServiceClient:
    """Async HTTP client for personal route pipeline (/personal/handle).

    RemoteMode is async-only at the orchestrator boundary.

    Implementation details:
    - Reuses a single httpx.AsyncClient across calls (connection pooling / keep-alive).
    - Supports explicit lifecycle via async context manager or aclose().
    - Allows dependency injection of an externally-managed AsyncClient.

    If an external client is injected, this class will not close it.
    """

    def __init__(self, base_url: str, timeout: float = 60.0, client: Optional[httpx.AsyncClient] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = client
        # If a client is injected, this class must not close it.
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncPersonalServiceClient":
        self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def handle_personal(self, token: str, main_id: str) -> Dict[str, Any]:
        client = self._get_client()
        r = await client.post("/personal/handle", json={"token": token, "main_id": main_id})
        r.raise_for_status()
        return r.json()


class AsyncSessionsClient:
    """Async HTTP client facade over SessionStore hosted in Personal service.

    Tightened implementation:
    - Reuses a single httpx.AsyncClient across calls (connection pooling / keep-alive).
    - Supports explicit lifecycle via async context manager or aclose().
    - Allows dependency injection of an externally-managed AsyncClient.

    If an external client is injected, this class will not close it.
    """

    def __init__(self, base_url: str, timeout: float = 10.0, client: Optional[httpx.AsyncClient] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = client
        # If a client is injected, this class must not close it.
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Use base_url to enable relative path requests and connection reuse.
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the internally-managed client (no-op for injected clients)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncSessionsClient":
        self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _request_json(self, method: str, path: str, *, timeout: Optional[float] = None, **kwargs):
        client = self._get_client()
        # Do not pass timeout=None, as that disables httpx timeouts and overrides the client default.
        req_kwargs = dict(kwargs)
        if timeout is not None:
            req_kwargs["timeout"] = timeout
        r = await client.request(method, path, **req_kwargs)
        r.raise_for_status()
        return r.json()

    async def create_conversation(self, user_id: Optional[str]) -> str:
        data = await self._request_json("POST", "/sessions", json={"user_id": user_id})
        return data["token"]

    async def list_conversations_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        data = await self._request_json("GET", f"/users/{user_id}/conversations")
        return data["items"]

    async def get_conversation_history(self, token: str) -> List[Dict[str, Any]]:
        data = await self._request_json("GET", f"/conversations/{token}/history")
        return data["messages"]

    async def get_session_meta(self, token: str) -> Dict[str, Any]:
        return await self._request_json("GET", f"/sessions/{token}")

    async def touch(self, token: str) -> None:
        await self._request_json("POST", f"/sessions/{token}/touch", timeout=5.0)

    async def start_main(self, token: str, question: str) -> str:
        data = await self._request_json("POST", f"/sessions/{token}/mains", json={"question": question})
        return data["main_id"]

    async def get_current_main(self, token: str) -> Optional[str]:
        data = await self._request_json("GET", f"/sessions/{token}/mains/current")
        return data.get("main_id")

    async def get_current_sub(self, token: str) -> Optional[str]:
        data = await self._request_json("GET", f"/sessions/{token}/submains/current")
        return data.get("sub_id")

    async def set_current_sub(self, token: str, sub_id: Optional[str]) -> None:
        await self._request_json("POST", f"/sessions/{token}/submains/current", json={"sub_id": sub_id})

    async def start_submain(self, token: str, parent_main_id: str, question: str, route: Optional[str] = None,
                            idx: Optional[int] = None, intro: Optional[str] = None) -> str:
        data = await self._request_json(
            "POST",
            f"/sessions/{token}/mains/{parent_main_id}/submains",
            json={"question": question, "route": route, "idx": idx, "intro": intro},
        )
        return data["sub_id"]

    async def list_submains(self, token: str, parent_main_id: str) -> List[Dict[str, Any]]:
        data = await self._request_json("GET", f"/sessions/{token}/mains/{parent_main_id}/submains")
        return data.get("submains") or []

    async def resolve_parent_main_id(self, token: str, main_id: str) -> str:
        data = await self._request_json("GET", f"/sessions/{token}/mains/{main_id}/parent")
        return data.get("parent_main_id") or main_id

    async def get_question_text(self, token: str, main_id: str) -> str:
        data = await self._request_json("GET", f"/sessions/{token}/mains/{main_id}/question")
        return data.get("question") or ""

    async def get_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        data = await self._request_json("GET", f"/sessions/{token}/mains/{main_id}/pending_input")
        return data.get("pending")

    async def pop_pending_input(self, token: str, main_id: str) -> Optional[Dict[str, Any]]:
        data = await self._request_json("DELETE", f"/sessions/{token}/mains/{main_id}/pending_input")
        return data.get("pending")

    async def set_pending_input(self, token: str, main_id: str, pending: Dict[str, Any],
                                log_prompt: bool = True) -> None:
        await self._request_json(
            "POST",
            f"/sessions/{token}/mains/{main_id}/pending_input",
            json={"pending": pending, "log_prompt": log_prompt},
        )

    async def append_followup(self, token: str, main_id: str, text: str, source: str) -> None:
        await self._request_json(
            "POST",
            f"/sessions/{token}/mains/{main_id}/followup",
            json={"text": text, "source": source},
        )

    async def append_system(self, token: str, main_id: str, text: str, source: str) -> None:
        await self._request_json(
            "POST",
            f"/sessions/{token}/mains/{main_id}/system",
            json={"text": text, "source": source},
        )

    async def append_answer(self, token: str, main_id: str, text: str, used_info: Optional[List[str]] = None,
                            source: Optional[str] = None) -> None:
        await self._request_json(
            "POST",
            f"/sessions/{token}/mains/{main_id}/answer",
            json={"text": text, "used_info": used_info, "source": source},
        )

    async def set_route(self, token: str, main_id: str, route: str) -> None:
        await self._request_json("POST", f"/sessions/{token}/mains/{main_id}/route", json={"route": route})

    async def get_facts(self, token: str, main_id: str) -> Dict[str, Any]:
        data = await self._request_json("GET", f"/sessions/{token}/mains/{main_id}/facts")
        return data["facts"]

    async def set_facts(self, token: str, main_id: str, facts: Dict[str, Any]):
        await self._request_json("PUT", f"/sessions/{token}/mains/{main_id}/facts", json={"facts": facts})

    async def get_cursor(self, token: str, main_id: str) -> Dict[str, Any]:
        data = await self._request_json("GET", f"/sessions/{token}/mains/{main_id}/cursor")
        return data["cursor"]

    async def build_bounded_conv_history(self, token: str, main_id: str, n_last_mains: int,
                                         latest_query_text: Optional[str] = None):
        data = await self._request_json(
            "GET",
            f"/sessions/{token}/bounded_history",
            params={
                "main_id": main_id,
                "n_last_mains": n_last_mains,
                "latest_query_text": latest_query_text,
            },
        )
        return data["history"]
