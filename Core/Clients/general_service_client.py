from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class AsyncGeneralServiceClient:
    """Async HTTP client for the General route service.

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

    async def __aenter__(self) -> "AsyncGeneralServiceClient":
        self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def ask(self, question: str, conv_history: List[Dict[str, Any]]) -> \
            Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question": question,
            "conv_history": conv_history,
        }
        client = self._get_client()
        r = await client.post("/general/ask", json=payload)  # Note: uses /general/ask
        r.raise_for_status()
        data = r.json()
        return {
            "answer": data.get("answer"),
            # Backwards/forwards compatibility:
            # - Our wrapper service returns `used_info`.
            # - The underlying General pipeline returns `citations`.
            "used_info": data.get("used_info") or data.get("citations"),
        }
