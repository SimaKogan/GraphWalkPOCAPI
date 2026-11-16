# Server-level threadpool configuration.
#
# Centralizes concurrency policy at the service boundary:
# - AnyIO worker-thread limiter (sync endpoints / AnyIO offloads)
# - asyncio default executor (asyncio.to_thread)
#
# Tuned via Settings (env-driven): SERVER_THREADPOOL_WORKERS.

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import anyio
from fastapi import FastAPI

from Config.settings import settings


def configure_threadpool(app: FastAPI, *, thread_name_prefix: str = "svc-tp") -> None:
    @app.on_event("startup")
    async def _startup_threadpool() -> None:
        workers = settings.server_threadpool_workers
        anyio.to_thread.current_default_thread_limiter().total_tokens = workers

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix)
        loop.set_default_executor(executor)

        app.state.threadpool_executor = executor
        app.state.threadpool_workers = workers

    @app.on_event("shutdown")
    async def _shutdown_threadpool() -> None:
        executor: Optional[ThreadPoolExecutor] = getattr(app.state, "threadpool_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            app.state.threadpool_executor = None
