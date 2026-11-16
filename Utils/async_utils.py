from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar


T = TypeVar("T")


async def to_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function in a worker thread."""
    return await asyncio.to_thread(fn, *args, **kwargs)

def run_coro_sync(coro: Awaitable[T]) -> T:
    """Execute an awaitable from a synchronous context.

    This is intended for LocalMode / synchronous pipelines.
    If you call this while an event loop is already running in the current thread,
    it will raise a RuntimeError with an actionable message.
    """
    # If there is no running loop in this thread, we can safely run the coroutine.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A loop is running in this thread. Do not block it.
    raise RuntimeError(
        "run_coro_sync() was called from a thread with a running event loop. "
        "Use 'await ...' (preferred) or move the call into a worker thread."
    )
