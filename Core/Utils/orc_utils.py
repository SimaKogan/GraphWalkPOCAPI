from __future__ import annotations

from typing import Any, List, Optional


def main_question(sessions, token: str, main_id: str) -> str:
    """Return the main question text for a given main.

    Supports both:
      - Low-level SessionStore (with .get(token) -> SessionMemory)
      - Higher-level sessions facades (LocalSessionsAdapter / SessionsClient) that expose build_bounded_conv_history().
    """

    # Case 1: SessionStore-style object with .get(token)
    get_attr = getattr(sessions, "get", None)
    if callable(get_attr):
        mem = get_attr(token)
        mains = mem.get("mains") or {}
        if main_id not in mains:
            raise KeyError(f"main_id {main_id!r} not found in session mains")
        return (mains[main_id] or {}).get("question") or ""

    # Case 2: Sessions facade (LocalSessionsAdapter, SessionsClient, etc.)
    if hasattr(sessions, "build_bounded_conv_history"):
        history = sessions.build_bounded_conv_history(token, main_id, n_last_mains=1)
        if not history:
            raise ValueError(f"No history found for token={token!r}, main_id={main_id!r}")

        # The first user message in that main is the main question
        for msg in history:
            if msg.get("role") == "user":
                return msg.get("text", "")

        # Fallback: take the first message text if roles are missing
        return history[0].get("text", "")

    raise TypeError(
        f"Unsupported sessions type {type(sessions)} for main_question: "
        "expected SessionStore-like object or sessions facade with build_bounded_conv_history()"
    )


def compose_topic_select_query(sessions, token: str, main_id: str) -> str:
    """Build query for WF selection critic/topic selection.

    Current semantics: main question + user follow-ups that were produced in response to wf_selection_critic clarifications.
    """
    mem = sessions.get(token)
    m = (mem.get("mains") or {})[main_id]
    parts = [m.get("question") or ""]
    for h in (m.get("history") or []):
        if (h.get("type") == "followup" and h.get("source") == "wf_selection_critic" and h.get("role") == "user"):
            parts.append(h.get("text") or "")
    return "\n".join([p for p in parts if (p or "").strip()])


def _extract_recent_router_clarifications(conv: List[dict[str, Any]], max_clarifications: int) -> List[str]:
    """Extract the latest clarification answers relevant to a router clarification step.

    We avoid relying on the follow-up's `source` (which may vary) and instead anchor
    on the last router prompt (assistant/system message with source='router').
    """

    last_router_prompt_idx: Optional[int] = None
    for i, turn in enumerate(conv):
        if (
                turn.get("role") == "assistant"
                and turn.get("type") == "system"
                and turn.get("source") == "router"
        ):
            last_router_prompt_idx = i

    if last_router_prompt_idx is not None:
        after = conv[last_router_prompt_idx + 1:]
        clarifications = [
            (t.get("text") or "").strip()
            for t in after
            if t.get("role") == "user" and t.get("text")
        ]
        clarifications = [c for c in clarifications if c]
        return clarifications[-max_clarifications:]

    # Fallback: old behavior (may fail if followups aren't tagged)
    clarifications = [
        (t.get("text") or "").strip()
        for t in conv
        if t.get("role") == "user" and t.get("source") == "router"
    ]
    clarifications = [c for c in clarifications if c]
    if clarifications:
        return clarifications[-max_clarifications:]

    # Last resort: treat any followups as clarifications (bounded to last N)
    clarifications = [
        (t.get("text") or "").strip()
        for t in conv
        if t.get("role") == "user" and t.get("type") == "followup"
    ]
    clarifications = [c for c in clarifications if c]
    return clarifications[-max_clarifications:]


def compose_router_query(sessions, token: str, main_id: str, max_clarifications: int = 3) -> str:
    """Build a router query composed of:

    - the original main question
    - the latest clarification answers from the user (bounded)

    The output is LLM-friendly and stable across storage/source variations.
    """

    base = main_question(sessions, token, main_id)
    conv = sessions.build_bounded_conv_history(token, main_id, n_last_mains=1, latest_query_text=None)
    clarifications = _extract_recent_router_clarifications(conv, max_clarifications=max_clarifications)

    if not clarifications:
        return base

    return base + "\n\nהבהרות מהמשתמש:\n" + "\n".join(f"- {c}" for c in clarifications)


async def compose_router_query_async(sessions_async, token: str, main_id: str, max_clarifications: int = 3) -> str:
    """Async variant of :func:`compose_router_query`.

    Intended for Orchestrator RemoteMode, where session access is over HTTP.
    """

    # Prefer explicit question endpoint when available
    base = ""
    get_q = getattr(sessions_async, "get_question_text", None)
    if callable(get_q):
        import inspect

        if inspect.iscoroutinefunction(get_q):
            base = await get_q(token, main_id)
        else:
            base = get_q(token, main_id)

    conv = await sessions_async.build_bounded_conv_history(token, main_id, n_last_mains=1, latest_query_text=None)

    if not base:
        for msg in conv:
            if msg.get("role") == "user":
                base = msg.get("text", "")
                break

    clarifications = _extract_recent_router_clarifications(conv, max_clarifications=max_clarifications)
    if not clarifications:
        return base

    return base + "\n\nהבהרות מהמשתמש:\n" + "\n".join(f"- {c}" for c in clarifications)
