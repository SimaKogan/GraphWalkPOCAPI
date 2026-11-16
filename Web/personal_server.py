from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Config.settings import settings
from Web.threadpool_config import configure_threadpool
from Tools.session_store import SessionStore
from Tools.session_tools import Answer, InputNeeded
from Core.Managers.personal_manager import PersonalManager

# Initialize settings & core objects
# exp_path = "Resources/SystemConfigs/exp_server_scenario_test_remote.json"
exp_path = "Resources/SystemConfigs/exp_server_1_remote.json"
settings.update(exp_path)

app = FastAPI(title="Personal Route + Sessions Service")
configure_threadpool(app)

store = SessionStore()
personal_manager = PersonalManager(embedder=None)


# --------- Pydantic models ---------


class CreateSessionIn(BaseModel):
    user_id: Optional[str] = None


class SessionMetaOut(BaseModel):
    token: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    current_main_id: Optional[str] = None
    current_sub_id: Optional[str] = None


class NewMainIn(BaseModel):
    question: str


class NewSubMainIn(BaseModel):
    question: str
    route: Optional[str] = None
    idx: Optional[int] = None
    intro: Optional[str] = None


class CurrentSubIn(BaseModel):
    sub_id: Optional[str] = None


class ParentMainOut(BaseModel):
    parent_main_id: str


class QuestionOut(BaseModel):
    question: str


class PendingInputOut(BaseModel):
    pending: Optional[Dict[str, Any]] = None


class PendingInputIn(BaseModel):
    pending: Dict[str, Any]
    log_prompt: bool = True


class FollowupIn(BaseModel):
    text: str
    source: str


class AnswerIn(BaseModel):
    text: str
    # Flex: List[str] (personal pipeline) OR List[{text,url,...}] (general pipeline)
    used_info: Optional[List[Any]] = None
    source: Optional[str] = None


class RouteIn(BaseModel):
    route: str


class FactsIn(BaseModel):
    facts: Dict[str, Any]


class CursorOut(BaseModel):
    cursor: Dict[str, Any]


class BoundedHistoryOut(BaseModel):
    history: List[Dict[str, Any]]


class PersonalHandleIn(BaseModel):
    token: str
    main_id: str


class PersonalHandleOut(BaseModel):
    text: str
    token: str
    used_info: Optional[List[Any]] = None
    facts: Optional[Dict[str, Any]] = None
    cursor: Optional[Dict[str, Any]] = None
    pending: bool = False
    need: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "personal"}


# --------- Session / conversation endpoints ---------


@app.post("/sessions", response_model=SessionMetaOut)
def create_session(payload: CreateSessionIn):
    token = store.new(user_id=payload.user_id)
    mem = store.get(token)
    return SessionMetaOut(
        token=token,
        user_id=mem.get("user_id"),
        conversation_id=mem.get("conversation_id"),
        current_main_id=mem.get("current_main_id"),
        current_sub_id=mem.get("current_sub_id"),
    )


@app.get("/sessions/{token}", response_model=SessionMetaOut)
def get_session_meta(token: str):
    try:
        mem = store.get(token)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionMetaOut(
        token=token,
        user_id=mem.get("user_id"),
        conversation_id=mem.get("conversation_id"),
        current_main_id=mem.get("current_main_id"),
        current_sub_id=mem.get("current_sub_id"),
    )


@app.post("/sessions/{token}/touch")
def touch_session(token: str):
    try:
        store.touch_updated_at(token)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.get("/users/{user_id}/conversations")
def list_conversations_for_user(user_id: str):
    return {"items": store.list_conversations_for_user(user_id)}


@app.get("/conversations/{token}/history")
def conversation_history(token: str):
    try:
        messages = store.get_conversation_history(token)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": messages}


# --------- Main-level endpoints ---------


@app.post("/sessions/{token}/mains")
def start_main(token: str, payload: NewMainIn):
    try:
        main_id = store.start_main(token, payload.question)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"main_id": main_id}


@app.get("/sessions/{token}/mains/current")
def get_current_main(token: str):
    try:
        main_id = store.get_current_main(token)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"main_id": main_id}


@app.get("/sessions/{token}/mains/{main_id}/parent", response_model=ParentMainOut)
def resolve_parent_main(token: str, main_id: str):
    try:
        mem = store.get(token)
        m = (mem.get("mains") or {}).get(main_id) or {}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    # If it's a submain, it will have parent_main_id set; if not, it resolves to itself
    parent = m.get("parent_main_id") or main_id
    return ParentMainOut(parent_main_id=parent)


@app.get("/sessions/{token}/mains/{main_id}/question", response_model=QuestionOut)
def get_question_text(token: str, main_id: str):
    try:
        mem = store.get(token)
        m = (mem.get("mains") or {}).get(main_id) or {}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    question = m.get("question") or ""
    return QuestionOut(question=question)


# --------- Submain (split) endpoints ---------


@app.get("/sessions/{token}/submains/current")
def get_current_sub(token: str):
    try:
        sub_id = store.get_current_sub(token)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"sub_id": sub_id}


@app.post("/sessions/{token}/submains/current")
def set_current_sub(token: str, payload: CurrentSubIn):
    try:
        store.set_current_sub(token, payload.sub_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.post("/sessions/{token}/mains/{parent_main_id}/submains")
def start_submain(token: str, parent_main_id: str, payload: NewSubMainIn):
    try:
        sub_id = store.start_submain(
            token=token,
            parent_main_id=parent_main_id,
            question_text=payload.question,
            route=payload.route,
            idx=payload.idx,
            intro=payload.intro,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"sub_id": sub_id}


@app.get("/sessions/{token}/mains/{parent_main_id}/submains")
def list_submains(token: str, parent_main_id: str):
    try:
        submains = store.list_submains(token, parent_main_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"submains": submains}


@app.get("/sessions/{token}/mains/{main_id}/pending_input", response_model=PendingInputOut)
def get_pending_input(token: str, main_id: str):
    try:
        pending = store.get_pending_input(token, main_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return PendingInputOut(pending=pending)


@app.post("/sessions/{token}/mains/{main_id}/pending_input")
def set_pending_input(token: str, main_id: str, payload: PendingInputIn):
    try:
        store.set_pending_input(token, main_id, payload.pending, payload.log_prompt)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"ok": True}


@app.delete("/sessions/{token}/mains/{main_id}/pending_input", response_model=PendingInputOut)
def pop_pending_input(token: str, main_id: str):
    try:
        pending = store.pop_pending_input(token, main_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return PendingInputOut(pending=pending)


@app.post("/sessions/{token}/mains/{main_id}/followup")
def append_followup(token: str, main_id: str, payload: FollowupIn):
    try:
        store.append_followup(token, main_id, payload.text, source=payload.source)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"ok": True}


@app.post("/sessions/{token}/mains/{main_id}/system")
def append_system(token: str, main_id: str, payload: FollowupIn):
    try:
        store.append_system(token, main_id, payload.text, source=payload.source)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"ok": True}


@app.post("/sessions/{token}/mains/{main_id}/answer")
def append_answer(token: str, main_id: str, payload: AnswerIn):
    try:
        store.append_answer(token, main_id, payload.text, used_info=payload.used_info, source=payload.source)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"ok": True}


@app.post("/sessions/{token}/mains/{main_id}/route")
def set_route(token: str, main_id: str, payload: RouteIn):
    try:
        store.set_route(token, main_id, payload.route)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"ok": True}


@app.get("/sessions/{token}/mains/{main_id}/facts")
def get_facts(token: str, main_id: str):
    try:
        facts = store.get_facts(token, main_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"facts": facts}


@app.put("/sessions/{token}/mains/{main_id}/facts")
def set_facts(token: str, main_id: str, payload: FactsIn):
    try:
        store.set_facts(token, main_id, payload.facts)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return {"ok": True}


@app.get("/sessions/{token}/mains/{main_id}/cursor", response_model=CursorOut)
def get_cursor(token: str, main_id: str):
    try:
        cursor = store.get_cursor(token, main_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return CursorOut(cursor=cursor)


@app.get("/sessions/{token}/bounded_history", response_model=BoundedHistoryOut)
def bounded_history(token: str, main_id: str, n_last_mains: int = 1, latest_query_text: Optional[str] = None):
    try:
        history = store.build_bounded_conv_history(token, main_id, n_last_mains=n_last_mains,
                                                   latest_query_text=latest_query_text)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    return BoundedHistoryOut(history=history)


# --------- Personal route endpoint ---------


@app.post("/personal/handle", response_model=PersonalHandleOut)
async def personal_handle(payload: PersonalHandleIn):
    token = payload.token
    main_id = payload.main_id
    try:
        ans: Answer = await personal_manager.handle_async(token, main_id, store)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session/main not found")
    except InputNeeded as need:
        need_dict = need.to_dict()
        # Orchestrator is responsible for persisting pending_input so it can
        # decorate / prefix the prompt_text (e.g., for decomposed sub-queries).
        return PersonalHandleOut(
            text=need.text,
            token=token,
            facts=store.get_facts(token, main_id),
            cursor=store.get_cursor(token, main_id),
            pending=True,
            need=need_dict,
        )

    return PersonalHandleOut(
        text=ans.text,
        token=ans.token,
        facts=ans.facts,
        cursor=ans.cursor,
        pending=ans.pending,
        need=ans.need,
        used_info=ans.used_info,
    )
