from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import bcrypt
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from Config.settings import settings
from Core.routing_orchestrator import Orchestrator
from Tools.session_tools import Answer
from Web.threadpool_config import configure_threadpool

# exp_path = "Resources/SystemConfigs/exp_server_1_local.json"
exp_path = "Resources/SystemConfigs/exp_server_1_remote.json"
# exp_path = "Resources/SystemConfigs/exp_server_scenario_test_local.json"
# exp_path = "Resources/SystemConfigs/exp_server_scenario_test_remote.json"
settings.update(exp_path)

app = FastAPI()
configure_threadpool(app)
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

USERS = {}  # email -> {"id": email, "pw_hash": bytes}

# Orchestrator wiring
mode = settings.general["orc_mode"] or "local"
if mode == "remote":
    personal_base_url = settings.personal_base_url
    general_base_url = settings.general_base_url
    orc = Orchestrator(mode="remote", personal_base_url=personal_base_url, general_base_url=general_base_url)
else:
    # Default: LocalMode, using in-process adapters (good for tests / dev).
    orc = Orchestrator(mode="local")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await orc.aclose()


# --------- Pydantic models ---------


class AuthIn(BaseModel):
    email: str
    password: str


class AuthOut(BaseModel):
    ok: bool
    user_id: str
    resumed: Optional[str] = None
    resume_token: Optional[str] = None


class ConversationItem(BaseModel):
    token: str
    title: str
    updated_at: str  # or datetime if you want


class ConversationsListOut(BaseModel):
    items: List[ConversationItem]


class RegisterOut(BaseModel):
    ok: bool
    user_id: str


class NewConversationOut(BaseModel):
    token: str


class ConversationHistoryOut(BaseModel):
    token: str
    messages: List[Dict[str, Any]]


class LogoutOut(BaseModel):
    ok: bool


class SendIn(BaseModel):
    token: str
    text: str


class SendOut(BaseModel):
    token: str
    text: str
    pending: bool
    need: Optional[Dict[str, Any]] = None


class MeOut(BaseModel):
    user_id: str


def require_user(request: Request) -> str:
    uid = request.session.get("user_id")
    if not uid or uid not in USERS:
        raise HTTPException(401, "Not authenticated")
    return uid


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return RedirectResponse("/chat" if request.session.get("user_id") else "/login")


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return FileResponse(str(TEMPLATES / "login.html"))


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse("/login")
    return FileResponse(str(TEMPLATES / "chat.html"))


# ---------- Auth API ----------


@app.post("/api/register", response_model=RegisterOut)
def api_register(payload: AuthIn, request: Request):
    email = payload.email.strip().lower()
    if not email or not payload.password:
        raise HTTPException(400, "Email and password required")
    if email in USERS:
        raise HTTPException(409, "User exists")
    pw_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt())
    USERS[email] = {"id": email, "pw_hash": pw_hash}
    request.session["user_id"] = email
    return {"ok": True, "user_id": email}


@app.post("/api/login", response_model=AuthOut)
async def api_login(payload: AuthIn, request: Request):
    email = payload.email.strip().lower()
    user = USERS.get(email)
    if not user or not bcrypt.checkpw(payload.password.encode("utf-8"), user["pw_hash"]):
        raise HTTPException(401, "Invalid credentials")

    request.session["user_id"] = email

    # optional resume
    resume_token = request.query_params.get("resume_token")
    resumed: Optional[Answer] = None

    if resume_token:
        try:
            meta = await orc.sessions_async.get_session_meta(resume_token)
            active_id = meta.get("current_sub_id") or meta.get("current_main_id")
            pending = await orc.sessions_async.get_pending_input(resume_token, active_id) if active_id else None
            if pending and "user_id" in (pending.get("return_keys") or []):
                resumed = await orc.ask_async(email, resume_token, email)
        except Exception:
            # best-effort resume
            resumed = None

    return {
        "ok": True,
        "user_id": email,
        "resumed": resumed.text if resumed else None,
        "resume_token": resume_token if resumed else None,
    }


@app.post("/api/logout", response_model=LogoutOut)
def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me", response_model=MeOut)
def api_me(user_id: str = Depends(require_user)):
    return {"user_id": user_id}


# ---------- Conversations (token-only) ----------
@app.get("/api/conversations", response_model=ConversationsListOut)
async def api_conversations(user_id: str = Depends(require_user)):
    return {"items": await orc.list_conversations(user_id)}


@app.post("/api/conversations/new", response_model=NewConversationOut)
async def api_conversations_new(user_id: str = Depends(require_user)):
    token = await orc.create_conversation(user_id)
    return {"token": token}


@app.get("/api/conversations/{token}/history", response_model=ConversationHistoryOut)
async def api_conversation_history(token: str, user_id: str = Depends(require_user)):
    # Basic ownership check via session meta
    meta = await orc.sessions_async.get_session_meta(token)
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Forbidden")
    hist = await orc.get_conversation_history(token)
    return {"token": token, "messages": hist}


# ---------- Chat ----------


@app.post("/api/send", response_model=SendOut)
async def api_send(payload: SendIn, user_id: str = Depends(require_user)):
    ans: Answer = await orc.ask_async(user_id, payload.token, payload.text)
    return {
        "token": payload.token,
        "text": ans.text,
        "pending": bool(ans.pending),
        "need": ans.need,
    }
