from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Config.settings import settings
from Core.Managers.general_manager import GeneralManager
from Utils.async_utils import to_thread
from Web.threadpool_config import configure_threadpool

# Initialize settings & QA agent
# exp_path = "Resources/SystemConfigs/exp_server_scenario_test_remote.json"
exp_path = "Resources/SystemConfigs/exp_server_1_remote.json"
settings.update(exp_path)

app = FastAPI(title="General Route Service")
configure_threadpool(app)

general_manager = GeneralManager()


class GeneralAskIn(BaseModel):
    question: str
    conv_history: List[Dict[str, Any]]


class GeneralAskOut(BaseModel):
    answer: str
    # General pipeline returns a list of citations: {"text": ..., "url": ...}
    used_info: Optional[List[Dict[str, Any]]] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "general"}


@app.get("/ask")
async def ask(userQuery: str):
    """Upstream-compatible endpoint.

    Mirrors the General pipeline interface: GET /ask?userQuery=...
    Returns {answer, citations, reasoning}.
    """

    try:
        out = await to_thread(general_manager.ask, question=userQuery, conv_history=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    citations = out.get("used_info") or out.get("citations") or []
    return {
        "answer": out.get("answer") or "",
        "citations": citations,
        "reasoning": "",
    }


@app.post("/general/ask", response_model=GeneralAskOut)
async def general_ask(payload: GeneralAskIn):
    try:
        # Run the integrated pipeline in a worker thread.
        # The pipeline uses blocking DB / LLM clients.
        out = await to_thread(general_manager.ask, question=payload.question, conv_history=payload.conv_history)
    except Exception as e:  # defensive
        raise HTTPException(status_code=500, detail=str(e))

    return GeneralAskOut(
        answer=out.get("answer") or "",
        used_info=out.get("used_info") or out.get("citations"),
    )
