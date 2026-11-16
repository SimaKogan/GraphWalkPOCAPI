"""Local mock of the (fictional) Maslul transit API.

Serves the invented demo riders from ``Resources/MockApi/riders.json`` so that the personal
route can run end to end without any external service::

    python -m Tools.MockApi.server            # listens on http://localhost:9000/mock-api

Every endpoint of ``Resources/Specs/transit_endpoints.json`` is served as
``GET /mock-api/<EndpointName>?riderId=<id>&...``. The response envelope is the one that
``Tools/transit_tools.py`` expects: ``{"data": [ {field: {fieldNameEn, fieldNameHe, value, ...}} ]}``.

Authentication is a bearer token (``TRANSIT_API_TOKEN``, default ``dev-token``).
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request

ROOT = Path(__file__).resolve().parents[2]
RIDERS_PATH = Path(os.getenv("MOCK_RIDERS_PATH", ROOT / "Resources" / "MockApi" / "riders.json"))
ENDPOINTS_PATH = ROOT / "Resources" / "Specs" / "transit_endpoints.json"

app = FastAPI(title="Maslul mock transit API (fictional)")


def _load_riders() -> dict:
    return json.loads(RIDERS_PATH.read_text(encoding="utf-8"))["riders"]


def _load_endpoints() -> dict:
    return json.loads(ENDPOINTS_PATH.read_text(encoding="utf-8"))["endpoints"]


def _expected_token() -> str:
    return os.getenv("TRANSIT_API_TOKEN", "dev-token")


@app.get("/health")
def health():
    return {"status": "ok", "riders": len(_load_riders())}


@app.get("/mock-api/{endpoint}")
def call_endpoint(endpoint: str, request: Request, riderId: str = Query(...),
                  authorization: str = Header(default="")):
    if authorization != f"Bearer {_expected_token()}":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    if endpoint not in _load_endpoints():
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: {endpoint}")

    rider = _load_riders().get(riderId)
    if rider is None:
        raise HTTPException(status_code=404, detail=f"Unknown rider: {riderId}")

    body = rider.get(endpoint)
    if body is None:
        raise HTTPException(status_code=404, detail=f"No data for rider {riderId} on {endpoint}")
    return {"data": [body]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("MOCK_API_PORT", "9000")))
