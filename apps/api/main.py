"""B22: the HTTP API and the investigation screen.

Run with: uv run uvicorn apps.api.main:app --reload, then open http://127.0.0.1:8000

POST /api/investigations              start a replay of a demo case or a live run
GET  /api/investigations              list investigations
GET  /api/investigations/{id}         status, backend and case
GET  /api/investigations/{id}/events  server-sent events: started, step..., finished
GET  /api/demo-cases                  saved investigations that can be replayed
With PUBLIC_DEMO=true only replays are allowed.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from apps.api import runners
from apps.api.store import InvestigationStore
from core.config import get_settings
from core.connectors.storage import get_storage

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Next Experiment")
store = InvestigationStore(get_storage())


class StartRequest(BaseModel):
    backend: Literal["replay", "local", "tf"] = "replay"
    demo: str = "fault6-tf"
    delay_s: float = 1.5  # pause between replayed steps


@app.get("/")
def screen() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/demo-cases")
def demo_cases() -> list[dict]:
    return [
        {"name": name, "title": d.get("title", name), "note": d.get("note", "")}
        for name, d in runners.demo_cases().items()
    ]


@app.post("/api/investigations")
def start(req: StartRequest) -> dict:
    if req.backend != "replay" and get_settings().public_demo:
        raise HTTPException(403, "the public demo only replays saved investigations")
    if req.backend == "replay":
        cases = runners.demo_cases()
        if req.demo not in cases:
            raise HTTPException(404, f"no demo case {req.demo!r}")
        inv_id = store.create(
            "replay", cases[req.demo]["case"], cases[req.demo]["title"]
        )
        target, args = runners.replay, (store, inv_id, req.demo, req.delay_s)
    else:
        inv_id = store.create(req.backend, {}, f"Fault 6, live ({req.backend})")
        target, args = runners.live, (store, inv_id, req.backend)
    threading.Thread(target=target, args=args, daemon=True).start()
    return {"id": inv_id}


@app.get("/api/investigations")
def list_investigations() -> list[dict]:
    return [
        {k: m[k] for k in ("id", "status", "backend", "title", "created_at")}
        for m in store.list()
    ]


@app.get("/api/investigations/{inv_id}")
def get_investigation(inv_id: str) -> dict:
    try:
        return store.get(inv_id)
    except KeyError:
        raise HTTPException(404, f"no investigation {inv_id}") from None


@app.get("/api/investigations/{inv_id}/events")
async def events(inv_id: str, request: Request) -> EventSourceResponse:
    get_investigation(inv_id)

    async def stream():
        seq = 0
        while not await request.is_disconnected():
            for event in store.events_since(inv_id, seq):
                seq = event["seq"] + 1
                yield {"event": event["type"], "data": json.dumps(event["data"])}
            if store.is_done(inv_id) and not store.events_since(inv_id, seq):
                break
            await asyncio.sleep(0.3)

    return EventSourceResponse(stream())
