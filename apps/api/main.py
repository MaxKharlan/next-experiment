"""B22: the HTTP API and the investigation screen.

Run with: uv run uvicorn apps.api.main:app --reload, then open http://127.0.0.1:8000

POST /api/investigations              start a replay of a demo case or a live run
GET  /api/investigations              list investigations
GET  /api/investigations/{id}         status, backend and case
GET  /api/investigations/{id}/events  server-sent events: started, step..., finished
GET  /api/investigations/{id}/report  persisted JSON report; ?download=true
GET  /api/demo-cases                  saved investigations that can be replayed
POST /api/investigations/{id}/retrain approve or decline the proposed full retrain
With PUBLIC_DEMO=true only replays are allowed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from apps.api import runners
from apps.api.case_input import CaseStart, prepare_case
from apps.api.label_audits import LabelAuditRequest, prepare_audit
from apps.api.store import get_investigation_store
from core.config import get_settings
from core.report.investigation import build_report

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Next Experiment")
store = get_investigation_store()


class StartRequest(BaseModel):
    backend: Literal["replay", "local", "tf"] = "replay"
    demo: str = "fault6-tf"
    delay_s: float = runners.REPLAY_DELAY_S  # pause between replayed steps


class Answer(BaseModel):
    approve: bool


@app.get("/")
def screen() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/capabilities")
def capabilities() -> dict:
    return {"explicit_cases": not get_settings().public_demo}


@app.post("/api/investigations/label-audit", status_code=201)
def label_audit(req: LabelAuditRequest) -> dict:
    if get_settings().public_demo:
        raise HTTPException(403, "the public demo only replays saved investigations")
    try:
        step = prepare_audit(req)
    except ValueError as err:
        raise HTTPException(422, str(err)) from None
    # Persist the digest and bounded counts, not the supplied token sequences.
    case = {
        "mode": "B",
        "symptom": "Caller-supplied label inspection",
        "job": {"model": "local labels", "hyperparameters": {}},
    }
    inv_id = store.create("local", case, req.title)
    store.emit(inv_id, "started", {"case": case, "backend": "local"})
    store.emit(inv_id, "step", step)
    store.emit(
        inv_id,
        "finished",
        {
            "status": "inconclusive",
            "audit_only": True,
            "llm_cost_usd": 0.0,
            "estimated_total_cost_usd": 0.0,
        },
    )
    return {"id": inv_id, "audit": step, "training_started": False}


@app.get("/api/demo-cases")
def demo_cases() -> list[dict]:
    return runners.demo_index()


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


def _prepare_explicit_case(req: CaseStart) -> dict:
    if get_settings().public_demo:
        raise HTTPException(403, "the public demo only replays saved investigations")
    try:
        return prepare_case(req)
    except (OSError, ValueError, TypeError) as err:
        raise HTTPException(422, str(err)) from None


@app.post("/api/investigations/validate")
def validate_case(req: CaseStart) -> dict:
    prepared = _prepare_explicit_case(req)
    return {
        "valid": True,
        "protocol": prepared["protocol"],
        "provider_job_verified": False,
        "training_started": False,
    }


@app.post("/api/investigations/from-case", status_code=202)
def start_from_case(
    req: CaseStart,
    idempotency_key: str | None = Header(default=None),
) -> dict:
    if get_settings().public_demo:
        raise HTTPException(403, "the public demo only replays saved investigations")
    identity = None
    if idempotency_key is not None:
        if not 1 <= len(idempotency_key) <= 128 or not all(
            33 <= ord(char) <= 126 for char in idempotency_key
        ):
            raise HTTPException(
                422, "Idempotency-Key must contain 1-128 visible ASCII characters"
            )
        identity = (
            hashlib.sha256(idempotency_key.encode()).hexdigest(),
            hashlib.sha256(
                json.dumps(
                    req.model_dump(), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )
        try:
            existing = store.find_request(*identity)
        except ValueError as err:
            raise HTTPException(409, str(err)) from None
        if existing is not None:
            return {"id": existing, "reused": True}
    prepared = _prepare_explicit_case(req)
    case = req.case.model_dump()
    title = f"{req.case.task}: {req.case.job.id}"
    if identity is None:
        inv_id, created = store.create("tf", case, title), True
    else:
        try:
            inv_id, created = store.create_once("tf", case, title, *identity)
        except ValueError as err:
            raise HTTPException(409, str(err)) from None
    if created:
        threading.Thread(
            target=runners.live,
            args=(store, inv_id, "tf"),
            kwargs={"prepared": prepared},
            daemon=True,
        ).start()
    return {"id": inv_id, "reused": not created}


@app.post("/api/investigations/{inv_id}/retrain")
def answer_retrain(inv_id: str, answer: Answer) -> dict:
    """The user's answer to the full retrain, the only level-3 step."""
    if get_settings().public_demo:
        raise HTTPException(403, "the public demo does not train models")
    get_investigation(inv_id)
    try:
        waiting = store.resume(inv_id, answer.approve)
    except ValueError as err:
        raise HTTPException(409, str(err)) from None
    threading.Thread(
        target=runners.retrain,
        args=(store, inv_id, waiting, answer.approve),
        daemon=True,
    ).start()
    return {"id": inv_id, "approved": answer.approve}


@app.get("/api/investigations")
def list_investigations() -> list[dict]:
    return [
        {k: m[k] for k in ("id", "status", "backend", "title", "created_at")}
        for m in store.list()
        if not get_settings().public_demo or m["backend"] == "replay"
    ]


@app.get("/api/investigations/{inv_id}")
def get_investigation(inv_id: str) -> dict:
    try:
        meta = store.get(inv_id)
    except KeyError:
        raise HTTPException(404, f"no investigation {inv_id}") from None
    _check_visibility(meta)
    return meta


def _check_visibility(meta: dict) -> None:
    if get_settings().public_demo and meta["backend"] != "replay":
        raise HTTPException(403, "the public demo only exposes replay investigations")


@app.get("/api/investigations/{inv_id}/report")
def investigation_report(inv_id: str, download: bool = False) -> JSONResponse:
    try:
        snapshot = store.snapshot(inv_id)
    except KeyError:
        raise HTTPException(404, f"no investigation {inv_id}") from None
    _check_visibility(snapshot["meta"])
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = (
            'attachment; filename="investigation-report.json"'
        )
    return JSONResponse(build_report(snapshot), headers=headers)


@app.get("/api/investigations/{inv_id}/events")
async def events(
    inv_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None),
) -> EventSourceResponse:
    get_investigation(inv_id)
    start_seq = 0
    if last_event_id:
        if (
            len(last_event_id) > 18
            or not last_event_id.isascii()
            or not last_event_id.isdecimal()
        ):
            raise HTTPException(
                422, "Last-Event-ID must be a nonnegative decimal sequence"
            )
        start_seq = int(last_event_id) + 1

    async def stream():
        seq = start_seq
        while not await request.is_disconnected():
            for event in store.events_since(inv_id, seq):
                seq = event["seq"] + 1
                yield {
                    "id": str(event["seq"]),
                    "event": event["type"],
                    "data": json.dumps(event["data"]),
                }
            if store.is_done(inv_id) and not store.events_since(inv_id, seq):
                break
            await asyncio.sleep(0.3)

    return EventSourceResponse(stream())
