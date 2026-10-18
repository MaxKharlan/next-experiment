"""B22: investigations and their event streams, kept in memory and written to storage.

Each investigation has investigations/<id>/meta.json (status, backend, case) and
investigations/<id>/events.jsonl. Runners in background threads call emit(); the
SSE endpoint reads events_since(). Storage can be local or the bucket, so finished
investigations survive a restart of the app.

An investigation that proposes the full retrain waits in "awaiting_approval" until
the user answers; resume() takes the answer once and returns the plan. A restart
keeps it waiting, because nothing was running.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from typing import Any

from core.connectors.storage import Storage

TERMINAL = {"finished", "failed"}
WAITING = "awaiting_approval"


class InvestigationStore:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.lock = threading.RLock()
        self.meta: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}

    def create(self, backend: str, case: dict[str, Any], title: str = "") -> str:
        inv_id = f"inv-{uuid.uuid4().hex[:8]}"
        meta = {
            "id": inv_id,
            "status": "queued",
            "backend": backend,
            "title": title,
            "case": case,
            "created_at": time.time(),
        }
        with self.lock:
            self.meta[inv_id] = meta
            self.events[inv_id] = []
            self._save(inv_id)
        return inv_id

    def emit(self, inv_id: str, kind: str, data: dict[str, Any]) -> None:
        with self.lock:
            events = self.events[inv_id]
            events.append({"seq": len(events), "type": kind, "data": data})
            if kind in ("started", "resumed"):
                self.meta[inv_id]["status"] = "running"
            elif kind in TERMINAL or kind == WAITING:
                self.meta[inv_id]["status"] = kind
            self._save(inv_id)

    def events_since(self, inv_id: str, seq: int) -> list[dict[str, Any]]:
        self._load(inv_id)
        with self.lock:
            return list(self.events[inv_id][seq:])

    def resume(self, inv_id: str, approved: bool) -> dict[str, Any]:
        """Take the user's answer to a waiting proposal; returns what was waiting."""
        self._load(inv_id)
        with self.lock:
            if self.meta[inv_id]["status"] != WAITING:
                raise ValueError(f"{inv_id} is not waiting for approval")
            waiting = next(
                (
                    e["data"]
                    for e in reversed(self.events[inv_id])
                    if e["type"] == WAITING
                ),
                None,
            )
            if waiting is None:
                raise ValueError(f"{inv_id} has no persisted approval proposal")
            self.emit(inv_id, "resumed", {"approved": approved})
            return waiting

    def is_done(self, inv_id: str) -> bool:
        return self.get(inv_id)["status"] in TERMINAL

    def get(self, inv_id: str) -> dict[str, Any]:
        self._load(inv_id)
        with self.lock:
            return dict(self.meta[inv_id])

    def snapshot(self, inv_id: str) -> dict[str, Any]:
        """Copy metadata and events under one lock for a consistent report."""
        with self.lock:
            self._load(inv_id)
            return copy.deepcopy(
                {"meta": self.meta[inv_id], "events": self.events[inv_id]}
            )

    def list(self) -> list[dict[str, Any]]:
        ids = {
            key.split("/")[1]
            for key in self.storage.list_keys("investigations/")
            if key.endswith("/meta.json")
        }
        rows = [self.get(i) for i in ids | set(self.meta)]
        return sorted(rows, key=lambda m: m["created_at"], reverse=True)

    def _save(self, inv_id: str) -> None:
        with self.lock:
            meta = json.dumps(self.meta[inv_id])
            events = "".join(json.dumps(e) + "\n" for e in self.events[inv_id])
            self.storage.put_text(f"investigations/{inv_id}/events.jsonl", events)
            self.storage.put_text(f"investigations/{inv_id}/meta.json", meta)

    def _load(self, inv_id: str) -> None:
        with self.lock:
            if inv_id in self.meta:
                return
        key = f"investigations/{inv_id}/meta.json"
        if not self.storage.exists(key):
            raise KeyError(inv_id)
        meta = json.loads(self.storage.get_text(key))
        if meta["status"] not in TERMINAL and meta["status"] != WAITING:
            # it was running in an app process that has stopped since
            meta["status"] = "failed"
        lines = self.storage.get_text(f"investigations/{inv_id}/events.jsonl")
        with self.lock:
            self.meta[inv_id] = meta
            self.events[inv_id] = [json.loads(x) for x in lines.splitlines() if x]
