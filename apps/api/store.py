"""B22: investigations and their event streams, kept in memory and written to storage.

Each investigation has investigations/<id>/meta.json (status, backend, case) and
investigations/<id>/events.jsonl. Runners in background threads call emit(); the
SSE endpoint reads events_since(). Storage can be local or the bucket, so finished
investigations survive a restart of the app. With a SQLite repository, metadata
and events are instead saved together in a database transaction.

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

from core.config import Settings, get_settings
from core.connectors.storage import Storage, get_storage
from core.db.investigations import OwnershipLost, SQLiteInvestigations

TERMINAL = {"finished", "failed"}
WAITING = "awaiting_approval"
WAITING_PAID = "awaiting_paid_step"


class InvestigationStore:
    def __init__(
        self, storage: Storage, repository: SQLiteInvestigations | None = None
    ):
        self.storage = storage
        self.repository = repository
        self.owners: dict[str, str] = {}
        self.lock = threading.RLock()
        self.meta: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}

    def create(self, backend, case, title="", *, request_identity=None):
        return self._create(backend, case, title, request_identity=request_identity)[0]

    def _create(
        self,
        backend: str,
        case: dict[str, Any],
        title: str = "",
        *,
        request_identity: tuple[str, str] | None = None,
    ) -> tuple[str, bool]:
        inv_id = f"inv-{uuid.uuid4().hex[:8]}"
        meta = {
            "id": inv_id,
            "status": "queued",
            "backend": backend,
            "title": title,
            "case": case,
            "created_at": time.time(),
        }
        if request_identity is not None:
            meta["request_key_hash"], meta["request_digest"] = request_identity
        if self.repository is not None:
            token = uuid.uuid4().hex
            inv_id, created = self.repository.create(meta, token)
            if created:
                self.owners[inv_id] = token
            return inv_id, created
        with self.lock:
            self.meta[inv_id] = meta
            self.events[inv_id] = []
            self._save(inv_id)
        return inv_id, True

    def find_request(self, key_hash: str, digest: str) -> str | None:
        """Resolve the request identity from durable state."""
        if self.repository is not None:
            return self.repository.find_request(key_hash, digest)
        with self.lock:
            for meta in self.list():
                if meta.get("request_key_hash") != key_hash:
                    continue
                if meta.get("request_digest") != digest:
                    raise ValueError(
                        "Idempotency key was already used with a different request"
                    )
                return meta["id"]
        return None

    def create_once(self, backend, case, title, key_hash, digest):
        if self.repository is not None:
            return self._create(
                backend, case, title, request_identity=(key_hash, digest)
            )
        with self.lock:
            existing = self.find_request(key_hash, digest)
            if existing is not None:
                return existing, False
            return self.create(
                backend, case, title, request_identity=(key_hash, digest)
            ), True

    def emit(self, inv_id: str, kind: str, data: dict[str, Any]) -> None:
        if self.repository is not None:
            if (
                getattr(self, "ownership_failed", None) is not None
                and self.ownership_failed.is_set()
            ):
                raise OwnershipLost("Worker lost its execution lease")
            self.repository.append(inv_id, self.owners.get(inv_id), kind, data)
            return
        with self.lock:
            events = self.events[inv_id]
            events.append({"seq": len(events), "type": kind, "data": data})
            if kind in ("started", "resumed"):
                self.meta[inv_id]["status"] = "running"
            elif kind in TERMINAL or kind in {WAITING, WAITING_PAID}:
                self.meta[inv_id]["status"] = kind
            self._save(inv_id)

    def events_since(self, inv_id: str, seq: int) -> list[dict[str, Any]]:
        if self.repository is not None:
            return self.repository.load(inv_id)[1][seq:]
        self._load(inv_id)
        with self.lock:
            return list(self.events[inv_id][seq:])

    def resume(
        self, inv_id: str, approved: bool, proposal_seq: int | None = None
    ) -> dict[str, Any]:
        """Take the user's answer to a waiting proposal; returns what was waiting."""
        if self.repository is not None:
            return self._resume_sqlite(inv_id, approved, WAITING, proposal_seq)
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
            self._check_proposal_seq(inv_id, WAITING, proposal_seq)
            self.emit(inv_id, "resumed", {"approved": approved})
            return waiting

    def resume_paid(
        self, inv_id: str, approved: bool, proposal_seq: int | None = None
    ) -> dict[str, Any]:
        """Consume one durable decision for the next paid investigation step."""
        if self.repository is not None:
            return self._resume_sqlite(inv_id, approved, WAITING_PAID, proposal_seq)
        self._load(inv_id)
        with self.lock:
            if self.meta[inv_id]["status"] != WAITING_PAID:
                raise ValueError(f"{inv_id} is not waiting for a paid-step decision")
            waiting = next(
                (
                    event["data"]
                    for event in reversed(self.events[inv_id])
                    if event["type"] == WAITING_PAID
                ),
                None,
            )
            if waiting is None:
                raise ValueError(f"{inv_id} has no persisted paid-step proposal")
            self._check_proposal_seq(inv_id, WAITING_PAID, proposal_seq)
            self.emit(
                inv_id,
                "resumed",
                {"approved": approved, "paid_step": waiting["proposal"]["kind"]},
            )
            return waiting

    def is_done(self, inv_id: str) -> bool:
        return self.get(inv_id)["status"] in TERMINAL

    def get(self, inv_id: str) -> dict[str, Any]:
        if self.repository is not None:
            return self.repository.load(inv_id)[0]
        self._load(inv_id)
        with self.lock:
            return dict(self.meta[inv_id])

    def snapshot(self, inv_id: str) -> dict[str, Any]:
        """Read metadata and events together for a consistent report."""
        if self.repository is not None:
            meta, events = self.repository.load(inv_id)
            return {"meta": meta, "events": events}
        with self.lock:
            self._load(inv_id)
            return copy.deepcopy(
                {"meta": self.meta[inv_id], "events": self.events[inv_id]}
            )

    def list(self) -> list[dict[str, Any]]:
        if self.repository is not None:
            ids = set(self.repository.ids())
        else:
            ids = {
                key.split("/")[1]
                for key in self.storage.list_keys("investigations/")
                if key.endswith("/meta.json")
            }
        rows = [self.get(i) for i in ids | set(self.meta)]
        return sorted(rows, key=lambda m: m["created_at"], reverse=True)

    def _check_proposal_seq(self, inv_id, kind, proposal_seq):
        if proposal_seq is not None:
            current = next(
                e["seq"] for e in reversed(self.events[inv_id]) if e["type"] == kind
            )
            if current != proposal_seq:
                raise ValueError("Approval refers to a different proposal")

    def _resume_sqlite(self, inv_id, approved, waiting_kind, proposal_seq):
        token = uuid.uuid4().hex
        waiting = self.repository.resume(
            inv_id, token, approved, waiting_kind, proposal_seq
        )
        self.owners[inv_id] = token
        return waiting

    def worker(self, inv_id):
        """Freeze this execution's token so a later approval cannot refresh it."""
        if self.repository is None:
            return self
        worker = copy.copy(self)
        worker.owners = {inv_id: self.owners[inv_id]}
        worker.ownership_failed = threading.Event()
        return worker

    def assert_owner(self, inv_id):
        if self.repository is not None:
            if (
                getattr(self, "ownership_failed", None) is not None
                and self.ownership_failed.is_set()
            ):
                raise OwnershipLost("Worker lost its execution lease")
            self.repository.assert_owner(inv_id, self.owners.get(inv_id))

    def run_owned(self, inv_id, target, args, kwargs):
        """Dispatch a claim once and renew it while the worker is alive."""
        if self.repository is None:
            return target(self, inv_id, *args, **kwargs)
        stopped = threading.Event()
        token = self.owners[inv_id]

        def heartbeat():
            while not stopped.wait(self.repository.lease_seconds / 3):
                try:
                    self.repository.renew(inv_id, token)
                except Exception:  # noqa: BLE001 - any renewal failure fences payment
                    self.ownership_failed.set()
                    return

        try:
            self.repository.activate(inv_id, token)
        except OwnershipLost:
            return
        keeper = threading.Thread(target=heartbeat, daemon=True)
        keeper.start()
        try:
            self.assert_owner(inv_id)
            target(self, inv_id, *args, **kwargs)
            if self.get(inv_id)["status"] in {"queued", "running"}:
                self.emit(
                    inv_id,
                    "failed",
                    {"error": "Worker returned without a waiting or terminal state"},
                )
        except OwnershipLost:
            pass  # A stale worker cannot append a failure to a newer execution.
        except Exception as err:  # noqa: BLE001 - close unhandled worker failures
            try:
                self.emit(inv_id, "failed", {"error": f"{type(err).__name__}: {err}"})
            except OwnershipLost:
                pass
        finally:
            stopped.set()
            keeper.join(timeout=1)

    def _save(self, inv_id: str) -> None:
        with self.lock:
            if self.repository is not None:
                self.repository.save(self.meta[inv_id], self.events[inv_id])
                return
            meta = json.dumps(self.meta[inv_id])
            events = "".join(json.dumps(e) + "\n" for e in self.events[inv_id])
            self.storage.put_text(f"investigations/{inv_id}/events.jsonl", events)
            self.storage.put_text(f"investigations/{inv_id}/meta.json", meta)

    def _load(self, inv_id: str) -> None:
        with self.lock:
            if inv_id in self.meta:
                return
            if self.repository is not None:
                meta, events = self.repository.load(inv_id)
            else:
                key = f"investigations/{inv_id}/meta.json"
                if not self.storage.exists(key):
                    raise KeyError(inv_id)
                meta = json.loads(self.storage.get_text(key))
                lines = self.storage.get_text(f"investigations/{inv_id}/events.jsonl")
                events = [json.loads(x) for x in lines.splitlines() if x]
            # Events are saved before metadata; recover the latest transition.
            persisted_status = meta["status"]
            for event in events:
                kind = event["type"]
                if kind in ("started", "resumed"):
                    meta["status"] = "running"
                elif kind in TERMINAL or kind in {WAITING, WAITING_PAID}:
                    meta["status"] = kind
            self.meta[inv_id] = meta
            self.events[inv_id] = events
            if meta["status"] not in TERMINAL and meta["status"] not in {
                WAITING,
                WAITING_PAID,
            }:
                self.emit(
                    inv_id,
                    "failed",
                    {
                        "error": "Investigation interrupted by an API process restart. "
                        "Saved results remain available. Provider jobs may still be "
                        "running; check them before starting another investigation.",
                        "reason": "process_restart",
                    },
                )
            elif meta["status"] != persisted_status:
                self._save(inv_id)


def get_investigation_store(settings: Settings | None = None) -> InvestigationStore:
    settings = settings or get_settings()
    if settings.investigation_backend == "storage":
        from core.db.single_worker import hold_single_worker

        hold_single_worker(settings.artifacts_dir)
    repository = (
        SQLiteInvestigations(settings.database_url)
        if settings.investigation_backend == "sqlite"
        else None
    )
    return InvestigationStore(get_storage(settings), repository)
