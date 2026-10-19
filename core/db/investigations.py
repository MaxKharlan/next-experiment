"""Transactional investigation state, request identities and execution leases."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import make_url

ACTIVE = {"queued", "running"}
RELEASE = {"finished", "failed", "awaiting_approval", "awaiting_paid_step"}


class OwnershipLost(ValueError):
    """The execution no longer owns this investigation; no further paid calls."""


class SQLiteInvestigations:
    def __init__(self, database_url: str, *, lease_seconds=60, clock=time.time):
        if not 0 < lease_seconds < float("inf"):
            raise ValueError("Lease duration must be finite and positive")
        self.lease_seconds = lease_seconds
        self.clock = clock
        url = make_url(database_url)
        if (
            url.get_backend_name() != "sqlite"
            or not url.database
            or url.database == ":memory:"
        ):
            raise ValueError(
                "Investigation persistence requires a file-backed SQLite URL"
            )
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, connect_args={"timeout": 30})
        schema = MetaData()
        self.snapshots = Table(
            "investigation_snapshots",
            schema,
            Column("id", String, primary_key=True),
            Column("metadata", JSON, nullable=False),
            Column("events", JSON, nullable=False),
        )
        self.leases = Table(
            "investigation_leases",
            schema,
            Column("id", String, primary_key=True),
            Column("token", String, nullable=False),
            Column("deadline", Float, nullable=False),
            Column("executing", Boolean, nullable=False),
        )
        with self.transaction() as connection:
            schema.create_all(connection)

    def save(self, meta: dict[str, Any], events: list[dict[str, Any]]) -> None:
        statement = insert(self.snapshots).values(
            id=meta["id"],
            metadata=meta,
            events=events,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "metadata": statement.excluded.metadata,
                "events": statement.excluded.events,
            },
        )
        with self.transaction() as connection:
            if self._lease(connection, meta["id"]) is not None:
                raise OwnershipLost("Managed snapshots require an execution token")
            connection.execute(statement)

    @contextmanager
    def transaction(self):
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _read(self, connection, inv_id):
        row = connection.execute(
            select(self.snapshots.c.metadata, self.snapshots.c.events).where(
                self.snapshots.c.id == inv_id
            )
        ).first()
        if row is None:
            raise KeyError(inv_id)
        return row[0], row[1]

    def _write(self, connection, meta, events):
        connection.execute(
            update(self.snapshots)
            .where(self.snapshots.c.id == meta["id"])
            .values(metadata=meta, events=events)
        )

    def _lease(self, connection, inv_id):
        return (
            connection.execute(select(self.leases).where(self.leases.c.id == inv_id))
            .mappings()
            .first()
        )

    def _require_owner(self, connection, inv_id, token):
        lease = self._lease(connection, inv_id)
        if (
            not token
            or lease is None
            or lease["token"] != token
            or lease["deadline"] <= self.clock()
        ):
            raise OwnershipLost(
                "Execution ownership expired or changed; reconcile before retrying"
            )
        return lease

    @staticmethod
    def _append(meta, events, kind, data):
        events.append({"seq": len(events), "type": kind, "data": data})
        if kind in {"started", "resumed"}:
            meta["status"] = "running"
        elif kind in RELEASE:
            meta["status"] = kind

    def load(self, inv_id):
        with self.transaction() as connection:
            meta, events = self._read(connection, inv_id)
            lease = self._lease(connection, inv_id)
            if meta["status"] in ACTIVE and (
                lease is None or lease["deadline"] <= self.clock()
            ):
                self._append(
                    meta,
                    events,
                    "failed",
                    {
                        "error": "Investigation execution lease expired. Provider operations may still be running; reconcile before starting another investigation.",
                        "reason": "lease_expired" if lease else "process_restart",
                    },
                )
                self._write(connection, meta, events)
                connection.execute(
                    update(self.leases)
                    .where(self.leases.c.id == inv_id)
                    .values(token="", deadline=0)
                )
            return meta, events

    def _find_request(self, connection, key_hash, digest):
        for meta in connection.scalars(select(self.snapshots.c.metadata)):
            if meta.get("request_key_hash") == key_hash:
                if meta.get("request_digest") != digest:
                    raise ValueError(
                        "Idempotency key was already used with a different request"
                    )
                return meta["id"]
        return None

    def find_request(self, key_hash, digest):
        with self.transaction() as connection:
            return self._find_request(connection, key_hash, digest)

    def create(self, meta, token):
        with self.transaction() as connection:
            if "request_key_hash" in meta:
                existing = self._find_request(
                    connection, meta["request_key_hash"], meta["request_digest"]
                )
                if existing is not None:
                    return existing, False
            connection.execute(
                insert(self.snapshots).values(id=meta["id"], metadata=meta, events=[])
            )
            connection.execute(
                insert(self.leases).values(
                    id=meta["id"],
                    token=token,
                    deadline=self.clock() + self.lease_seconds,
                    executing=False,
                )
            )
            return meta["id"], True

    def append(self, inv_id, token, kind, data):
        with self.transaction() as connection:
            self._require_owner(connection, inv_id, token)
            meta, events = self._read(connection, inv_id)
            self._append(meta, events, kind, data)
            self._write(connection, meta, events)
            if kind in RELEASE:
                connection.execute(
                    update(self.leases)
                    .where(self.leases.c.id == inv_id)
                    .values(token="", deadline=0)
                )

    def resume(self, inv_id, token, approved, waiting_kind, proposal_seq=None):
        with self.transaction() as connection:
            meta, events = self._read(connection, inv_id)
            if meta["status"] != waiting_kind:
                raise ValueError(f"{inv_id} is not waiting for {waiting_kind}")
            waiting = next(
                (e["data"] for e in reversed(events) if e["type"] == waiting_kind), None
            )
            if waiting is None:
                raise ValueError("No persisted approval proposal")
            current_seq = next(
                e["seq"] for e in reversed(events) if e["type"] == waiting_kind
            )
            if proposal_seq is not None and proposal_seq != current_seq:
                raise ValueError("Approval refers to a different proposal")
            data = {"approved": approved}
            if waiting_kind == "awaiting_paid_step":
                data["paid_step"] = waiting["proposal"]["kind"]
            self._append(meta, events, "resumed", data)
            self._write(connection, meta, events)
            claim = insert(self.leases).values(
                id=inv_id,
                token=token,
                deadline=self.clock() + self.lease_seconds,
                executing=False,
            )
            connection.execute(
                claim.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "token": claim.excluded.token,
                        "deadline": claim.excluded.deadline,
                        "executing": False,
                    },
                )
            )
            return waiting

    def assert_owner(self, inv_id, token):
        with self.transaction() as connection:
            self._require_owner(connection, inv_id, token)

    def activate(self, inv_id, token):
        with self.transaction() as connection:
            lease = self._require_owner(connection, inv_id, token)
            if lease["executing"]:
                raise OwnershipLost("This execution has already been dispatched")
            connection.execute(
                update(self.leases)
                .where(self.leases.c.id == inv_id)
                .values(executing=True)
            )

    def renew(self, inv_id, token):
        with self.transaction() as connection:
            self._require_owner(connection, inv_id, token)
            connection.execute(
                update(self.leases)
                .where(self.leases.c.id == inv_id)
                .values(deadline=self.clock() + self.lease_seconds)
            )

    def ids(self) -> list[str]:
        with self.engine.connect() as connection:
            return list(connection.scalars(select(self.snapshots.c.id)))
