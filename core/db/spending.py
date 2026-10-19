"""SQLite spending reservations. Ambiguous provider outcomes retain their hold."""

from __future__ import annotations

import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class BudgetExceeded(ValueError):
    pass


class SpendingLedger:
    def __init__(self, path, cap_usd):
        if isinstance(cap_usd, bool) or not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("Budget must be finite and positive")
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, cap REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS spending (id TEXT PRIMARY KEY, purpose TEXT NOT NULL, reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS provider_receipts ("
                "operation_id TEXT PRIMARY KEY, provider_request_id TEXT, "
                "provider_response_id TEXT, outcome TEXT NOT NULL, "
                "http_status INTEGER, received_at REAL NOT NULL)"
            )
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?)", (cap_usd,))
            if db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0] != cap_usd:
                raise ValueError("Persisted budget differs; refusing to reset it")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, operation_id, purpose, amount):
        if isinstance(amount, bool) or not math.isfinite(amount) or amount < 0:
            raise ValueError("Reservation must be finite and nonnegative")
        with self.transaction() as db:
            if db.execute(
                "SELECT 1 FROM spending WHERE id=?", (operation_id,)
            ).fetchone():
                raise ValueError(
                    "Operation already recorded; reconcile before retrying"
                )
            used = db.execute(
                "SELECT COALESCE(SUM(CASE WHEN status='settled' THEN actual ELSE reserved END),0) FROM spending"
            ).fetchone()[0]
            cap = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            if used + amount > cap + 1e-12:
                raise BudgetExceeded("The budget does not cover this operation")
            db.execute(
                "INSERT INTO spending VALUES (?, ?, ?, NULL, 'pending')",
                (operation_id, purpose, amount),
            )

    def settle(self, operation_id, actual):
        if isinstance(actual, bool) or not math.isfinite(actual) or actual < 0:
            raise ValueError("Actual cost must be finite and nonnegative")
        with self.transaction() as db:
            row = db.execute(
                "SELECT status FROM spending WHERE id=?", (operation_id,)
            ).fetchone()
            if row is None or row[0] != "pending":
                raise ValueError("Expected a pending reservation")
            # Record overruns truthfully. Subsequent reservations will fail.
            db.execute(
                "UPDATE spending SET actual=?, status='settled' WHERE id=?",
                (actual, operation_id),
            )

    def record_provider_receipt(
        self, operation_id, *, request_id, response_id, outcome, http_status
    ):
        """Keep response references even if usage is absent; never settle a hold."""
        for value in (request_id, response_id):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    "Provider identifiers must be nonblank strings or null"
                )
        if outcome not in {"response", "http_error"}:
            raise ValueError("Unknown receipt outcome")
        values = (request_id, response_id, outcome, http_status)
        with self.transaction() as db:
            if (
                db.execute(
                    "SELECT 1 FROM spending WHERE id=?", (operation_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("Receipt requires an existing reservation")
            old = db.execute(
                "SELECT provider_request_id, provider_response_id, outcome, http_status "
                "FROM provider_receipts WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if old is not None:
                if old != values:
                    raise ValueError(
                        "Provider receipt conflicts with the recorded operation"
                    )
                return
            db.execute(
                "INSERT INTO provider_receipts VALUES (?, ?, ?, ?, ?, ?)",
                (operation_id, *values, time.time()),
            )

    @staticmethod
    def _receipts(db):
        return [
            dict(
                zip(
                    (
                        "operation_id",
                        "provider_request_id",
                        "provider_response_id",
                        "outcome",
                        "http_status",
                        "received_at",
                    ),
                    row,
                    strict=True,
                )
            )
            for row in db.execute(
                "SELECT * FROM provider_receipts ORDER BY received_at, operation_id"
            )
        ]

    def snapshot(self):
        with self.transaction() as db:
            cap = db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            rows = [
                dict(
                    zip(
                        ("id", "purpose", "reserved_usd", "actual_usd", "status"),
                        row,
                        strict=True,
                    )
                )
                for row in db.execute("SELECT * FROM spending ORDER BY rowid")
            ]
            receipts = self._receipts(db)
        used = sum(
            r["actual_usd"] if r["status"] == "settled" else r["reserved_usd"]
            for r in rows
        )
        return {
            "cap_usd": cap,
            "committed_usd": used,
            "available_usd": max(0, cap - used),
            "operations": rows,
            "provider_receipts": receipts,
        }
