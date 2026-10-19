"""Atomic, persistent credit reservations for cached external requests."""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class CreditBudgetError(RuntimeError):
    pass


def amount(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Credits must be finite and nonnegative")
    return value


class CreditLedger:
    def __init__(self, path, total_limit, investigation_limit):
        self.path = str(path)
        self.total_limit = amount(total_limit)
        self.investigation_limit = amount(investigation_limit)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS limits (scope TEXT PRIMARY KEY, cap REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, investigation TEXT NOT NULL, reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL)"
            )
            self._limit(db, "total", self.total_limit)

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

    @staticmethod
    def _limit(db, scope, cap):
        db.execute("INSERT OR IGNORE INTO limits VALUES (?, ?)", (scope, cap))
        if (
            db.execute("SELECT cap FROM limits WHERE scope=?", (scope,)).fetchone()[0]
            != cap
        ):
            raise ValueError("Persisted credit limit differs; refusing to reset it")

    def reserve(self, operation, investigation, estimate):
        amount(estimate)
        with self.transaction() as db:
            self._limit(db, "investigation:" + investigation, self.investigation_limit)
            if db.execute("SELECT 1 FROM requests WHERE id=?", (operation,)).fetchone():
                raise CreditBudgetError(
                    "Request already recorded; reconcile it before retrying"
                )
            rows = db.execute(
                "SELECT investigation, CASE WHEN status='settled' THEN actual ELSE reserved END FROM requests"
            ).fetchall()
            if (
                sum(v for _, v in rows) + estimate > self.total_limit
                or sum(v for i, v in rows if i == investigation) + estimate
                > self.investigation_limit
            ):
                raise CreditBudgetError(
                    "Tavily credit reservation exceeds the persisted limit"
                )
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, NULL, 'pending')",
                (operation, investigation, estimate),
            )

    def settle(self, operation, actual):
        amount(actual)
        with self.transaction() as db:
            row = db.execute(
                "SELECT status FROM requests WHERE id=?", (operation,)
            ).fetchone()
            if row != ("pending",):
                raise ValueError("Expected pending credit reservation")
            db.execute(
                "UPDATE requests SET actual=?, status='settled' WHERE id=?",
                (actual, operation),
            )

    def snapshot(self):
        with self.transaction() as db:
            rows = [
                dict(
                    zip(
                        (
                            "id",
                            "investigation",
                            "reserved_credits",
                            "actual_credits",
                            "status",
                        ),
                        row,
                        strict=True,
                    )
                )
                for row in db.execute("SELECT * FROM requests ORDER BY rowid")
            ]
        committed = {}
        for row in rows:
            value = (
                row["actual_credits"]
                if row["status"] == "settled"
                else row["reserved_credits"]
            )
            inv = row["investigation"]
            committed[inv] = committed.get(inv, 0) + value
        return {
            "unit": "credits",
            "committed_by_investigation": committed,
            "committed_credits": sum(committed.values()),
            "limit_credits": self.total_limit,
            "operations": rows,
        }
