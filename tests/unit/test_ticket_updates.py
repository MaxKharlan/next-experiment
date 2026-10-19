import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.tasks.ticket_updates import build, parse_history
from core.db.spending import SpendingLedger
from scripts import ticket_update_development as pilot


def test_text_oracle_ignores_void_and_other_ticket():
    text = """Return the current record for T-800000.
Initial T-800000: product="router"; priority="low"; callback=false; due_date=null
Initial T-800001: product="scanner"; priority="high"; callback=true; due_date=null
Update 1 for T-800000 [applied]: priority="normal"
Update 2 for T-800001 [applied]: priority="high"
Update 3 for T-800000 [void]: priority="low"
Update 4 for T-800000 [applied]: callback=true"""
    assert parse_history(text) == {
        "ticket_id": "T-800000",
        "product": "router",
        "priority": "normal",
        "callback": True,
        "due_date": None,
    }


def test_development_has_no_final_test_and_reproduces(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    a, b = build(left), build(right)
    assert set(a) == {"train", "validation", "check"}
    assert {r: v["sha256"] for r, v in a.items()} == {
        r: v["sha256"] for r, v in b.items()
    }
    assert not (left / "test.jsonl").exists()
    for path in left.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert parse_history(row["messages"][1]["content"]) == json.loads(
                row["messages"][-1]["content"]
            )


def test_admission_rejects_ceiling_and_weak_learning():
    assert not pilot.base_gate(1, {"minimum": 0.1, "maximum": 0.9})
    rule = {
        "healthy_minimum": 0.1,
        "healthy_maximum": 0.95,
        "minimum_healthy_gain_over_base": 0.05,
        "minimum_healthy_minus_damaged": 0.08,
    }
    assert pilot.sensitivity(
        0.6, {"healthy": [0.65] * 3, "damaged": [0.64] * 3}, rule
    ) == ["insufficient_damage_sensitivity"]
    assert "healthy_floor_or_ceiling" in pilot.sensitivity(
        0.5, {"healthy": [1] * 3, "damaged": [0.6] * 3}, rule
    )


def setup_submission(root, monkeypatch):
    plan = {
        "id": "new-pilot",
        "model": "model",
        "recipe": {},
        "base_gate": {"minimum": 0.1, "maximum": 0.9},
        "jobs": {
            "healthy-s1201": {
                "train": "train",
                "validation": "valid",
                "seed": 1201,
                "reservation_usd": 0.2,
            }
        },
    }
    (root / "scores").mkdir()
    (root / "scores/base.json").write_text(
        json.dumps({"protocol_sha256": "hash", "result": {"value": 0.5}})
    )

    def load(root):
        state = root / "state.json"
        return plan, json.loads(state.read_text()) if state.exists() else {}, "hash"

    monkeypatch.setattr(pilot, "load", load)
    monkeypatch.setattr(
        pilot, "checked_score", lambda path, plan, digest: json.loads(path.read_text())
    )
    calls = []

    def create(*args, **kwargs):
        calls.append("create")
        raise TimeoutError("unknown provider outcome")

    monkeypatch.setattr(
        pilot,
        "provider",
        lambda env: SimpleNamespace(upload=lambda path: "file", create_job=create),
    )
    ledger = SpendingLedger(root / "spending.sqlite3", 5)
    ledger.reserve("historical", "old", 0.3)
    ledger.settle("historical", 0.2710464)
    return calls, ledger


def test_unknown_submission_keeps_prior_history_and_hold(tmp_path, monkeypatch):
    calls, ledger = setup_submission(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError):
        pilot.submit(tmp_path, None)
    with pytest.raises(ValueError, match="Ambiguous"):
        pilot.submit(tmp_path, None)
    assert calls == ["create"]
    snapshot = ledger.snapshot()
    assert snapshot["committed_usd"] == pytest.approx(0.4710464)
    assert [r["status"] for r in snapshot["operations"]] == ["settled", "pending"]


def test_terminal_pilot_cannot_submit(tmp_path, monkeypatch):
    calls, ledger = setup_submission(tmp_path, monkeypatch)
    (tmp_path / "decision.json").write_text("{}")
    with pytest.raises(ValueError, match="Terminal"):
        pilot.submit(tmp_path, None)
    assert calls == []
    assert len(ledger.snapshot()["operations"]) == 1


def test_paid_gate_rechecks_saved_answers(tmp_path):
    data = build(tmp_path / "data")
    rows = [
        json.loads(line)
        for line in Path(data["check"]["path"]).read_text().splitlines()
    ]
    record = {
        "protocol_sha256": "digest",
        "result": {
            "data_sha256": data["check"]["sha256"],
            "runtime": {},
            "metric": "json_fields_exact",
            "value": 1.0,
            "examples": [
                {
                    "answer": r["messages"][-1]["content"],
                    "expected": r["messages"][-1]["content"],
                    "score": 1.0,
                }
                for r in rows
            ],
        },
    }
    path = tmp_path / "score.json"
    path.write_text(json.dumps(record))
    plan = {
        "datasets": data,
        "evaluation_runtime": {},
        "evaluator": {"metric": "json_fields_exact"},
    }
    assert pilot.checked_score(path, plan, "digest")["result"]["value"] == 1
    record["result"]["examples"][0]["answer"] = "{}"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="prediction score changed"):
        pilot.checked_score(path, plan, "digest")
