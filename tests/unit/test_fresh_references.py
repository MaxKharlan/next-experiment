import json
from types import SimpleNamespace

import pytest

from core.db.spending import BudgetExceeded, SpendingLedger
from scripts import fresh_answer_references as batch


def setup(tmp_path, monkeypatch, *, ambiguous=False, prior_cost=0):
    ledger = SpendingLedger(tmp_path / "spending.sqlite3", 5)
    ledger.reserve("development", "development", prior_cost)
    ledger.settle("development", prior_cost)
    calls = []
    plan = {
        "model": "Qwen/Qwen3-0.6B",
        "recipe": {},
        "jobs": {
            "strong": {
                "train": "train.jsonl",
                "validation": "validation.jsonl",
                "seed": 4103,
                "reservation_usd": 0.1,
            }
        },
    }

    def load(root, expected):
        path = root / "state.json"
        return plan, json.loads(path.read_text()) if path.exists() else {}, ledger

    def create(*args, **kwargs):
        calls.append("create")
        if ambiguous:
            raise TimeoutError("unknown outcome")
        return "ftjob-fresh"

    def upload(path):
        calls.append("upload")
        return "file-fresh"

    monkeypatch.setattr(batch, "load", load)
    monkeypatch.setattr(
        batch, "provider", lambda env: SimpleNamespace(upload=upload, create_job=create)
    )
    return calls, ledger


def test_fresh_references_share_existing_budget(tmp_path, monkeypatch):
    calls, ledger = setup(tmp_path, monkeypatch, prior_cost=4.95)
    with pytest.raises(BudgetExceeded):
        batch.submit(tmp_path, "digest", None)
    assert calls == []
    assert ledger.snapshot()["committed_usd"] == 4.95


def test_fresh_submission_reused_without_paying_again(tmp_path, monkeypatch):
    calls, ledger = setup(tmp_path, monkeypatch, prior_cost=0.28)
    batch.submit(tmp_path, "digest", None)
    batch.submit(tmp_path, "digest", None)
    assert calls == ["upload", "upload", "create"]
    assert ledger.snapshot()["committed_usd"] == pytest.approx(0.38)


def test_unknown_fresh_submission_keeps_reservation(tmp_path, monkeypatch):
    calls, ledger = setup(tmp_path, monkeypatch, ambiguous=True)
    with pytest.raises(TimeoutError):
        batch.submit(tmp_path, "digest", None)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        batch.submit(tmp_path, "digest", None)
    assert calls == ["upload", "upload", "create"]
    assert ledger.snapshot()["operations"][-1]["status"] == "pending"
