import json
from types import SimpleNamespace

import pytest

from core.db.spending import BudgetExceeded, SpendingLedger
from scripts import fresh_answer_calibration as batch


def setup(tmp_path, monkeypatch, ambiguous=False, cap=5):
    calls = []
    plan = {
        "model": "Qwen/Qwen3-0.6B",
        "cap_usd": cap,
        "recipe": {},
        "jobs": {
            "healthy-s1101": {
                "train": "train.jsonl",
                "validation": "validation.jsonl",
                "seed": 1101,
                "reservation_usd": 0.1,
            }
        },
    }

    def load(root, expected):
        path = root / "state.json"
        return plan, json.loads(path.read_text()) if path.exists() else {}

    def create(*args, **kwargs):
        calls.append("create")
        if ambiguous:
            raise TimeoutError("provider may have accepted")
        return "ftjob-test"

    def upload(path):
        calls.append("upload")
        return "file-test"

    monkeypatch.setattr(batch, "load", load)
    monkeypatch.setattr(
        batch, "provider", lambda env: SimpleNamespace(upload=upload, create_job=create)
    )
    return calls


def test_batch_reuses_successful_submission(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch)
    batch.submit(tmp_path, "digest", None)
    batch.submit(tmp_path, "digest", None)
    assert calls == ["upload", "upload", "create"]
    assert (
        SpendingLedger(tmp_path / "spending.sqlite3", 5).snapshot()["committed_usd"]
        == 0.1
    )


def test_batch_retains_unknown_submission_reservation(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch, ambiguous=True)
    with pytest.raises(TimeoutError):
        batch.submit(tmp_path, "digest", None)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        batch.submit(tmp_path, "digest", None)
    assert calls == ["upload", "upload", "create"]
    ledger = SpendingLedger(tmp_path / "spending.sqlite3", 5).snapshot()
    assert ledger["operations"][0]["status"] == "pending"


def test_batch_budget_gate_precedes_provider_calls(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch, cap=0.05)
    with pytest.raises(BudgetExceeded):
        batch.submit(tmp_path, "digest", None)
    assert calls == []
