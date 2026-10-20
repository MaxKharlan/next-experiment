"""Public boundary tests: budget persistence, uploads and idempotent execution."""

import copy
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.judge import service as s
from core.db.spending import BudgetExceeded


def payload():
    result = {"task": "Classify delivery or billing", "symptom": "Loss barely improves"}
    for role, count in [("train", 16), ("validation", 4), ("check", 4), ("test", 4)]:
        result[role] = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{role} example {i}: where is my parcel?",
                    },
                    {"role": "assistant", "content": "delivery"},
                ]
            }
            for i in range(count)
        ]
    return result


class MemoryHub(s.HubStore):
    def __init__(self, cap=15):
        self.lock = threading.RLock()
        self.files = {"wallet.json": {"cap_usd": cap, "runs": {}}}
        self.head = 0

    def read(self, name, revision=None):
        return copy.deepcopy(self.files[name])

    def wallet(self):
        return self.head, self.read("wallet.json")

    def commit(self, files, parent):
        assert parent == self.head
        self.files.update(copy.deepcopy(files))
        self.head += 1


def test_upload_boundaries_and_split_leakage(tmp_path):
    upload = s.Upload.model_validate(payload())
    protocol = s.prepare(upload, "a" * 32, tmp_path)
    assert protocol["original_job_verified"] is False
    assert protocol["model"] == "Qwen/Qwen3-0.6B"
    assert protocol["pair_estimate_usd"] < 0.75
    assert all(
        Path(d["path"]).is_relative_to(tmp_path) for d in protocol["datasets"].values()
    )
    data = payload()
    data["check"][0] = data["train"][0]
    with pytest.raises(ValueError, match="Identical record"):
        s.prepare(s.Upload.model_validate(data), "b" * 32, tmp_path)
    for change in (
        {"model": "evil/path"},
        {"n_epochs": 100},
        {"train": []},
        {"learning_rate": float("nan")},
    ):
        with pytest.raises(ValueError):
            s.Upload.model_validate({**payload(), **change})


def test_durable_claim_auth_and_unknown_cost():
    hub = MemoryHub(cap=1)
    assert hub.claim("a", "key", "payload") is True
    assert hub.claim("a", "key", "payload") is False
    with pytest.raises(ValueError):
        hub.claim("a", "key", "different")
    with pytest.raises(BudgetExceeded):
        hub.claim("b", "key", "payload")
    with pytest.raises(s.HTTPException):
        hub.authenticated("a", "wrong key")
    hub.save("a", {"status": "failed", "events": []})
    assert hub.files["wallet.json"]["runs"]["a"]["reserved_usd"] == 1
    assert not hub.available()["available"]


def test_terminal_known_cost_releases_unused_allowance():
    hub = MemoryHub(cap=1)
    hub.claim("a", "key", "payload")
    hub.save(
        "a",
        {"status": "complete"},
        {"committed_usd": 0.1, "operations": [{"status": "settled"}]},
    )
    assert hub.available()["funded_remaining_usd"] == 0.9
    # A new container sees the same wallet; no reset from an environment cap.
    assert hub.read("wallet.json")["cap_usd"] == 1


def test_api_idempotence_and_no_public_list(monkeypatch, tmp_path):
    hub = MemoryHub()
    monkeypatch.setattr(s, "_store", hub)
    monkeypatch.setattr(s, "ROOT", tmp_path)
    launched = []

    class Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            launched.append(self.kwargs)

    monkeypatch.setattr(s.threading, "Thread", Thread)
    with TestClient(s.app) as client:
        headers = {"X-Run-Id": "a" * 32, "X-Run-Key": "b" * 64}
        assert (
            client.post("/api/runs", json=payload(), headers=headers).status_code == 200
        )
        assert (
            client.post("/api/runs", json=payload(), headers=headers).json()["reused"]
            is True
        )
        assert len(launched) == 1
        assert client.get("/api/runs/" + "a" * 32).status_code == 404
        assert client.get("/api/runs").status_code == 405
        assert client.get("/api/runs/" + "a" * 32, headers=headers).status_code == 200
        assert client.post("/api/runs", content=b"x" * 300001).status_code == 413
        assert (
            client.post(
                "/api/runs",
                json={**payload(), "train_path": "/etc/passwd"},
                headers=headers,
            ).status_code
            == 422
        )


def test_worker_failure_retains_hold_and_keeps_details_private(monkeypatch, tmp_path):
    from core.connectors.token_factory import TokenFactory

    hub = MemoryHub(cap=1)
    run_id = "c" * 32
    hub.claim(run_id, "key", "payload")
    monkeypatch.setattr(s, "ROOT", tmp_path)
    upload = s.Upload.model_validate(payload())
    protocol = s.prepare(upload, run_id, tmp_path)

    def unavailable(*args, **kwargs):
        raise RuntimeError("private diagnostic sentinel")

    monkeypatch.setattr(TokenFactory, "from_settings", unavailable)
    s.execute(hub, run_id, upload, protocol)
    saved = hub.authenticated(run_id, "key")
    assert saved["status"] == "failed"
    assert "sentinel" not in str(saved)
    assert not hub.available()["available"]
    diagnostic = tmp_path / run_id / "diagnostic.txt"
    assert "sentinel" in diagnostic.read_text()
    assert diagnostic.stat().st_mode & 0o777 == 0o600
