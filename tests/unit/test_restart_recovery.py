import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage


@pytest.mark.parametrize("started", [False, True])
def test_restart_records_one_failure_and_preserves_results(tmp_path, started):
    storage = LocalStorage(tmp_path)
    original = InvestigationStore(storage)
    inv_id = original.create("tf", {})
    if started:
        original.emit(inv_id, "started", {})
        original.emit(inv_id, "llm_call", {"id": "call-1", "cost_usd": 0.01})
        original.emit(inv_id, "step", {"kind": "intervention", "job_id": "job-1"})
    before = original.events_since(inv_id, 0)
    fresh = InvestigationStore(storage)
    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(pool.map(lambda _: fresh.snapshot(inv_id), range(16)))
    assert all(state == states[0] for state in states)
    after = states[0]["events"]
    assert states[0]["meta"]["status"] == "failed"
    assert after[:-1] == before
    assert after[-1]["seq"] == len(before)
    assert after[-1]["type"] == "failed"
    assert after[-1]["data"]["reason"] == "process_restart"
    assert InvestigationStore(storage).snapshot(inv_id) == states[0]


@pytest.mark.parametrize(
    "kind",
    ["finished", "failed", "awaiting_approval", "awaiting_paid_step", "resumed"],
)
def test_events_recover_transition_saved_before_metadata(tmp_path, kind):
    storage = LocalStorage(tmp_path)
    original = InvestigationStore(storage)
    inv_id = original.create("tf", {})
    original.emit(inv_id, "started", {})
    if kind == "resumed":
        original.emit(inv_id, "awaiting_approval", {"plan": {}})
    meta_key = f"investigations/{inv_id}/meta.json"
    old_meta = storage.get_text(meta_key)
    event_data = {"plan": {}, "error": "original failure"}
    if kind == "awaiting_paid_step":
        event_data["proposal"] = {"kind": "causal_check"}
    original.emit(inv_id, kind, event_data)
    storage.put_text(meta_key, old_meta)
    fresh = InvestigationStore(storage)
    snapshot = fresh.snapshot(inv_id)
    expected = "failed" if kind == "resumed" else kind
    assert snapshot["meta"]["status"] == expected
    assert json.loads(storage.get_text(meta_key))["status"] == expected
    if kind == "resumed":
        assert snapshot["events"][-1]["data"]["reason"] == "process_restart"
    else:
        assert snapshot["events"] == original.events_since(inv_id, 0)
    if kind == "awaiting_approval":
        assert fresh.resume(inv_id, False)["plan"] == {}
    if kind == "awaiting_paid_step":
        waiting = fresh.resume_paid(inv_id, False)
        assert waiting["plan"] == {}


def test_restart_failure_reaches_stream_and_report(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    original = InvestigationStore(storage)
    inv_id = original.create("tf", {})
    original.emit(inv_id, "started", {})
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    monkeypatch.setattr(
        main, "get_settings", lambda: type("S", (), {"public_demo": False})()
    )
    client = TestClient(main.app)
    stream = client.get(f"/api/investigations/{inv_id}/events")
    assert stream.status_code == 200
    assert "event: failed" in stream.text
    assert "process_restart" in stream.text
    report = client.get(f"/api/investigations/{inv_id}/report").json()
    assert report["execution_status"] == "failed"
    assert report["partial"] is True
    assert report["error"]["reason"] == "process_restart"
