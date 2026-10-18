from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main, runners
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    return TestClient(main.app)


def report(client, inv_id):
    response = client.get(f"/api/investigations/{inv_id}/report")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def test_replay_report_preserves_steps_and_downloads_after_restart(
    client, tmp_path, monkeypatch
):
    inv_id = main.store.create("replay", {}, "Saved demo")
    runners.replay(main.store, inv_id, "fault6-tf", 0)
    before = report(client, inv_id)
    assert before["execution_status"] == "finished"
    assert before["partial"] is False
    assert before["verdict"]["status"] == "confirmed"
    assert before["comparison"] is not None
    assert before["steps"] == runners.replay_steps(
        runners.demo_cases()["fault6-tf"]["report"]
    )
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
    response = client.get(f"/api/investigations/{inv_id}/report?download=true")
    assert response.json() == before
    assert response.headers["content-disposition"].startswith("attachment;")


def test_waiting_report_contains_protocol_and_approval_progress(client):
    inv_id = main.store.create("tf", {"task": "banking77"})
    main.store.emit(inv_id, "started", {"protocol": {"name": "custom"}})
    main.store.emit(
        inv_id, "step", {"kind": "verdict", "status": "confirmed", "patch": "diff"}
    )
    main.store.emit(
        inv_id,
        "awaiting_approval",
        {"plan": {"cost_usd": 0.1}, "summary": {"status": "confirmed"}},
    )
    waiting = report(client, inv_id)
    assert waiting["partial"] is True
    assert waiting["execution_status"] == "awaiting_approval"
    assert waiting["protocol"]["name"] == "custom"
    assert waiting["approval"] is None
    main.store.resume(inv_id, True)
    main.store.emit(inv_id, "step", {"kind": "comparison", "gain": 0.3})
    main.store.emit(inv_id, "finished", {"status": "confirmed"})
    completed = report(client, inv_id)
    assert completed["approval"] == {"approved": True}
    assert completed["comparison"]["gain"] == 0.3
    assert completed["partial"] is False
    assert completed["last_event_seq"] > waiting["last_event_seq"]


def test_failure_and_interrupted_run_remain_partial(client, tmp_path, monkeypatch):
    failed = main.store.create("tf", {})
    main.store.emit(failed, "failed", {"error": "Missing dataset"})
    assert report(client, failed)["error"] == {"error": "Missing dataset"}
    running = main.store.create("tf", {})
    main.store.emit(running, "started", {})
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
    interrupted = report(client, running)
    assert interrupted["execution_status"] == "failed"
    assert interrupted["partial"] is True
    assert "interrupted" in interrupted["error"]["error"]
    assert interrupted["diagnosis_status"] is None


def test_snapshot_does_not_expose_mutable_store_data(client):
    inv_id = main.store.create("tf", {"nested": {"value": 1}})
    main.store.emit(inv_id, "step", {"kind": "probe", "raw": {"count": 2}})
    snapshot = main.store.snapshot(inv_id)
    snapshot["meta"]["case"]["nested"]["value"] = 99
    snapshot["events"][0]["data"]["raw"]["count"] = 99
    again = main.store.snapshot(inv_id)
    assert again["meta"]["case"]["nested"]["value"] == 1
    assert again["events"][0]["data"]["raw"]["count"] == 2


def test_unknown_and_public_reports(client, monkeypatch):
    assert client.get("/api/investigations/missing/report").status_code == 404
    live = main.store.create("tf", {})
    replay = main.store.create("replay", {})
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(public_demo=True))
    assert client.get(f"/api/investigations/{live}/report").status_code == 403
    assert report(client, replay)["partial"] is True
