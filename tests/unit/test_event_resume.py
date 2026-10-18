import json

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    store = InvestigationStore(storage)
    monkeypatch.setattr(main, "store", store)
    inv_id = store.create("replay", {})
    for kind, data in [
        ("started", {"backend": "replay"}),
        ("step", {"kind": "intervention", "cost_usd": 0.1}),
        ("awaiting_approval", {"plan": {}}),
        ("resumed", {}),
        ("finished", {"status": "confirmed"}),
    ]:
        store.emit(inv_id, kind, data)
    return TestClient(main.app), inv_id, storage


def read_events(client, inv_id, cursor=None):
    headers = {} if cursor is None else {"Last-Event-ID": cursor}
    response = client.get(f"/api/investigations/{inv_id}/events", headers=headers)
    assert response.status_code == 200
    records = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if "event" in fields:
            records.append(
                (int(fields["id"]), fields["event"], json.loads(fields["data"]))
            )
    return records


def test_resume_at_every_boundary_delivers_each_event_once(saved_run):
    client, inv_id, _ = saved_run
    all_events = read_events(client, inv_id)
    assert [event[0] for event in all_events] == list(range(5))
    assert [event[1] for event in all_events] == [
        "started",
        "step",
        "awaiting_approval",
        "resumed",
        "finished",
    ]
    for index, event in enumerate(all_events):
        resumed = read_events(client, inv_id, str(event[0]))
        assert all_events[: index + 1] + resumed == all_events
    assert read_events(client, inv_id, "") == all_events


def test_resume_after_store_restart_keeps_ids_and_payloads(saved_run, monkeypatch):
    client, inv_id, storage = saved_run
    original = read_events(client, inv_id)
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    assert read_events(client, inv_id, "1") == original[2:]


@pytest.mark.parametrize("cursor", ["-1", "1.5", "abc", "+1", " 1", "1" * 19])
def test_invalid_cursor_rejected_before_streaming(saved_run, cursor):
    client, inv_id, _ = saved_run
    response = client.get(
        f"/api/investigations/{inv_id}/events", headers={"Last-Event-ID": cursor}
    )
    assert response.status_code == 422


def test_unknown_investigation_still_returns_404(saved_run):
    client, _, _ = saved_run
    response = client.get(
        "/api/investigations/missing/events", headers={"Last-Event-ID": "0"}
    )
    assert response.status_code == 404
