from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage


@pytest.fixture
def runs(tmp_path, monkeypatch):
    store = InvestigationStore(LocalStorage(tmp_path))
    monkeypatch.setattr(main, "store", store)
    ids = {}
    for backend in ("replay", "tf", "local"):
        inv_id = store.create(backend, {"job_id": f"private-{backend}"}, backend)
        store.emit(inv_id, "started", {"dataset": f"/private/{backend}/train.jsonl"})
        store.emit(inv_id, "finished", {})
        ids[backend] = inv_id
    return TestClient(main.app), ids


@pytest.mark.parametrize("public", [True, False])
def test_list_filters_live_runs_only_in_public_mode(runs, monkeypatch, public):
    client, ids = runs
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=public)
    )
    response = client.get("/api/investigations")
    assert response.status_code == 200
    expected = {ids["replay"]} if public else set(ids.values())
    assert {row["id"] for row in response.json()} == expected


@pytest.mark.parametrize("backend", ["tf", "local", "replay"])
@pytest.mark.parametrize("suffix", ["", "/events", "/report", "/report?download=true"])
def test_all_read_routes_obey_public_visibility(runs, monkeypatch, backend, suffix):
    client, ids = runs
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(public_demo=True))
    response = client.get(f"/api/investigations/{ids[backend]}{suffix}")
    assert response.status_code == (200 if backend == "replay" else 403)
    if backend != "replay":
        assert "private-" not in response.text
        assert "/private/" not in response.text
        assert "text/event-stream" not in response.headers.get("content-type", "")
        assert "content-disposition" not in response.headers


@pytest.mark.parametrize("backend", ["tf", "local"])
@pytest.mark.parametrize("suffix", ["", "/events", "/report"])
def test_private_mode_keeps_live_reads(runs, monkeypatch, backend, suffix):
    client, ids = runs
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    assert client.get(f"/api/investigations/{ids[backend]}{suffix}").status_code == 200
