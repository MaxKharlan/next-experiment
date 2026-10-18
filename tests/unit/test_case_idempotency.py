from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.connectors.storage import LocalStorage
from tests.e2e.test_investigation_api import make_payload


@pytest.fixture
def setup(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / "store")
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    calls = []

    def worker(store, inv_id, backend, *, prepared):
        calls.append(inv_id)
        store.emit(inv_id, "finished", {"status": "inconclusive"})

    # Synchronous thread double makes worker launch counts deterministic.
    class Thread:
        def __init__(self, target, args, kwargs, daemon):
            self.run = lambda: target(*args, **kwargs)

        def start(self):
            self.run()

    monkeypatch.setattr(main.threading, "Thread", Thread)
    monkeypatch.setattr(main.runners, "live", worker)
    return make_payload(tmp_path), storage, calls


def test_retry_reuses_id_even_after_restart(setup, monkeypatch):
    payload, storage, calls = setup
    # Invoke the route directly so the thread double does not affect TestClient.
    from apps.api.case_input import CaseStart

    req = CaseStart.model_validate(payload)
    first = main.start_from_case(req, "request-1")
    assert first["reused"] is False
    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    again = main.start_from_case(req, "request-1")
    assert again == {"id": first["id"], "reused": True}
    assert calls == [first["id"]]


def test_changed_request_conflicts_and_new_key_can_start(setup):
    from fastapi import HTTPException

    from apps.api.case_input import CaseStart

    payload, _, calls = setup
    main.start_from_case(CaseStart.model_validate(payload), "same")
    payload["experiment"]["min_effect"] = 0.08
    with pytest.raises(HTTPException) as error:
        main.start_from_case(CaseStart.model_validate(payload), "same")
    assert error.value.status_code == 409
    main.start_from_case(CaseStart.model_validate(payload), "new")
    assert len(calls) == 2


def test_store_serializes_concurrent_claims(tmp_path):
    store = InvestigationStore(LocalStorage(tmp_path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(
            pool.map(
                lambda _: store.create_once("tf", {}, "test", "key", "digest"),
                range(16),
            )
        )
    assert len({inv_id for inv_id, _ in answers}) == 1
    assert sum(created for _, created in answers) == 1


def test_http_header_and_public_restriction(tmp_path, monkeypatch):
    store = InvestigationStore(LocalStorage(tmp_path / "store"))
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    payload = make_payload(tmp_path)
    # Reject invalid keys before any worker can run.
    with TestClient(main.app) as client:
        response = client.post(
            "/api/investigations/from-case",
            json=payload,
            headers={"Idempotency-Key": "has spaces"},
        )
        assert response.status_code == 422
        monkeypatch.setattr(
            main, "get_settings", lambda: SimpleNamespace(public_demo=True)
        )
        response = client.post(
            "/api/investigations/from-case",
            json=payload,
            headers={"Idempotency-Key": "valid"},
        )
        assert response.status_code == 403
    assert store.list() == []
