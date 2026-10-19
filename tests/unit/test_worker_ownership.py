import multiprocessing
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from apps.api.store import InvestigationStore
from core.config import Settings
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.db.investigations import OwnershipLost, SQLiteInvestigations
from core.db.single_worker import hold_single_worker
from core.db.spending import SpendingLedger


def make_store(root, **options):
    return InvestigationStore(
        LocalStorage(root),
        SQLiteInvestigations(f"sqlite:///{root}/state.db", **options),
    )


def process_race(root, action, inv_id, barrier, results):
    store = make_store(root)
    if inv_id:
        store.get(inv_id)  # Each worker first observes the same waiting proposal.
    barrier.wait(timeout=15)
    try:
        if action == "create":
            inv_id, created = store.create_once("tf", {}, "case", "key", "digest")
            if created:
                store.worker(inv_id).run_owned(
                    inv_id, record_payment, (), {"root": root}
                )
            results.put((inv_id, created))
        else:
            resume = store.resume if action == "retrain" else store.resume_paid
            resume(inv_id, True, 0)
            store.worker(inv_id).run_owned(inv_id, record_payment, (), {"root": root})
            results.put(True)
    except ValueError:
        results.put(False)


def record_payment(store, inv_id, *, root):
    store.assert_owner(inv_id)
    ledger = SpendingLedger(Path(root) / "spending.db", 1)
    ledger.reserve("paid-step", "training", 0.1)
    # Durable stand-in for a provider accepting a billable job.
    with (Path(root) / "provider-calls.txt").open("a") as calls:
        calls.write("accepted\n")
    ledger.settle("paid-step", 0.1)
    store.emit(inv_id, "finished", {"job_id": "fake-job"})


@pytest.mark.parametrize("action", ["create", "retrain", "paid-step"])
def test_distinct_processes_dispatch_only_one_paid_operation(tmp_path, action):
    store = make_store(tmp_path)
    inv_id = None
    if action != "create":
        inv_id = store.create("tf", {})
        kind = "awaiting_approval" if action == "retrain" else "awaiting_paid_step"
        store.emit(inv_id, kind, {"proposal": {"kind": "causal_check"}})
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(3), ctx.Queue()
    processes = [
        ctx.Process(
            target=process_race, args=(tmp_path, action, inv_id, barrier, results)
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    try:
        answers = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    if action == "create":
        assert len({inv for inv, _ in answers}) == 1
        assert sum(created for _, created in answers) == 1
        inv_id = answers[0][0]
    else:
        assert answers.count(True) == 1
    assert (tmp_path / "provider-calls.txt").read_text() == "accepted\n"
    snapshot = store.snapshot(inv_id)
    assert snapshot["meta"]["status"] == "finished"
    assert [e["seq"] for e in snapshot["events"]] == list(
        range(len(snapshot["events"]))
    )
    assert (
        SpendingLedger(tmp_path / "spending.db", 1).snapshot()["committed_usd"] == 0.1
    )


def test_observers_refresh_events_without_failing_the_live_owner(tmp_path):
    first, other = make_store(tmp_path), make_store(tmp_path)
    inv = first.create("tf", {})
    assert other.get(inv)["status"] == "queued"
    first.emit(inv, "started", {})
    assert other.get(inv)["status"] == "running"
    first.emit(inv, "step", {"job_id": "job"})
    assert len(other.events_since(inv, 0)) == 2
    with pytest.raises(OwnershipLost):
        other.emit(inv, "failed", {})
    first.emit(inv, "finished", {})
    assert other.snapshot(inv) == first.snapshot(inv)


def test_expired_owner_cannot_restart_renew_or_write(tmp_path):
    now = [100.0]
    first = make_store(tmp_path, clock=lambda: now[0], lease_seconds=10)
    other = make_store(tmp_path, clock=lambda: now[0], lease_seconds=10)
    inv = first.create("tf", {})
    first.emit(inv, "started", {})
    ledger = SpendingLedger(tmp_path / "spending.db", 1)
    ledger.reserve("unknown", "training", 0.1)
    now[0] = 111
    assert other.get(inv)["status"] == "failed"
    snapshot = other.snapshot(inv)
    assert snapshot["events"][-1]["data"]["reason"] == "lease_expired"
    assert first.snapshot(inv) == snapshot
    for action in (
        lambda: first.repository.renew(inv, first.owners[inv]),
        lambda: first.emit(inv, "finished", {}),
        lambda: other.resume(inv, True),
        lambda: other.resume_paid(inv, True),
        lambda: first.assert_owner(inv),
    ):
        with pytest.raises(ValueError):
            action()
    assert ledger.snapshot()["operations"][0]["status"] == "pending"
    assert other.create_once("tf", {}, "", "same", "digest")[1]


def test_old_execution_and_old_approval_cannot_authorize_next_step(tmp_path):
    store = make_store(tmp_path)
    inv = store.create("tf", {})
    old = store.worker(inv)
    old.emit(inv, "awaiting_paid_step", {"proposal": {"kind": "first"}})
    store.resume_paid(inv, True, 0)
    active = store.worker(inv)
    active.emit(inv, "awaiting_paid_step", {"proposal": {"kind": "second"}})
    with pytest.raises(ValueError, match="different proposal"):
        store.resume_paid(inv, True, 0)
    store.resume_paid(inv, True, 2)
    for worker in (old, active):
        with pytest.raises(OwnershipLost):
            worker.emit(inv, "failed", {})
        with pytest.raises(OwnershipLost):
            worker.assert_owner(inv)
    assert store.get(inv)["status"] == "running"


def test_one_claim_cannot_be_dispatched_twice(tmp_path):
    store = make_store(tmp_path)
    inv = store.create("tf", {})
    worker = store.worker(inv)
    worker.run_owned(inv, record_payment, (), {"root": tmp_path})
    worker.run_owned(inv, record_payment, (), {"root": tmp_path})
    assert (tmp_path / "provider-calls.txt").read_text() == "accepted\n"


def test_heartbeat_keeps_a_quiet_worker_alive(tmp_path):
    store = make_store(tmp_path, lease_seconds=0.6)
    observer = make_store(tmp_path, lease_seconds=0.6)
    inv = store.create("tf", {})
    entered, release = threading.Event(), threading.Event()

    def quiet(worker, inv_id):
        worker.emit(inv_id, "started", {})
        entered.set()
        assert release.wait(timeout=5)
        worker.emit(inv_id, "finished", {})

    worker = store.worker(inv)
    thread = threading.Thread(target=worker.run_owned, args=(inv, quiet, (), {}))
    thread.start()
    try:
        assert entered.wait(timeout=2)
        time.sleep(1.0)
        assert observer.get(inv)["status"] == "running"
    finally:
        release.set()
        thread.join(timeout=3)
    assert observer.get(inv)["status"] == "finished"


def test_lost_lease_blocks_provider_inference_training_and_search(tmp_path):
    now = [100]
    store = make_store(tmp_path, clock=lambda: now[0], lease_seconds=10)
    inv = store.create("tf", {})
    worker = store.worker(inv)
    now[0] = 111
    calls = []
    client = SimpleNamespace()
    tf = TokenFactory(client, Settings(_env_file=None))
    tf.before_paid = lambda: worker.assert_owner(inv)
    tavily = Tavily(
        "test",
        LocalStorage(tmp_path),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: calls.append(request))
        ),
    )
    tavily.before_paid = tf.before_paid
    for action in (
        lambda: tf.chat(tf.model("super"), []),
        lambda: tf.create_job("model", "file"),
        lambda: tavily.search("query", investigation_id=inv),
    ):
        with pytest.raises(OwnershipLost):
            action()
    assert not calls and not tavily.ledger.snapshot()["operations"]


def hold_legacy_lock(root, result):
    try:
        hold_single_worker(root)
        result.put("acquired")
    except ValueError:
        result.put("blocked")


def test_legacy_storage_rejects_another_process(tmp_path):
    hold_single_worker(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=hold_legacy_lock, args=(tmp_path, result))
    process.start()
    try:
        assert result.get(timeout=10) == "blocked"
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def interrupted_worker(root, ready, release):
    store = make_store(root)
    inv, _ = store.create_once("tf", {}, "", "request", "digest")
    store.emit(inv, "started", {})
    ledger = SpendingLedger(Path(root) / "spending.db", 1)
    ledger.reserve("unknown", "training", 0.1)
    ready.put(inv)  # Simulate a provider request whose response has not arrived.
    release.wait(timeout=30)


def test_killed_worker_retains_hold_and_request_identity(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Queue(), ctx.Event()
    process = ctx.Process(target=interrupted_worker, args=(tmp_path, ready, release))
    process.start()
    try:
        inv = ready.get(timeout=15)
    finally:
        process.kill()
        process.join(timeout=5)
    store = make_store(tmp_path, clock=lambda: time.time() + 61)
    assert store.get(inv)["status"] == "failed"
    assert store.create_once("tf", {}, "", "request", "digest") == (inv, False)
    with pytest.raises(ValueError):
        store.resume_paid(inv, True)
    rows = SpendingLedger(tmp_path / "spending.db", 1).snapshot()["operations"]
    assert len(rows) == 1 and rows[0]["status"] == "pending"


def test_lease_loss_between_reservation_and_send_keeps_hold(tmp_path):
    now = [100]
    store = make_store(tmp_path, clock=lambda: now[0], lease_seconds=10)
    inv = store.create("tf", {})
    worker = store.worker(inv)
    ledger = SpendingLedger(tmp_path / "spending.db", 1)
    original = ledger.reserve

    def reserve(*args):
        original(*args)
        now[0] = 111

    ledger.reserve = reserve
    client = SimpleNamespace(with_options=lambda **kw: SimpleNamespace())
    tf = TokenFactory(client, Settings(_env_file=None), spending=ledger)
    tf.before_paid = lambda: worker.assert_owner(inv)
    with pytest.raises(OwnershipLost):
        tf.chat(tf.model("super"), [])
    assert ledger.snapshot()["operations"][0]["status"] == "pending"


def test_http_requires_identity_and_binds_approval_to_proposal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from apps.api import main

    store = make_store(tmp_path)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    launched = []
    monkeypatch.setattr(main, "_launch", lambda *args, **kwargs: launched.append(args))
    client = TestClient(main.app)
    assert client.post("/api/investigations", json={"backend": "tf"}).status_code == 422
    headers = {"Idempotency-Key": "start"}
    first = client.post("/api/investigations", json={"backend": "tf"}, headers=headers)
    inv = first.json()["id"]
    assert client.post(
        "/api/investigations", json={"backend": "tf"}, headers=headers
    ).json() == {"id": inv, "reused": True}
    assert len(launched) == 1
    store.emit(inv, "awaiting_paid_step", {"proposal": {"kind": "first"}})
    route = f"/api/investigations/{inv}/paid-step"
    assert client.post(route, json={"approve": True}).status_code == 409
    assert (
        client.post(route, json={"approve": True, "proposal_seq": 0}).status_code == 200
    )
    store.emit(inv, "awaiting_paid_step", {"proposal": {"kind": "second"}})
    assert (
        client.post(route, json={"approve": True, "proposal_seq": 0}).status_code == 409
    )
    assert len(launched) == 2
    assert store.get(inv)["status"] == "awaiting_paid_step"
