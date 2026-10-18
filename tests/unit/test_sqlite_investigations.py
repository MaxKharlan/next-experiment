import pytest
from sqlalchemy.exc import StatementError

from apps.api.store import WAITING, get_investigation_store
from core.config import Settings
from core.db.investigations import SQLiteInvestigations


def settings(tmp_path):
    return Settings(
        _env_file=None,
        investigation_backend="sqlite",
        storage_backend="local",
        artifacts_dir=str(tmp_path / "artifacts"),
        database_url=f"sqlite:///{tmp_path}/db/state.db",
    )


def test_snapshot_write_failure_keeps_previous_state(tmp_path):
    repo = SQLiteInvestigations(settings(tmp_path).database_url)
    repo.save({"id": "one", "status": "running"}, [])
    with pytest.raises(StatementError):
        repo.save({"id": "one", "status": "finished"}, [{"invalid": object()}])
    assert repo.load("one") == ({"id": "one", "status": "running"}, [])


def test_waiting_restart_and_one_answer(tmp_path):
    cfg = settings(tmp_path)
    store = get_investigation_store(cfg)
    inv_id = store.create("tf", {}, request_identity=("key", "digest"))
    store.emit(inv_id, WAITING, {"plan": {"cost": 1}})
    fresh = get_investigation_store(cfg)
    assert fresh.create_once("tf", {}, "", "key", "digest") == (inv_id, False)
    assert fresh.resume(inv_id, False) == {"plan": {"cost": 1}}
    with pytest.raises(ValueError, match="not waiting"):
        fresh.resume(inv_id, False)
    assert not list((tmp_path / "artifacts").rglob("meta.json"))
    with pytest.raises(ValueError, match="different request"):
        fresh.create_once("tf", {}, "", "key", "changed")


def test_interrupted_run_recovers_once_and_keeps_events(tmp_path):
    cfg = settings(tmp_path)
    store = get_investigation_store(cfg)
    inv_id = store.create("tf", {})
    store.emit(inv_id, "started", {})
    store.emit(inv_id, "usage", {"cost_usd": 0.1})
    fresh = get_investigation_store(cfg)
    assert fresh.list()[0]["status"] == "failed"
    again = get_investigation_store(cfg)
    events = again.snapshot(inv_id)["events"]
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert events[-1]["data"]["reason"] == "process_restart"
    assert events[1]["data"]["cost_usd"] == 0.1


def test_completed_report_snapshot_survives_new_connection(tmp_path):
    cfg = settings(tmp_path)
    store = get_investigation_store(cfg)
    inv_id = store.create("replay", {"name": "case"})
    store.emit(inv_id, "finished", {"result": "saved"})
    assert get_investigation_store(cfg).snapshot(inv_id) == store.snapshot(inv_id)
    with pytest.raises(KeyError):
        get_investigation_store(cfg).get("missing")


@pytest.mark.parametrize(
    "url", ["sqlite:///:memory:", "sqlite://", "postgresql://localhost/db"]
)
def test_reject_nonpersistent_or_non_sqlite_urls(url):
    with pytest.raises(ValueError, match="file-backed SQLite"):
        SQLiteInvestigations(url)
