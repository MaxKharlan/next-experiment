from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.store import InvestigationStore
from core.agent.context import InvestigationContext
from core.agent.schemas import Hypothesis
from core.connectors.storage import LocalStorage
from core.interventions.code_faults import (
    CODE_FAULTS,
    LocalCodePairRunner,
    investigate_code,
    make_code_pair,
)
from tests.unit.test_loop import FakeRouter


@pytest.fixture(params=["storage", "sqlite"])
def client(request, tmp_path, monkeypatch):
    from apps.api.store import get_investigation_store
    from core.config import Settings

    settings = Settings(
        _env_file=None,
        investigation_backend=request.param,
        database_url=f"sqlite:///{tmp_path}/state.db",
        artifacts_dir=str(tmp_path),
        public_demo=False,
    )
    monkeypatch.setattr(main, "store", get_investigation_store(settings))
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return TestClient(main.app)


def payload():
    return {
        "training_labels": {
            "reference": [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]}],
            "observed": [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, -100]}],
            "eos_token_id": 2,
        }
    }


def test_api_audit_is_persisted_streamed_and_downloadable(client):
    response = client.post("/api/investigations/label-audit", json=payload())
    assert response.status_code == 201
    data = response.json()
    assert data["audit"]["raw"]["eos"]["status"] == "removed"
    assert data["training_started"] is False
    inv_id = data["id"]
    report = client.get(f"/api/investigations/{inv_id}/report?download=true")
    assert report.json()["diagnosis_status"] == "inconclusive"
    assert report.json()["steps"] == [data["audit"]]
    assert report.json()["execution_status"] == "finished"
    assert "input_ids" not in report.text
    assert "attachment" in report.headers["content-disposition"]
    events = client.get(f"/api/investigations/{inv_id}/events").text
    assert "training_label_checks" in events
    assert "event: finished" in events


@pytest.mark.parametrize("bad", ["tokens", "bool", "path", "oversize"])
def test_api_rejects_invalid_audits_without_creating_state(client, bad):
    body = payload()
    if bad == "tokens":
        body["training_labels"]["observed"][0]["input_ids"][-1] = 9
    elif bad == "bool":
        body["training_labels"]["eos_token_id"] = True
    elif bad == "path":
        body["path"] = "/private/anything"
    else:
        body["title"] = "x" * 121
    assert client.post("/api/investigations/label-audit", json=body).status_code == 422
    assert main.store.list() == []


def test_audit_public_demo_protection(client, monkeypatch):
    inv_id = client.post("/api/investigations/label-audit", json=payload()).json()["id"]
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(public_demo=True))
    assert (
        client.post("/api/investigations/label-audit", json=payload()).status_code
        == 403
    )
    for route in ("report", "events"):
        assert client.get(f"/api/investigations/{inv_id}/{route}").status_code == 403
    assert client.get("/api/investigations").json() == []


def context(fault):
    return InvestigationContext(
        "test",
        "B",
        "local",
        {"code_fault": fault, "hyperparameters": {"gradient_accumulation_steps": 3}},
        budget_left_usd=1.0,
    )


def hypothesis(fault):
    return Hypothesis(
        id="h1", fault_class=fault, prior=0.8, location="trainer", rationale="Test"
    )


@pytest.mark.parametrize("fault_class,fault", CODE_FAULTS.items())
def test_repairs_run_both_arms_with_matched_inputs(tmp_path, fault_class, fault):
    ctx = context(fault)
    before = deepcopy(ctx.job_config)
    target, control = make_code_pair(hypothesis(fault_class), ctx)
    calls = []

    def execute(hp, switch, seed, out):
        calls.append((hp, switch, seed, out))
        return 0.9 if switch is None else 0.2

    runner = LocalCodePairRunner(
        before["hyperparameters"], [0, 1], tmp_path / "pair", execute
    )
    runner(target, control)
    assert [c[1] for c in calls] == [None, None, fault, fault]
    assert [c[2] for c in calls] == [0, 1, 0, 1]
    assert all(c[0] == before["hyperparameters"] for c in calls)
    assert target.metrics == [0.9, 0.9] and control.metrics == [0.2, 0.2]
    assert ctx.job_config == before


def test_agent_uses_local_code_factory(tmp_path):
    ctx = context("eos_as_pad")
    runner = LocalCodePairRunner(
        ctx.job_config["hyperparameters"],
        [0, 1, 2],
        tmp_path / "agent",
        lambda hp, fault, seed, out: 1.0 if fault is None else 0.0,
    )
    result = investigate_code(
        ctx,
        {"training_labels": payload()["training_labels"]},
        FakeRouter([]),
        runner,
        initial_hypotheses=[hypothesis("pad_eos_masking")],
        probes=[],
        explain=False,
        noise_sd=None,
    )
    assert result.status == "confirmed"
    assert result.steps[0]["kind"] == "training_label_checks"
    intervention = next(s for s in result.steps if s["kind"] == "intervention")
    assert intervention["target"]["patch"] == {"code_fault": None}


def test_wrong_mode_and_mismatched_switch_rejected():
    ctx = context("double_shift")
    with pytest.raises(ValueError, match="does not match"):
        make_code_pair(hypothesis("pad_eos_masking"), ctx)
    ctx.mode = "A"
    with pytest.raises(ValueError, match="mode B"):
        make_code_pair(hypothesis("double_label_shift"), ctx)


def test_failure_does_not_publish_partial_metrics(tmp_path):
    target, control = make_code_pair(
        hypothesis("pad_eos_masking"), context("eos_as_pad")
    )

    def execute(hp, fault, seed, out):
        return float("nan") if fault else 1.0

    runner = LocalCodePairRunner({}, [0, 1], tmp_path / "pair", execute)
    with pytest.raises(ValueError, match="finite"):
        runner(target, control)
    assert not target.metrics and not control.metrics
    with pytest.raises(FileExistsError):
        runner(target, control)


def test_restart_keeps_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
    monkeypatch.setattr(
        main, "get_settings", lambda: SimpleNamespace(public_demo=False)
    )
    client = TestClient(main.app)
    inv_id = client.post("/api/investigations/label-audit", json=payload()).json()["id"]
    before = client.get(f"/api/investigations/{inv_id}/report").json()
    monkeypatch.setattr(main, "store", InvestigationStore(LocalStorage(tmp_path)))
    assert client.get(f"/api/investigations/{inv_id}/report").json() == before


def test_lora_executor_separates_training_validation_and_check(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    import json

    from core.interventions.code_faults import LocalLoRAExecutor
    from jobs.eval_local import exact_match
    from jobs.kaggle import train_lora

    paths = [tmp_path / f"{name}.jsonl" for name in ("train", "valid", "check")]
    for i, path in enumerate(paths):
        path.write_text(
            json.dumps({"messages": [{"role": "user", "content": str(i)}]}) + "\n"
        )
    calls = []
    monkeypatch.setattr(
        train_lora, "train", lambda *args, **kw: calls.append((args, kw))
    )
    monkeypatch.setattr(exact_match, "evaluate_adapter", lambda *args, **kw: 0.75)
    executor = LocalLoRAExecutor(*paths, "local-model")
    out = tmp_path / "run"
    out.mkdir()
    assert executor({"n_epochs": 1}, "eos_as_pad", 7, out) == 0.75
    assert calls[0][0][:2] == tuple(paths[:2])
    assert calls[0][1]["code_fault"] == "eos_as_pad"
    assert "code_fault" not in calls[0][1]["hp"]
    assert calls[0][1]["generate"] is False
    assert json.loads((out / "check-score.json").read_text())["exact_match"] == 0.75
    paths[2].write_text(paths[0].read_text())
    with pytest.raises(ValueError, match="changed"):
        executor({}, None, 0, out)
    with pytest.raises(ValueError, match="disjoint"):
        LocalLoRAExecutor(*paths, "local-model")


def test_cpu_experiment_repeats_predictions_and_has_disjoint_data(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    import json

    import torch

    from scripts.mode_b_quality import corpus, execute_training

    torch.set_num_threads(1)
    train, check = corpus(1729, 12), corpus(2718, 6)
    assert not {tuple(r["input_ids"]) for r in train} & {
        tuple(r["input_ids"]) for r in check
    }
    hp = {"learning_rate": 0.01, "steps": 2, "gradient_accumulation_steps": 2}
    scores = []
    records = []
    for name in ("first", "second"):
        out = tmp_path / name
        out.mkdir()
        scores.append(execute_training(train, check, hp, "eos_as_pad", 4, out))
        records.append(json.loads((out / "run.json").read_text()))
    assert scores[0] == scores[1]
    assert records[0] == records[1]
    assert records[0]["label_audit"]["eos"]["removed_eos_targets"] == 12
    assert scores[0] == sum(p["exact_match"] for p in records[0]["predictions"]) / 6
