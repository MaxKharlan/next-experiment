"""Exercise the live API pipeline with deterministic provider and training doubles."""

import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from apps.api import main, runners
from apps.api.store import InvestigationStore
from core import config
from core.agent.schemas import Hypothesis, HypothesisList, Verdict
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.answer_check import AnswerDrop, digest


def make_payload(tmp_path):
    datasets = {}
    for role in ("train", "validation", "check", "test"):
        path = tmp_path / f"{role}.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"{role} question {i}"},
                    {
                        "role": "assistant",
                        "content": (
                            "card_a"
                            if role in {"train", "validation"} and i >= 8
                            else "card_arrival"
                        ),
                    },
                ]
            }
            for i in range(10)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        datasets[role] = {"ref": str(path), "examples": 10, "sha256": digest(path)}
    return {
        "case": {
            "task": "banking77",
            "symptom": "The model emits incomplete labels.",
            "mode": "A",
            "job": {
                "id": "ftjob-e2e",
                "checkpoints": [],
                "model": "Qwen/Qwen3-0.6B",
                "hyperparameters": {
                    "learning_rate": 0.0002,
                    "n_epochs": 3,
                    "lora_r": 16,
                    "lora_alpha": 16,
                    "batch_size": 8,
                    "context_length": 8192,
                },
            },
            "datasets": datasets,
            "evaluator": {"name": "intent_label", "metric": "exact_match"},
            "budget_usd": 0.75,
        },
        "experiment": {
            "name": "e2e-only",
            "noise_sd": 0.019,
            "min_effect": 0.06,
            "tokens_per_example": 57,
            "seeds": [0, 1],
        },
    }


class SavedProvider:
    def __init__(self, case):
        self.case = case
        self.jobs = []
        self.schemas = []

    def job(self, job_id):
        self.jobs.append(job_id)
        assert job_id == self.case["job"]["id"]
        return SimpleNamespace(status="succeeded", **self.case["job"])

    def model(self, role):
        return f"test-{role}"

    def structured(self, model, messages, schema, **kwargs):
        self.schemas.append(schema)
        if schema is HypothesisList:
            assert "answer_cut_evidence" in messages[-1]["content"]
            result = HypothesisList(
                hypotheses=[
                    Hypothesis(
                        id="h1",
                        fault_class="answer_truncation",
                        location="train",
                        prior=0.9,
                        rationale="Some labels are prefixes of complete labels.",
                    )
                ]
            )
        else:
            assert schema is Verdict
            # Deliberately disagree: the causal decision must come from code.
            result = Verdict(
                hypothesis_id="wrong",
                status="rejected",
                evidence=[],
                patch="",
                summary="Saved model response.",
            )
        return result, []


@pytest.mark.parametrize("confirmed", [True, False], ids=["confirmed", "small-effect"])
@pytest.mark.parametrize(
    "retrain_approved", [True, False], ids=["approve-retrain", "decline-retrain"]
)
def test_answer_cut_investigation_survives_restart(
    tmp_path, monkeypatch, confirmed, retrain_approved
):
    payload = make_payload(tmp_path)
    provider = SavedProvider(payload["case"])
    pairs = []
    network_attempts = []

    def forbid_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("Network access is forbidden in this test")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXT_EXPERIMENT_WEIGHTS", "0")
    settings = SimpleNamespace(public_demo=False, tavily_api_key=None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(TokenFactory, "from_settings", lambda: provider)
    storage = LocalStorage(tmp_path / "store")
    monkeypatch.setattr(main, "store", InvestigationStore(storage))

    from core.interventions import answer_check

    original_investigate = answer_check.investigate_with_answers

    def inspect_context(ctx, *args, **kwargs):
        assert set(ctx.files) == {"train", "validation"}
        return original_investigate(ctx, *args, **kwargs)

    monkeypatch.setattr(answer_check, "investigate_with_answers", inspect_context)

    def pair_runner(backend, inv_id, tf, storage, job_hp, protocol):
        assert backend == "tf" and tf is provider
        assert protocol["job_id"] == payload["case"]["job"]["id"]
        assert job_hp == payload["case"]["job"]["hyperparameters"]

        def run(target, control):
            assert isinstance(target, AnswerDrop) and isinstance(control, AnswerDrop)
            assert target.drop_lines == [9, 10]
            assert len(control.drop_lines) == 2
            assert set(target.drop_lines).isdisjoint(control.drop_lines)
            train = Path(protocol["datasets"]["train"]["path"])
            assert len(target.rows(train)) == len(control.rows(train)) == 8
            pairs.append((target, control))
            target.metrics = [0.50, 0.52] if confirmed else [0.11, 0.13]
            control.metrics = [0.10, 0.12]

        run.supports_answer_drop = True
        return run

    monkeypatch.setattr(runners, "_pair_runner", pair_runner)
    from core.interventions.retrain import Retrainer
    from scripts import retrain_tf
    from tests.unit.test_tf_runner import FakeTF

    retrain_provider, scored = FakeTF(), []

    def make_retrainer(tf, storage, investigation_id, **kwargs):
        def score_adapter(adapter, data):
            scored.append(str(data))
            return 0.50 if "ftjob-0" in adapter.name else 0.10

        def score_base(model, data, template):
            scored.append(str(data))
            return 0.0

        return Retrainer(
            retrain_provider,
            storage,
            kwargs["train"],
            kwargs["valid"],
            kwargs["heldout"],
            score_adapter,
            score_base,
            f"investigations/{investigation_id}/retrain",
            sleep=lambda _: None,
        )

    monkeypatch.setattr(retrain_tf, "make_retrainer", make_retrainer)
    with TestClient(main.app) as client:
        response = client.post("/api/investigations/from-case", json=payload)
        assert response.status_code == 202, response.text
        inv_id = response.json()["id"]
        root = f"/api/investigations/{inv_id}"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get(root).json()["status"]
            if status in {
                "finished",
                "failed",
                "awaiting_approval",
                "awaiting_paid_step",
            }:
                break
            time.sleep(0.01)
        assert status == "awaiting_paid_step"
        response = client.post(
            f"{root}/paid-step",
            json={
                "approve": True,
                "proposal_seq": main.store.events_since(inv_id, 0)[-1]["seq"],
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get(root).json()["status"]
            if status in {"finished", "failed", "awaiting_approval"}:
                break
            time.sleep(0.01)
        if confirmed:
            assert status == "awaiting_approval"
            waiting_report = client.get(f"{root}/report").json()
            assert waiting_report["partial"] is True
            assert waiting_report["retrain_plan"]["data_patch"]["line_numbers"] == [
                9,
                10,
            ]
            assert not retrain_provider.created and not scored
            # The full retrain proposal survives loss of the original in-memory store.
            monkeypatch.setattr(main, "store", InvestigationStore(storage))
            response = client.post(
                f"{root}/retrain",
                json={
                    "approve": retrain_approved,
                    "proposal_seq": main.store.events_since(inv_id, 0)[-1]["seq"],
                },
            )
            assert response.status_code == 200, response.text
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = client.get(root).json()["status"]
                if status in {"finished", "failed"}:
                    break
                time.sleep(0.01)
        response = client.get(f"{root}/report")
        assert response.status_code == 200
        report = response.json()
        assert status == "finished", report
        assert report["partial"] is False
        assert report["diagnosis_status"] == (
            "confirmed" if confirmed else "inconclusive"
        )
        assert report["input_case"]["job"] == payload["case"]["job"]
        assert report["protocol"]["job_id"] == "ftjob-e2e"
        assert (report["retrain_plan"] is not None) == confirmed
        assert report["error"] is None
        events = report["events"]
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert events[0]["type"] == "started"
        assert events[-1]["type"] == "finished"
        assert all(
            e["type"] in {"step", "awaiting_paid_step", "awaiting_approval", "resumed"}
            for e in events[1:-1]
        )
        assert [e["type"] for e in events].count("awaiting_paid_step") == 1
        assert [e["type"] for e in events].count("resumed") == (2 if confirmed else 1)
        steps = report["steps"]
        assert steps[0]["probe"] == "answer_cut"
        assert steps[0]["causal_check_available"] is True
        kinds = [s["kind"] for s in steps]
        assert kinds.index("hypotheses") < kinds.index("intervention")
        checks = [s for s in steps if s["kind"] == "intervention"]
        assert len(checks) == len(pairs) == 1
        if confirmed:
            assert "retrain_proposal" in kinds
            if retrain_approved:
                assert kinds[-1] == "comparison"
                assert len(retrain_provider.created) == 1
                fixed_rows = Path(retrain_provider.uploads[0]).read_text().splitlines()
                assert len(fixed_rows) == 8
                assert all(
                    json.loads(r)["messages"][-1]["content"] == "card_arrival"
                    for r in fixed_rows
                )
                assert scored == [payload["case"]["datasets"]["test"]["ref"]] * 3
                assert (
                    report["comparison"]["provenance"]["dataset_sha256"]["test"]
                    == payload["case"]["datasets"]["test"]["sha256"]
                )
                operations = report["summary"]["spending"]["operations"]
                assert [op["purpose"] for op in operations] == [
                    "causal_check",
                    "retrain",
                ]
                assert all(op["status"] == "settled" for op in operations)
                assert report["summary"]["retrain_estimated_cost_usd"] > 0
            else:
                assert kinds[-1] == "retrain_declined"
                assert not retrain_provider.created and not scored
            assert report["verdict"]["status"] == "confirmed"
            assert "probe:answer_cut" in report["verdict"]["evidence"]
            assert "intervention:h1" in report["verdict"]["evidence"]
            assert "drop_training_lines" in report["verdict"]["patch"]
        else:
            assert checks[0]["result"]["status"] != "confirmed"
            assert report["verdict"] is None
            assert "verdict" not in kinds

        stream = client.get(f"{root}/events")
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        streamed = []
        for block in stream.text.replace("\r\n", "\n").strip().split("\n\n"):
            fields = dict(line.split(": ", 1) for line in block.splitlines())
            streamed.append((fields["event"], json.loads(fields["data"])))
        assert streamed == [(e["type"], e["data"]) for e in events]

    monkeypatch.setattr(main, "store", InvestigationStore(storage))
    with TestClient(main.app) as restarted:
        downloaded = restarted.get(f"{root}/report?download=true")
        assert downloaded.status_code == 200
        assert downloaded.json() == report
        assert downloaded.headers["content-disposition"].startswith("attachment;")
        assert restarted.get(root).json()["status"] == "finished"
    assert provider.jobs == ["ftjob-e2e"]
    assert provider.schemas == (
        [HypothesisList, Verdict] if confirmed else [HypothesisList]
    )
    assert not network_attempts
    for entry in payload["case"]["datasets"].values():
        assert digest(entry["ref"]) == entry["sha256"]
