import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from apps.api.case_input import CaseStart, prepare_case
from core.agent.context import InvestigationContext
from core.connectors.storage import LocalStorage
from core.db.spending import BudgetExceeded, SpendingLedger
from core.interventions.answer_check import bind_pair, digest
from core.interventions.data_retrain import plan_data_retrain, retrain_data
from core.interventions.retrain import Retrainer, RetrainPlan
from core.probes.level0.answers import AnswerCut
from tests.e2e.test_investigation_api import make_payload
from tests.unit.test_tf_runner import FakeTF


def setup(tmp_path):
    payload = make_payload(tmp_path)
    protocol = prepare_case(CaseStart.model_validate(payload))["protocol"]
    ctx = InvestigationContext(
        "inv",
        "A",
        protocol["job_id"],
        {},
        files={"train": protocol["datasets"]["train"]["path"]},
    )
    raw = json.loads(AnswerCut().run(ctx).summary)
    target, _ = bind_pair("h1", raw, digest(ctx.files["train"]))
    plan = plan_data_retrain(target, protocol)
    tf = FakeTF()
    seen = []

    def score_adapter(adapter, path):
        seen.append(str(path))
        return 0.5 if "ftjob-0" in adapter.name else 0.1

    def score_base(model, path, template):
        seen.append(str(path))
        return 0.0

    retrainer = Retrainer(
        tf,
        LocalStorage(tmp_path / "artifacts"),
        Path(ctx.files["train"]),
        Path(protocol["datasets"]["validation"]["path"]),
        Path(protocol["datasets"]["test"]["path"]),
        score_adapter,
        score_base,
        "investigations/inv/retrain",
        sleep=lambda _: None,
    )
    ledger = SpendingLedger(tmp_path / "budget.sqlite", protocol["budget_usd"])
    return protocol, plan, retrainer, ledger, seen


def test_approved_rows_reach_training_and_clean_test_provenance(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    assert plan.train_examples == 8
    assert plan.data_patch["line_numbers"] == [9, 10]
    restored = RetrainPlan(**json.loads(json.dumps(asdict(plan))))
    steps = []
    result = retrain_data(retrainer, restored, protocol, ledger, steps.append)
    tf = retrainer.tf
    rows = [json.loads(line) for line in Path(tf.uploads[0]).read_text().splitlines()]
    assert len(rows) == 8
    assert {r["messages"][-1]["content"] for r in rows} == {"card_arrival"}
    assert tf.uploads[1] == protocol["datasets"]["validation"]["path"]
    assert len(tf.created) == 1
    assert (
        tf.created["ftjob-0"]["hyperparameters"]["learning_rate"]
        == protocol["hyperparameters"]["learning_rate"]
    )
    assert seen == [protocol["datasets"]["test"]["path"]] * 3
    assert result["provenance"]["dataset_sha256"]["test"] == digest(seen[0])
    assert result["provenance"]["fixed_train_sha256"] == digest(tf.uploads[0])
    assert ledger.snapshot()["operations"][0]["status"] == "settled"
    assert [s["kind"] for s in steps] == [
        "spending_reserved",
        "retrain",
        "spending_settled",
    ]
    for entry in protocol["datasets"].values():
        assert digest(entry["path"]) == entry["sha256"]
    assert retrain_data(retrainer, plan, protocol, ledger) == result
    assert len(tf.created) == 1 and len(tf.uploads) == 2 and len(seen) == 3


@pytest.mark.parametrize("role", ["train", "validation", "check", "test"])
def test_changed_split_blocks_all_provider_calls(tmp_path, role):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    path = Path(protocol["datasets"][role]["path"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        retrain_data(retrainer, plan, protocol, ledger)
    assert not retrainer.tf.uploads and not seen
    assert ledger.snapshot()["operations"] == []


def test_changed_plan_and_wrong_evaluation_split_are_rejected(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    with pytest.raises(ValueError, match="frozen estimate"):
        retrain_data(retrainer, replace(plan, cost_usd=0), protocol, ledger)
    with pytest.raises(ValueError, match="frozen final test"):
        retrain_data(
            replace(retrainer, heldout_path=retrainer.valid_path),
            plan,
            protocol,
            ledger,
        )
    with pytest.raises(ValueError, match="provenance-bound"):
        retrainer(plan)
    assert not retrainer.tf.uploads and not seen


def test_budget_failure_happens_before_upload(tmp_path):
    protocol, plan, retrainer, _, seen = setup(tmp_path)
    ledger = SpendingLedger(tmp_path / "tiny.sqlite", plan.cost_usd / 2)
    with pytest.raises(BudgetExceeded):
        retrain_data(retrainer, plan, protocol, ledger)
    assert not retrainer.tf.uploads and not seen


def test_ambiguous_submission_keeps_reservation_and_cannot_resubmit(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    attempts = []

    def lost_response(*args, **kwargs):
        attempts.append(True)
        raise TimeoutError("Provider may have accepted the job")

    retrainer.tf.create_job = lost_response
    with pytest.raises(TimeoutError):
        retrain_data(retrainer, plan, protocol, ledger)
    assert ledger.snapshot()["operations"][0]["status"] == "pending"
    with pytest.raises(ValueError, match="reconcile"):
        retrain_data(retrainer, plan, protocol, ledger)
    assert len(attempts) == 1 and not seen


def test_changed_test_during_training_blocks_scoring(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    original = retrainer.tf.job

    def poll(job_id):
        job = original(job_id)
        if job.status == "succeeded":
            retrainer.heldout_path.write_text("changed after training started")
        return job

    retrainer.tf.job = poll
    with pytest.raises(ValueError, match="changed"):
        retrain_data(retrainer, plan, protocol, ledger)
    assert not seen
    assert ledger.snapshot()["operations"][0]["status"] == "pending"
