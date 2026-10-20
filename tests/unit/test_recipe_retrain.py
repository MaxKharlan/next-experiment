from dataclasses import replace
from pathlib import Path

import pytest

from core.db.spending import BudgetExceeded, SpendingLedger
from core.interventions.pairs import Intervention
from core.interventions.recipe_retrain import retrain_recipe
from core.interventions.retrain import plan_retrain
from tests.unit.test_data_retrain import setup as data_setup


def setup(tmp_path):
    protocol, _, retrainer, ledger, seen = data_setup(tmp_path)
    plan = plan_retrain(
        Intervention(
            "h-lr", "target", {"learning_rate": 0.0001}, "raise learning rate"
        ),
        protocol["hyperparameters"],
        protocol["model"],
        protocol["datasets"]["train"]["examples"],
        protocol["tokens_per_example"],
        protocol["job_id"],
    )
    return protocol, plan, retrainer, ledger, seen


def test_recipe_reserves_and_reuses_completed_result(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    create = retrainer.tf.create_job

    def guarded(*args, **kwargs):
        assert ledger.snapshot()["operations"][0]["status"] == "pending"
        return create(*args, **kwargs)

    retrainer.tf.create_job = guarded
    result = retrain_recipe(retrainer, plan, protocol, ledger)
    assert (
        result["provenance"]["dataset_sha256"]["test"]
        == protocol["datasets"]["test"]["sha256"]
    )
    assert retrainer.tf.created["ftjob-0"]["hyperparameters"]["learning_rate"] == 0.0001
    assert ledger.snapshot()["committed_usd"] == plan.cost_usd
    assert retrain_recipe(retrainer, plan, protocol, ledger) == result
    assert len(retrainer.tf.created) == 1 and len(seen) == 3


@pytest.mark.parametrize("role", ["train", "validation", "check", "test"])
def test_changed_data_blocks_recipe_before_payment(tmp_path, role):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    assert not seen
    Path(protocol["datasets"][role]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        retrain_recipe(retrainer, plan, protocol, ledger)
    assert not retrainer.tf.uploads and not ledger.snapshot()["operations"]


def test_recipe_budget_and_changed_plan_fail_before_upload(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    with pytest.raises(ValueError, match="frozen estimate"):
        retrain_recipe(retrainer, replace(plan, cost_usd=0), protocol, ledger)
    tiny = SpendingLedger(tmp_path / "tiny.db", plan.cost_usd / 2)
    with pytest.raises(BudgetExceeded):
        retrain_recipe(retrainer, plan, protocol, tiny)
    assert not retrainer.tf.uploads and not seen


def test_unknown_recipe_submission_keeps_hold_and_blocks_duplicate(tmp_path):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    calls = []

    def lost(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("unknown provider outcome")

    retrainer.tf.create_job = lost
    with pytest.raises(TimeoutError):
        retrain_recipe(retrainer, plan, protocol, ledger)
    with pytest.raises(ValueError, match="reconcile"):
        retrain_recipe(retrainer, plan, protocol, ledger)
    assert len(calls) == 1 and ledger.snapshot()["operations"][0]["status"] == "pending"
    # Also protect the underlying standalone retrainer from a duplicate attempt.
    with pytest.raises(ValueError, match="reconcile"):
        retrainer(plan)
    assert len(calls) == 1 and not seen


def test_recipe_result_recovers_after_settlement_crash(tmp_path, monkeypatch):
    protocol, plan, retrainer, ledger, seen = setup(tmp_path)
    original = ledger.settle

    def lost(*args):
        original(*args)
        raise RuntimeError("crash after commit")

    monkeypatch.setattr(ledger, "settle", lost)
    with pytest.raises(RuntimeError, match="crash"):
        retrain_recipe(retrainer, plan, protocol, ledger)
    monkeypatch.setattr(ledger, "settle", original)
    retrain_recipe(retrainer, plan, protocol, ledger)
    assert len(retrainer.tf.created) == 1 and len(seen) == 3
    assert ledger.snapshot()["committed_usd"] == plan.cost_usd
