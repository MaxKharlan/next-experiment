"""Full answer-drop retraining bound to the observed rows approved by the user."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from core.interventions.answer_check import AnswerDrop, digest
from core.interventions.retrain import plan_retrain


def _hashes(protocol):
    datasets = protocol["datasets"]
    if set(datasets) != {"train", "validation", "check", "test"}:
        raise ValueError("Data retraining requires four frozen dataset splits")
    paths = [Path(entry["path"]).resolve() for entry in datasets.values()]
    if len(set(paths)) != 4:
        raise ValueError("Data retraining requires distinct dataset files")
    hashes = {}
    for role, entry in datasets.items():
        hashes[role] = digest(entry["path"])
        if hashes[role] != entry["sha256"]:
            raise ValueError(f"Dataset changed before data retraining: {role}")
    return hashes


def _fixed_text(target, protocol):
    return "\n".join(target.rows(protocol["datasets"]["train"]["path"])) + "\n"


def plan_data_retrain(target: AnswerDrop, protocol):
    hashes = _hashes(protocol)
    if target.kind != "target" or target.patch:
        raise ValueError(
            "Data retraining requires the tested target with no recipe patch"
        )
    fixed = _fixed_text(target, protocol)
    # Price all source tokens as a conservative bound for this deletion-only fix.
    # The retained rows may have a higher mean length than the source file.
    plan = plan_retrain(
        target,
        protocol["hyperparameters"],
        protocol["model"],
        protocol["datasets"]["train"]["examples"],
        protocol["tokens_per_example"],
        protocol["job_id"],
    )
    plan.train_examples = len(fixed.splitlines())
    plan.data_patch = {
        "version": 1,
        "operation": "drop_training_lines",
        "source_sha256": target.source_sha256,
        "line_numbers": list(target.drop_lines),
        "fixed_train_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
        "dataset_sha256": hashes,
        "cost_basis": "source-file token estimate before deletion; not a provider invoice",
    }
    return plan


def retrain_data(retrainer, plan, protocol, spending, on_step=None):
    """Materialize only approved deletions; reserve cost before any provider call.

    Completed results are reused. An interrupted attempt requires reconciliation,
    including when a create-job request may have succeeded without a response.
    """
    patch = plan.data_patch
    if (
        not patch
        or patch.get("version") != 1
        or patch.get("operation") != "drop_training_lines"
    ):
        raise ValueError("Unsupported data retrain patch")
    if (
        plan.jobs != ["fixed"]
        or plan.failed_job_id != protocol["job_id"]
        or not plan.failed_job_id.startswith("ftjob-")
        or plan.patch
        or plan.model != protocol["model"]
        or plan.base_hyperparameters != protocol["hyperparameters"]
    ):
        raise ValueError("Data retrain plan differs from the original protocol")
    target = AnswerDrop(
        plan.hypothesis_id,
        "target",
        {},
        "approved row deletion",
        drop_lines=patch["line_numbers"],
        source_sha256=patch["source_sha256"],
    )
    if asdict(plan_data_retrain(target, protocol)) != asdict(plan):
        raise ValueError("Data retrain plan differs from the frozen estimate or patch")
    fixed = _fixed_text(target, protocol)
    fixed_hash = hashlib.sha256(fixed.encode()).hexdigest()
    if (
        fixed_hash != patch["fixed_train_sha256"]
        or len(fixed.splitlines()) != plan.train_examples
    ):
        raise ValueError("Data retrain patch differs from the approved rows")
    if (
        Path(retrainer.valid_path).resolve()
        != Path(protocol["datasets"]["validation"]["path"]).resolve()
        or Path(retrainer.heldout_path).resolve()
        != Path(protocol["datasets"]["test"]["path"]).resolve()
    ):
        raise ValueError("Data retraining must evaluate the frozen final test")

    storage, prefix = retrainer.storage, retrainer.prefix
    binding_key, result_key = (
        f"{prefix}/data-binding.json",
        f"{prefix}/data-result.json",
    )
    attempt_key = f"{prefix}/data-attempt.json"
    binding = {"version": 1, "plan": asdict(plan), "protocol": protocol}
    if storage.exists(binding_key):
        if json.loads(storage.get_text(binding_key)) != binding:
            raise ValueError("Saved data retrain belongs to another protocol or plan")
    elif storage.exists(f"{prefix}/retrain.json"):
        raise ValueError("Existing retrain jobs lack data provenance")

    def verify():
        if _hashes(protocol) != patch["dataset_sha256"]:
            raise ValueError("Data retrain protocol differs from the approved hashes")

    verify()
    if storage.exists(result_key):
        return json.loads(storage.get_text(result_key))
    if storage.exists(attempt_key):
        raise ValueError(
            "Incomplete data retrain: reconcile the attempt before retrying"
        )
    storage.put_text(binding_key, json.dumps(binding, indent=2))
    key = storage.put_text(f"{prefix}/fixed-train.jsonl", fixed)
    fixed_path = storage.root / key
    operation_id = "answer-data-retrain"
    spending.reserve(operation_id, "retrain", plan.cost_usd)
    storage.put_text(
        attempt_key, json.dumps({"operation_id": operation_id, "state": "started"})
    )
    if on_step:
        on_step(
            {
                "kind": "spending_reserved",
                "operation_id": operation_id,
                "proposal": {"kind": "retrain", "estimated_cost_usd": plan.cost_usd},
            }
        )

    def verify_inputs():
        verify()
        if digest(fixed_path) != fixed_hash:
            raise ValueError("Materialized training data changed during retraining")

    def score_adapter(*args):
        verify_inputs()
        return retrainer.score_adapter(*args)

    def score_base(*args):
        verify_inputs()
        return retrainer.score_base(*args)

    execute = replace(
        retrainer,
        train_path=fixed_path,
        score_adapter=score_adapter,
        score_base=score_base,
    )
    verify_inputs()
    comparison = execute(replace(plan, data_patch=None), on_step=on_step)
    verify_inputs()
    comparison["provenance"] = {
        "version": "answer-data-retrain-v1",
        "dataset_sha256": patch["dataset_sha256"],
        "fixed_train_sha256": fixed_hash,
        "retained_examples": plan.train_examples,
        "model": plan.model,
        "hyperparameters": plan.hyperparameters,
        "evaluator": protocol.get("evaluator"),
        "data_patch": patch,
    }
    spending.settle(operation_id, plan.cost_usd)
    if on_step:
        on_step(
            {
                "kind": "spending_settled",
                "operation_id": operation_id,
                "cost_usd": plan.cost_usd,
                "basis": patch["cost_basis"],
            }
        )
    storage.put_text(result_key, json.dumps(comparison, indent=2))
    return comparison
