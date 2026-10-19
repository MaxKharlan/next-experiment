"""Budgeted API recipe retraining bound to the approved plan and frozen data."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from core.interventions.data_retrain import _hashes
from core.interventions.pairs import Intervention
from core.interventions.retrain import plan_retrain


def retrain_recipe(retrainer, plan, protocol, spending, on_step=None):
    if (
        plan.data_patch is not None
        or plan.jobs != ["fixed"]
        or plan.failed_job_id != protocol["job_id"]
        or not plan.failed_job_id.startswith("ftjob-")
        or plan.model != protocol["model"]
        or plan.base_hyperparameters != protocol["hyperparameters"]
    ):
        raise ValueError("Recipe retrain plan differs from the original protocol")
    expected = plan_retrain(
        Intervention(plan.hypothesis_id, "target", plan.patch, "approved recipe fix"),
        protocol["hyperparameters"],
        protocol["model"],
        protocol["datasets"]["train"]["examples"],
        protocol["tokens_per_example"],
        protocol["job_id"],
    )
    if asdict(expected) != asdict(plan):
        raise ValueError("Recipe retrain differs from the frozen estimate or patch")
    hashes = _hashes(protocol)
    for role, path in (
        ("train", retrainer.train_path),
        ("validation", retrainer.valid_path),
        ("test", retrainer.heldout_path),
    ):
        if Path(path).resolve() != Path(protocol["datasets"][role]["path"]).resolve():
            raise ValueError(
                "Recipe retraining requires the frozen training and final test files"
            )
    storage, prefix = retrainer.storage, retrainer.prefix
    binding_key = f"{prefix}/recipe-binding.json"
    attempt_key = f"{prefix}/recipe-attempt.json"
    result_key = f"{prefix}/recipe-result.json"
    binding = {"version": 1, "plan": asdict(plan), "protocol": protocol}
    if storage.exists(binding_key):
        if json.loads(storage.get_text(binding_key)) != binding:
            raise ValueError("Saved recipe retrain belongs to another plan or protocol")
    elif storage.exists(f"{prefix}/retrain.json"):
        raise ValueError("Existing retrain jobs lack recipe provenance")
    operation = "recipe-retrain"

    def settle():
        rows = [r for r in spending.snapshot()["operations"] if r["id"] == operation]
        if len(rows) != 1 or rows[0]["reserved_usd"] != plan.cost_usd:
            raise ValueError("Recipe retrain reservation differs")
        if rows[0]["status"] == "pending":
            spending.settle(operation, plan.cost_usd)
        elif rows[0]["actual_usd"] != plan.cost_usd:
            raise ValueError("Recipe retrain settlement differs")

    if storage.exists(result_key):
        result = json.loads(storage.get_text(result_key))
        if result.get("provenance", {}).get("dataset_sha256") != hashes:
            raise ValueError("Saved recipe result provenance differs")
        settle()
        return result
    if storage.exists(attempt_key):
        raise ValueError(
            "Incomplete recipe retrain: reconcile the attempt before retrying"
        )
    storage.put_text(binding_key, json.dumps(binding, indent=2))
    spending.reserve(operation, "retrain", plan.cost_usd)
    storage.put_text(
        attempt_key, json.dumps({"operation_id": operation, "state": "started"})
    )
    if on_step:
        on_step(
            {
                "kind": "spending_reserved",
                "operation_id": operation,
                "proposal": {"kind": "retrain", "estimated_cost_usd": plan.cost_usd},
            }
        )

    def verify():
        if _hashes(protocol) != hashes:
            raise ValueError("Recipe retrain data changed")

    def score_adapter(*args):
        verify()
        return retrainer.score_adapter(*args)

    def score_base(*args):
        verify()
        return retrainer.score_base(*args)

    verify()
    result = replace(retrainer, score_adapter=score_adapter, score_base=score_base)(
        plan, on_step=on_step
    )
    verify()
    result["provenance"] = {
        "version": "recipe-retrain-v1",
        "dataset_sha256": hashes,
        "model": plan.model,
        "hyperparameters": plan.hyperparameters,
        "evaluator": protocol.get("evaluator"),
        "evaluation_runtime": protocol.get("evaluation_runtime"),
        "cost_basis": "approved training estimate; not a provider invoice",
    }
    # Persist the verified result first, so a crash during settlement is recoverable.
    storage.put_text(result_key, json.dumps(result, indent=2))
    settle()
    if on_step:
        on_step(
            {
                "kind": "spending_settled",
                "operation_id": operation,
                "cost_usd": plan.cost_usd,
                "basis": result["provenance"]["cost_basis"],
            }
        )
    return result
