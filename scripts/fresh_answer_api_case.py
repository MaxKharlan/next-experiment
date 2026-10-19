"""Freeze one strong-condition API request after fresh references finish.

Scores only the original strong job on check. The symptom uses that measurement,
without fault injection truth. Final test is passed to the protected evaluator and
is not scored here. A shared-ledger allocation includes all API inference/training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from apps.api.case_input import CaseStart, prepare_case
from bench.tasks.structured import sha
from core.case import EvaluatorSpec
from core.eval import evaluate
from core.interventions.tf_runner import weights_digest
from scripts.fresh_answer_calibration import save
from scripts.fresh_answer_references import load


def prepare(root, references, expected):
    plan, state, ledger = load(references, expected)
    development = Path(plan["development_root"])
    handoff = json.loads((development / "mps-handoff.json").read_text())
    if handoff.get("released") is not True:
        raise ValueError("B9 MPS handoff required")
    if set(state) != {"healthy", "mild", "strong"} or any(
        entry["status"] != "succeeded" or not entry.get("settled")
        for entry in state.values()
    ):
        raise ValueError("All three reference jobs must finish and settle")
    decision = json.loads((development / "power-decision.json").read_text())
    if decision["status"] != "passed":
        raise ValueError("Power gate not passed")
    root.mkdir(parents=True, exist_ok=False)
    cohort = Path(plan["cohort"])
    original = json.loads((cohort / "execution-plan.json").read_text())
    observed = cohort / "observed/strong"
    refs = {}
    for role in ("train", "validation", "check", "test"):
        path = observed / f"{role}.jsonl"
        digest = sha(path)
        if digest != original["datasets"]["strong"][role]:
            raise ValueError("Observed cohort changed")
        refs[role] = {"ref": str(path.resolve()), "sha256": digest}
    strong = state["strong"]
    adapter = Path(strong["adapter"])
    if weights_digest(adapter) != strong["weights_sha256"]:
        raise ValueError("Strong checkpoint changed")
    evaluator = {
        "name": "ticket_json_v1",
        "metric": "json_fields_exact",
        "parameters": {"max_new_tokens": 192},
    }
    result = evaluate(
        refs["check"]["ref"],
        EvaluatorSpec.model_validate(evaluator),
        adapter=adapter,
        base_model=plan["model"],
        runtime=plan["evaluation_runtime"],
        device="mps",
    )
    save(root / "original-check.json", result.to_dict())
    checkpoints = json.loads(
        (references / "provider/strong-checkpoints.json").read_text()
    )
    checkpoint_metrics = [
        {
            "epoch": i,
            "train_loss": ck["metrics"].get("train_loss"),
            "valid_loss": ck["metrics"].get("valid_loss"),
        }
        for i, ck in enumerate(sorted(checkpoints, key=lambda c: c["step_number"]), 1)
    ]
    # Leave $0.50 unallocated for taxes/contingency, as preregistered.
    budget = round(ledger.snapshot()["available_usd"] - 0.5, 8)
    if budget <= 0:
        raise ValueError("No remaining API allocation")
    payload = {
        "case": {
            "task": "json_extraction",
            "mode": "A",
            "symptom": f"A succeeded JSON extraction fine-tune has json_fields_exact={result.value:.6f} on {len(result.examples)} check examples. Investigate whether data or recipe changes improve answer quality. A high score may justify no intervention.",
            "job": {
                "id": strong["job_id"],
                "model": plan["model"],
                "hyperparameters": plan["recipe"],
                "checkpoints": checkpoint_metrics,
            },
            "datasets": refs,
            "evaluator": evaluator,
            "budget_usd": budget,
        },
        "experiment": {
            "name": "fresh-answer-20261019-strong",
            "noise_sd": decision["noise"]["sd"],
            "min_effect": 0.08,
            "tokens_per_example": 306,
            "seeds": decision["selected_seeds"],
            "evaluation_runtime": plan["evaluation_runtime"],
        },
    }
    prepared = prepare_case(CaseStart.model_validate(payload))
    save(root / "request.json", payload)
    save(root / "prepared.json", prepared)
    sources = subprocess.check_output(
        ["git", "ls-files", "core", "apps/api", "scripts", "bench", "uv.lock"],
        text=True,
    ).splitlines()
    binding = {
        "reference_protocol_sha256": expected,
        "power_decision_sha256": plan["power_decision_sha256"],
        "request_sha256": sha(root / "request.json"),
        "prepared_sha256": sha(root / "prepared.json"),
        "original_check_sha256": sha(root / "original-check.json"),
        "source_sha256": {p: sha(Path(p)) for p in sources if Path(p).is_file()},
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "idempotency_key": "fresh-answer-20261019-strong-v1",
        "api_allocation_usd": budget,
        "stopping": original["stopping_rule"],
        "approval_policy": "At most one answer-drop target/control pair; decline other paid interventions. At most one frozen answer-drop final retrain. Restart the idle API once at a persisted approval boundary before approving.",
        "final_test_model_evaluations_before_start": 0,
    }
    save(root / "execution.json", binding)
    ledger.reserve(
        "fresh-api-allocation", "api_inference_pair_and_final_retrain", budget
    )
    save(root / "parent-spending.json", ledger.snapshot())
    print(
        json.dumps(
            {
                "request_sha256": binding["request_sha256"],
                "execution_sha256": sha(root / "execution.json"),
                "budget_usd": budget,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    prepare(args.root, args.references, args.protocol_sha256)


if __name__ == "__main__":
    main()
