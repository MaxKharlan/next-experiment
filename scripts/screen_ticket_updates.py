"""Measure challenge sensitivity of fixed existing adapters, with no training.

This exploratory screen cannot unlock paid execution under either stopped
protocol. It tests whether history complexity exposes a content failure once
models have already learned the bare-JSON output convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.tasks.structured import sha
from core.case import EvaluatorSpec
from core.eval.runtime import verify_runtime
from core.interventions.stats import noise_floor
from core.interventions.tf_runner import weights_digest
from scripts.fresh_answer_calibration import save


def prepare(root, old, candidate):
    root.mkdir(parents=True, exist_ok=False)
    prior = json.loads((old / "protocol.json").read_text())
    data_plan = json.loads((candidate / "protocol.json").read_text())
    state = json.loads((old / "state.json").read_text())
    if len(state) != 6 or any(e["status"] != "succeeded" for e in state.values()):
        raise ValueError("Expected six completed existing adapters")
    plan = {
        "id": "ticket-updates-fixed-adapter-screen-20261019",
        "phase": "exploratory_development_sensitivity",
        "purpose": "Test content headroom and damage sensitivity after a learned bare-JSON convention. No paid submission, seed selection or final-test evaluation is authorized by this screen.",
        "prior_protocol_sha256": sha(old / "protocol.json"),
        "candidate_protocol_sha256": sha(candidate / "protocol.json"),
        "data": data_plan["datasets"]["check"],
        "evaluator": prior["evaluator"],
        "model": prior["model"],
        "runtime": prior["evaluation_runtime"],
        "adapters": {
            name: {k: e[k] for k in ("adapter", "job_id", "weights_sha256")}
            for name, e in state.items()
        },
        "source_sha256": {
            p: sha(Path(p))
            for p in (
                "scripts/screen_ticket_updates.py",
                "core/eval/__init__.py",
                "core/eval/data.py",
                "core/eval/generate.py",
                "core/eval/structured.py",
                "core/eval/runtime.py",
            )
        },
        "eligibility": {
            "healthy_min": 0.1,
            "healthy_max": 0.95,
            "minimum_mean_damage_gap": 0.08,
            "require_nonzero_pooled_sd": True,
        },
        "stopping": "Exactly six fixed adapter evaluations, no training/repeats/seed selection. Report all outcomes. Failure rejects this candidate for a paid confirmatory run; a pass requires a distinct new preregistered development training protocol. All predecessor stopping decisions remain terminal.",
        "cost_usd": 0,
    }
    save(root / "protocol.json", plan)
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")


def score(root):
    from core.eval import evaluate

    digest = (root / "protocol.sha256").read_text().strip()
    if sha(root / "protocol.json") != digest:
        raise ValueError("Screen protocol changed")
    plan = json.loads((root / "protocol.json").read_text())
    for p, expected in plan["source_sha256"].items():
        if sha(Path(p)) != expected:
            raise ValueError("Source changed")
    if sha(Path(plan["data"]["path"])) != plan["data"]["sha256"]:
        raise ValueError("Data changed")
    verify_runtime(plan["runtime"])
    if (root / "decision.json").exists():
        raise ValueError("Screen finished")
    groups = {"healthy": [], "damaged": []}
    for name, entry in plan["adapters"].items():
        if weights_digest(Path(entry["adapter"])) != entry["weights_sha256"]:
            raise ValueError("Adapter changed")
        path = root / "scores" / f"{name}.json"
        if not path.exists():
            result = evaluate(
                plan["data"]["path"],
                EvaluatorSpec.model_validate(plan["evaluator"]),
                adapter=entry["adapter"],
                base_model=plan["model"],
                runtime=plan["runtime"],
                device="mps",
            )
            save(
                path,
                {
                    "protocol_sha256": digest,
                    "job_id": entry["job_id"],
                    "weights_sha256": entry["weights_sha256"],
                    "result": result.to_dict(),
                },
            )
            print(name, result.value, flush=True)
        record = json.loads(path.read_text())
        if (
            record["protocol_sha256"] != digest
            or record["job_id"] != entry["job_id"]
            or record["result"]["data_sha256"] != plan["data"]["sha256"]
        ):
            raise ValueError("Score binding mismatch")
        groups[name.split("-", 1)[0]].append(record["result"]["value"])
    means = {k: sum(v) / len(v) for k, v in groups.items()}
    sd = noise_floor(groups).sd
    rule = plan["eligibility"]
    failures = []
    if not rule["healthy_min"] <= means["healthy"] <= rule["healthy_max"]:
        failures.append("healthy_floor_or_ceiling")
    if means["healthy"] - means["damaged"] < rule["minimum_mean_damage_gap"]:
        failures.append("insufficient_damage_sensitivity")
    if sd == 0:
        failures.append("zero_sd")
    save(
        root / "decision.json",
        {
            "status": "rejected" if failures else "eligible_for_new_development_design",
            "reasons": failures,
            "groups": groups,
            "means": means,
            "pooled_sd": sd,
            "protocol_sha256": digest,
            "selected_seeds": [],
            "paid_jobs": 0,
            "inference": "Exploratory sensitivity of fixed historical adapters on new development histories. No fresh causal or final-test claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.old, args.candidate)
    else:
        score(args.root)


if __name__ == "__main__":
    main()
