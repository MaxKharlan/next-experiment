"""Apply the frozen development-noise gate before any fresh training jobs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from bench.causal_power import simulate
from bench.tasks.structured import sha
from core.interventions.stats import noise_floor
from core.interventions.tf_runner import weights_digest
from scripts.fresh_answer_calibration import load, save


def decide(groups, digests, rule):
    if set(groups) != {"healthy", "damaged"} or any(
        len(values) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
        for values in groups.values()
    ):
        raise ValueError("Require exactly three finite scores in each development arm")
    if set(digests) != set(groups) or any(
        len(values) != 3 or any(not value for value in values)
        for values in digests.values()
    ):
        raise ValueError("Require three weight digests in each arm")
    noise = noise_floor(groups)
    result = {
        "groups": groups,
        "noise": asdict(noise),
        "selected_seeds": [],
        "power": [],
        "assumptions": "Independent normal seed-level metrics with equal known SD; six development runs only. MC interval excludes uncertainty in the estimated SD and task transfer.",
    }
    if any(len(set(values)) != len(values) for values in digests.values()):
        return {**result, "status": "stopped", "reason": "identical_weights"}
    if noise.sd == 0:
        return {**result, "status": "stopped", "reason": "zero_sd"}
    for count in rule["candidate_seeds"]:
        row = simulate(
            effect=rule["anticipated_effect"],
            noise_sd=noise.sd,
            seeds=count,
            trials=rule["trials"],
            min_effect=rule["min_effect"],
            random_seed=rule["random_seed"],
        )
        result["power"].append(row)
        if row["mc_interval_95"][0] >= rule["minimum_lower_mc_bound"]:
            return {
                **result,
                "status": "passed",
                "reason": "prespecified_power_gate",
                "selected_seeds": list(range(2101, 2101 + count)),
            }
    return {**result, "status": "stopped", "reason": "insufficient_power"}


def analyze(root, expected):
    lock = json.loads((root / "power-analysis-lock.json").read_text())
    if (
        lock["protocol_sha256"] != expected
        or lock["analysis_source_sha256"] != sha(Path(__file__))
        or lock["scores_present_at_lock"] != 0
    ):
        raise ValueError("Power analysis differs from its pre-scoring lock")
    output = root / "power-decision.json"
    if output.exists():
        raise FileExistsError("Power decision already exists; do not repeat selection")
    plan, state = load(root, expected)
    if set(state) != set(plan["jobs"]):
        raise ValueError("Incomplete job state")
    groups, digests = {"healthy": [], "damaged": []}, {"healthy": [], "damaged": []}
    for name, entry in state.items():
        record = json.loads((root / "scores" / f"{name}.json").read_text())
        if entry["status"] != "succeeded" or record["job_id"] != entry["job_id"]:
            raise ValueError("Score is not from the planned successful job")
        if record["protocol_sha256"] != expected:
            raise ValueError("Score protocol differs")
        if record["weights_sha256"] != weights_digest(Path(entry["adapter"])):
            raise ValueError("Adapter weights changed")
        result = record["result"]
        if (
            result["runtime"] != plan["evaluation_runtime"]
            or result["data_sha256"] != plan["source_datasets"]["check"]["sha256"]
            or result["metric"] != plan["evaluator"]["metric"]
        ):
            raise ValueError("Score evaluator binding differs")
        arm = name.split("-", 1)[0]
        groups[arm].append(result["value"])
        digests[arm].append(record["weights_sha256"])
    result = decide(groups, digests, plan["power_rule"])
    result.update(protocol_sha256=expected, weight_digests=digests)
    save(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    analyze(args.root, args.protocol_sha256)


if __name__ == "__main__":
    main()
