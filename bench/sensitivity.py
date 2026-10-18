"""Descriptive recipe sensitivity, keeping backend/dataset studies separate."""

import argparse
import json
import math
import statistics
from pathlib import Path

FIELDS = {
    "lr_too_low": "learning_rate",
    "too_many_epochs": "n_epochs",
    "lora_rank_too_small": "lora_r",
}


def wilson_interval(hits, total):
    if not 0 <= hits <= total or total < 1:
        raise ValueError("Invalid detection counts")
    z = 1.96
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def detection_report(rows):
    """Injected-recipe top-1 recovery, distinct from confirming harmful faults."""
    healthy = {(r["study"], r["seed"]): r for r in rows if r["fault"] is None}
    if len(healthy) != sum(r["fault"] is None for r in rows):
        raise ValueError("Duplicate healthy detection case")
    seen = set()
    groups = {}
    for row in rows:
        identity = (
            row["study"],
            row["seed"],
            row["fault"],
            json.dumps(row["hyperparameters"], sort_keys=True),
        )
        if identity in seen:
            raise ValueError("Duplicate detection case")
        seen.add(identity)
        base = healthy.get((row["study"], row["seed"]))
        if base is None:
            raise ValueError("Missing healthy detection case")
        fault = row["fault"]
        if fault is None:
            continue
        field = FIELDS[fault]
        value = row["hyperparameters"][field]
        base_value = base["hyperparameters"][field]
        strength = value / base_value if field == "n_epochs" else base_value / value
        groups.setdefault((row["study"], fault, value, strength), []).append(row)
    curves = []
    for (study, fault, value, strength), cases in sorted(groups.items()):
        hits = sum(bool(r["ranking"]) and r["ranking"][0][0] == fault for r in cases)
        curves.append(
            {
                "study": study,
                "fault": fault,
                "parameter": FIELDS[fault],
                "value": value,
                "severity_ratio": strength,
                "hits": hits,
                "total": len(cases),
                "top1_recovery": hits / len(cases),
                "wilson_95": wilson_interval(hits, len(cases)),
                "errors": sum(bool(r.get("error")) for r in cases),
            }
        )
    thresholds = []
    for study, fault in sorted({(r["study"], r["fault"]) for r in curves}):
        points = sorted(
            [r for r in curves if (r["study"], r["fault"]) == (study, fault)],
            key=lambda r: r["severity_ratio"],
        )
        observed = next((r["value"] for r in points if r["top1_recovery"] >= 0.8), None)
        supported = next((r["value"] for r in points if r["wilson_95"][0] >= 0.8), None)
        thresholds.append(
            {
                "study": study,
                "fault": fault,
                "minimum_tested_value_at_observed_80_percent": observed,
                "minimum_tested_value_with_95_percent_lower_bound_at_80_percent": supported,
            }
        )
    baselines = []
    for study in sorted({r["study"] for r in rows}):
        cases = [r for (s, _), r in healthy.items() if s == study]
        baselines.append(
            {
                "study": study,
                "total": len(cases),
                "hyperparameters": cases[0]["hyperparameters"],
                "recipe_top1_on_healthy": sum(
                    bool(r["ranking"]) and r["ranking"][0][0] in FIELDS for r in cases
                ),
                "any_fault_top1_on_healthy": sum(
                    bool(r["ranking"])
                    and r["ranking"][0][0] not in {"other", "unlisted"}
                    for r in cases
                ),
                "top1_counts": {
                    fault: sum(
                        bool(r["ranking"]) and r["ranking"][0][0] == fault
                        for r in cases
                    )
                    for fault in FIELDS
                },
            }
        )
    return {
        "scope": "pre-intervention top-1 recipe recovery; no causal-confirmation sensitivity",
        "curves": curves,
        "thresholds": thresholds,
        "healthy": baselines,
        "cost_usd": sum(r["cost_usd"] for r in rows),
    }


def summarize_study(rows):
    if len({r["study"] for r in rows}) != 1:
        raise ValueError("Do not mix studies, backends or evaluation datasets")
    healthy = {r["seed"]: r for r in rows if r["fault"] is None}
    if len(healthy) != sum(r["fault"] is None for r in rows):
        raise ValueError("Duplicate healthy seeds")
    groups = {}
    seen = set()
    for row in rows:
        if (
            row["seed"] not in healthy
            or not math.isfinite(row["exact_match"])
            or not 0 <= row["exact_match"] <= 1
        ):
            raise ValueError("Missing seed baseline or invalid score")
        if row["fault"] is None:
            continue
        field = FIELDS[row["fault"]]
        value = row["hyperparameters"][field]
        key = (row["fault"], value)
        identity = (*key, row["seed"])
        if identity in seen:
            raise ValueError("Duplicate fault/strength/seed")
        seen.add(identity)
        baseline = healthy[row["seed"]]
        if row["hyperparameters"].keys() != baseline["hyperparameters"].keys():
            raise ValueError("Recipe fields differ from the healthy baseline")
        for hp, baseline_value in baseline["hyperparameters"].items():
            if hp != field and row["hyperparameters"].get(hp) != baseline_value:
                raise ValueError("Multiple recipe fields changed")
        groups.setdefault(key, []).append((row, baseline))
    curves = []
    for (fault, value), pairs in sorted(groups.items()):
        scores = [r["exact_match"] for r, _ in pairs]
        deltas = [r["exact_match"] - h["exact_match"] for r, h in pairs]
        curves.append(
            {
                "fault": fault,
                "parameter": FIELDS[fault],
                "value": value,
                "seeds": [r["seed"] for r, _ in pairs],
                "scores": scores,
                "mean_exact_match": statistics.mean(scores),
                "mean_change_from_healthy": statistics.mean(deltas),
                "seed_sd": statistics.stdev(scores) if len(scores) > 1 else None,
            }
        )
    return {
        "study": rows[0]["study"],
        "healthy": list(healthy.values()),
        "curves": curves,
        "scope": "descriptive validation sensitivity; no causal significance or held-out claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    reports = [
        summarize_study([r for r in rows if r["study"] == study])
        for study in sorted({r["study"] for r in rows})
    ]
    with args.out.open("x") as handle:
        json.dump(reports, handle, indent=2)


if __name__ == "__main__":
    main()
