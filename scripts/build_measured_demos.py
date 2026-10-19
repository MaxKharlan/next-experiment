"""Build public answer-cut experiment replays from versioned results, offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from core.interventions.stats import causal_check
from core.report.kaggle import analyze

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0b0db5feba473e78cfce4a4af926fb33e2ceb540"
KAGGLE = "docs/results/answer-cut-kaggle-20261019"
TF = "docs/results/answer-cut-tf-20261019"
SCOPE = (
    "Saved training experiment, not an end-to-end agent investigation. "
    "Dropping flagged examples beats the same-size random-drop control; "
    "quality remains below the healthy recipe."
)


def _read(root, path):
    return json.loads((root / path).read_text())


def _sources(root, paths):
    return [
        {
            "path": path,
            "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
            "url": f"https://github.com/MaxKharlan/next-experiment/blob/{REVISION}/{path}",
        }
        for path in paths
    ]


def build_cases(root: Path = ROOT) -> dict[str, dict]:
    bundle = f"{KAGGLE}/bundles/answers-cut-30.json"
    kaggle = analyze(root / bundle, min_effect=0.06)
    saved = _read(root, f"{KAGGLE}/analysis-30.json")
    if kaggle["causal_result"] != saved["causal_result"]:
        raise ValueError("Saved Kaggle analysis differs from recomputed results")
    plan = _read(root, f"{TF}/plan.json")
    jobs = _read(root, f"{TF}/jobs.json")
    scores = _read(root, f"{TF}/results.json")
    tf_runs = []
    for name, score in scores.items():
        job, recipe = jobs["jobs"][name], plan["jobs"][name]
        if (
            score["job_id"] != job["id"]
            or job["status"] != "succeeded"
            or score["trained_tokens"] != job["trained_tokens"]
        ):
            raise ValueError(f"Provider job provenance mismatch: {name}")
        tf_runs.append(
            {
                "run_id": name,
                "role": name.split("-s")[0],
                "seed": recipe["seed"],
                "metric": score["exact_match"],
                "job_id": job["id"],
                "train_sha256": recipe["train_sha256"],
                "adapter_sha256": jobs["adapter_sha256"][name],
                "trained_tokens": score["trained_tokens"],
            }
        )
    cases = {}
    for backend, runs, noise, sources in (
        (
            "Kaggle T4",
            kaggle["runs"],
            None,
            [bundle, f"{KAGGLE}/analysis-30.json", f"{KAGGLE}/evidence/pair.json"],
        ),
        (
            "Token Factory",
            tf_runs,
            0.019,
            [
                f"{TF}/results.json",
                f"{TF}/jobs.json",
                f"{TF}/plan.json",
                "docs/answer-cut.md",
                "scripts/answer_cut_tf.py",
            ],
        ),
    ):
        grouped = {
            role: [r["metric"] for r in runs if r["role"] == role]
            for role in ("healthy", "damaged", "target", "control")
        }
        for values in grouped.values():
            if not values or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(
                    "Each arm requires finite exact-match scores in [0, 1]"
                )
        result = asdict(
            causal_check(
                grouped["target"], grouped["control"], min_effect=0.06, noise_sd=noise
            )
        )
        means = {role: fmean(values) for role, values in grouped.items()}
        gate = (
            math.comb(
                len(grouped["target"]) + len(grouped["control"]), len(grouped["target"])
            )
            * 0.05
            >= 1
        )
        is_tf = backend == "Token Factory"
        measurement = {
            "backend": backend,
            "model": kaggle["model"],
            "task": "banking77",
            "evaluation": kaggle["evaluation"],
            "evaluation_provenance": (
                "Shared A1 check file per the saved experiment protocol; the TF score file does not independently record its hash."
                if is_tf
                else "Check file hash and size recorded in the Kaggle bundle."
            ),
            "runs": runs,
            "means": means,
            "causal_result": result,
            "noise_sd": noise,
            "permutation_gate_applied": gate,
            "quality_drop": means["healthy"] - means["damaged"],
            "gap_to_healthy": means["healthy"] - means["target"],
            "estimated_training_cost_usd": (
                sum(r["trained_tokens"] for r in runs) * plan["price_usd_per_m"] / 1e6
                if is_tf
                else None
            ),
            "scope": SCOPE,
            "limitations": [
                "One task and one model; unique root-cause specificity is not established.",
                "Target and control keep 6144 examples each; control uses about 9% more training tokens.",
                "Healthy and damaged references have one job each."
                if is_tf
                else "Three seeds per arm; noise is estimated from these arms. The separate three-run 10% damage condition is not shown here.",
                "Permutation p cannot reach 0.05 with two seeds; the saved rule uses the interval alone."
                if is_tf
                else "Permutation p is 0.05, the smallest value three seeds per arm allow.",
                "Check-set results; no final untouched test-set evaluation or full agent trace.",
            ],
            "sources": _sources(root, sources),
        }
        name = "answer-cut-tf" if is_tf else "answer-cut-kaggle"
        cases[name] = {
            "title": f"Answer cuts on {backend} — measured experiment",
            "note": SCOPE,
            "case": {
                "kind": "measured_experiment",
                "task": "banking77",
                "symptom": "30% of training answers were cut before training.",
                "job": {"model": kaggle["model"]},
            },
            "report": {
                "investigation_id": name,
                "status": "measured",
                "hypotheses": [],
                "llm_calls": {},
                "llm_cost_usd": 0.0,
                "evidence_scope": SCOPE,
                "measured_experiment": measurement,
                "steps": [{"kind": "measured_experiment", "measurement": measurement}],
            },
        }
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "demo")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, case in build_cases().items():
        (args.out / f"{name}.json").write_text(
            json.dumps(case, indent=2, allow_nan=False) + "\n"
        )
        print(name)


if __name__ == "__main__":
    main()
