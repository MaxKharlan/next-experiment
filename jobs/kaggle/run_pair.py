"""B8: run a causal check locally: target and control mini-jobs, then the verdict.

Builds the intervention pair for a fault class from the failed job's hyperparameters,
trains each arm once per seed with train_lora.py, scores exact match on the
validation file and applies core.interventions.stats.causal_check. Writes
<out>/<arm>-s<seed>/ for every run and <out>/pair.json with the arms, the per-seed
metrics and the verdict.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bench.faults.recipe import HEALTHY_HP, STRENGTHS
from core.agent.context import InvestigationContext
from core.agent.schemas import Hypothesis
from core.interventions.pairs import Intervention, make_pair, take_fraction
from core.interventions.stats import causal_check
from jobs.kaggle.train_lora import read_jsonl, train


def train_arm(
    arm: Intervention,
    base: dict[str, Any],
    train_path: str | Path,
    valid_path: str | Path,
    out: Path,
    seeds: list[int],
    model: str = "Qwen/Qwen3-0.6B",
) -> None:
    """Train one arm per seed and append each exact match to arm.metrics.

    Runs with a summary.json under out/<kind>-s<seed>/ are reused, not retrained.
    """
    print(f"{arm.kind}: {arm.description}", flush=True)
    train_path = Path(train_path)
    if arm.data_fraction < 1:
        subset = take_fraction(read_jsonl(train_path), arm.data_fraction)
        train_path = out / f"{arm.kind}-train.jsonl"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text(
            "".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8"
        )
    for seed in seeds:
        run_dir = out / f"{arm.kind}-s{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            summary = train(
                train_path,
                valid_path,
                run_dir,
                model_id=model,
                hp=arm.hyperparameters(base),
                seed=seed,
                generate=True,
            )
        arm.metrics.append(summary["final"]["exact_match"])
        arm.ft_job_ids.append(f"local:{run_dir}")
        print(f"  seed {seed}: exact match {arm.metrics[-1]:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fault", required=True, choices=sorted(STRENGTHS))
    parser.add_argument(
        "--job-hp", default="{}", help="JSON: the failed job's hyperparameters"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    # 2 noise-floor SDs of exact match from docs/b10-noise-floor.md
    parser.add_argument("--min-effect", type=float, default=0.03)
    parser.add_argument("--noise-sd", type=float, default=0.0146)
    args = parser.parse_args()

    out = Path(args.out)
    base = {**HEALTHY_HP, **json.loads(args.job_hp)}
    ctx = InvestigationContext(
        investigation_id="local",
        mode="A",
        job_id="local",
        job_config={"hyperparameters": base},
    )
    hypothesis = Hypothesis(
        id="h1",
        fault_class=args.fault,
        location="hyperparameters",
        prior=0.5,
        rationale="set from the command line",
    )
    target, control = make_pair(hypothesis, ctx)
    for arm in (target, control):
        train_arm(arm, base, args.train, args.valid, out, args.seeds, args.model)

    result = causal_check(
        target.metrics,
        control.metrics,
        min_effect=args.min_effect,
        noise_sd=args.noise_sd,
    )
    record = {
        "fault": args.fault,
        "job_hyperparameters": base,
        "target": asdict(target),
        "control": asdict(control),
        "result": asdict(result),
    }
    (out / "pair.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"{result.status}: effect {result.effect:+.3f} "
        f"[{result.ci_low:+.3f}, {result.ci_high:+.3f}], p={result.p_value:.3f}"
    )


if __name__ == "__main__":
    main()
