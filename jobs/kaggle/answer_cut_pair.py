"""A7 effect check: drop the examples the answer_cut probe flags against as many random ones.

Runs after jobs/kaggle/answer_cut_effect.py, from the repository root. The faulted
arm is that script's answers-cut-<strength> runs; this adds the target and control
arms on the same seeds, scores them on the untouched check split and applies
core.interventions.stats.causal_check. Results go to runs/pair.json.

    PYTHONPATH=. python -m jobs.kaggle.answer_cut_pair [--strength 30] [--smoke]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from core.agent.context import InvestigationContext
from core.case import EvaluatorSpec
from core.eval import evaluate
from core.interventions.data_pairs import answer_cut_pair, flagged_lines
from core.interventions.stats import causal_check
from core.probes.level0.answers import AnswerCut

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--strength", type=int, default=30)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

TASK = Path(f"artifacts/tasks/banking77-answers-cut-{args.strength}")
CHECK = Path("artifacts/tasks/banking77/check.jsonl")
SEEDS = [0, 1] if args.smoke else [0, 1, 2]
RUNS = Path("runs-smoke" if args.smoke else "runs")
SPEC = EvaluatorSpec(name="intent_label", metric="exact_match")


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text().splitlines(True) if x.strip()]


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    out_path = RUNS / "pair.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    train_rows = lines(TASK / "train.jsonl")
    valid = TASK / "validation.jsonl"
    check = CHECK
    if args.smoke:
        train_rows = train_rows[:40]
        (RUNS / "check.jsonl").write_text("".join(lines(CHECK)[:4]))
        check = RUNS / "check.jsonl"
        (RUNS / "valid.jsonl").write_text("".join(lines(valid)[:8]))
        valid = RUNS / "valid.jsonl"
    probe_train = RUNS / "probe-train.jsonl"
    probe_train.write_text("".join(train_rows))
    ctx = InvestigationContext(
        "a7", "A", "kaggle", {}, files={"train": str(probe_train)}
    )
    probe = AnswerCut().run(ctx)
    raw = json.loads(probe.summary)
    print("probe:", probe.outcome, raw["files"]["train"]["cut"], "flagged", flush=True)
    arms = answer_cut_pair("answer_truncation", flagged_lines(raw), len(train_rows))
    for arm in arms:
        path = RUNS / f"{arm.kind}-train.jsonl"
        path.write_text("".join(arm.apply(train_rows)))
        print(arm.kind, arm.description, flush=True)
        for seed in SEEDS:
            key = f"cut{args.strength}-{arm.kind}-s{seed}"
            if key in results:
                print(key, "done before", results[key]["exact_match"])
                continue
            run_dir = RUNS / key
            started = time.monotonic()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jobs.kaggle.train_lora",
                    "--train",
                    str(path),
                    "--valid",
                    str(valid),
                    "--out",
                    str(run_dir),
                    "--seed",
                    str(seed),
                    "--evals-per-epoch",
                    "1",
                ],
                check=True,
            )
            trained = time.monotonic() - started
            result = evaluate(check, SPEC, adapter=run_dir / "adapter")
            results[key] = {
                "exact_match": result.value,
                "train_s": round(trained),
                "eval_s": round(time.monotonic() - started - trained),
            }
            out_path.write_text(json.dumps(results, indent=2))
            print(key, results[key], flush=True)
    verdict(results)


def verdict(results: dict) -> None:
    """Means per arm, the healthy runs' spread as noise and the causal check."""
    effect_runs = RUNS / "results.json"
    earlier = json.loads(effect_runs.read_text()) if effect_runs.exists() else {}
    arm = {
        name: [
            row["exact_match"]
            for key, row in {**earlier, **results}.items()
            if key.rsplit("-s", 1)[0] == name
        ]
        for name in (
            "healthy",
            f"answers-cut-{args.strength}",
            f"cut{args.strength}-target",
            f"cut{args.strength}-control",
        )
    }
    summary = {
        name: {"n": len(v), "mean": round(statistics.fmean(v), 4)}
        for name, v in arm.items()
        if v
    }
    print(json.dumps(summary, indent=2))
    healthy = arm["healthy"]
    noise = statistics.stdev(healthy) if len(healthy) >= 2 else None
    fix, control = arm[f"cut{args.strength}-target"], arm[f"cut{args.strength}-control"]
    if len(fix) >= 2 and len(control) >= 2:
        # three SDs of the healthy runs' spread, at least 0.02: a first guess for
        # this setup, not the 0.06 measured on Token Factory
        min_effect = max(0.02, 3 * noise) if noise else 0.02
        check_result = causal_check(fix, control, min_effect=min_effect)
        print("noise_sd", noise, "min_effect", round(min_effect, 4))
        print(check_result)


if __name__ == "__main__":
    main()
