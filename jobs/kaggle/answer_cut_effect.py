"""A5 effect check for answer_truncation: healthy runs against cut answers, on check.

Run from the repository root on Kaggle (CUDA) or locally with --smoke, after the
BANKING77 splits and their faulted copies exist (see docs/answer-cut.md).
Each run is jobs/kaggle/train_lora.py with the healthy recipe on one task directory;
core.eval scores its adapter on the untouched check split. Results go to
runs/results.json; a run already there is not repeated.

    PYTHONPATH=. python -m jobs.kaggle.answer_cut_effect [--smoke]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from core.case import EvaluatorSpec
from core.eval import evaluate

SMOKE = "--smoke" in sys.argv
TASKS = {
    "healthy": "artifacts/tasks/banking77",
    "answers-cut-10": "artifacts/tasks/banking77-answers-cut-10",
    "answers-cut-30": "artifacts/tasks/banking77-answers-cut-30",
}
SEEDS = [0] if SMOKE else [0, 1, 2]
RUNS = Path("runs-smoke" if SMOKE else "runs")
SPEC = EvaluatorSpec(name="intent_label", metric="exact_match")


def head(path: Path, n: int, out: Path) -> Path:
    lines = [x for x in path.read_text().splitlines(True) if x.strip()]
    out.write_text("".join(lines[:n]))
    return out


def main() -> None:
    RUNS.mkdir(exist_ok=True)
    results_path = RUNS / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    check = Path(TASKS["healthy"]) / "check.jsonl"
    if SMOKE:
        check = head(check, 4, RUNS / "check.jsonl")
    labels = {
        json.loads(x)["messages"][-1]["content"] for x in check.read_text().splitlines()
    }
    for seed in SEEDS:
        for name, task in TASKS.items():
            key = f"{name}-s{seed}"
            if key in results:
                print(key, "done before", results[key]["exact_match"])
                continue
            out = RUNS / key
            out.mkdir(exist_ok=True)
            train, valid = Path(task) / "train.jsonl", Path(task) / "validation.jsonl"
            if SMOKE:
                train = head(train, 16, out / "train.jsonl")
                valid = head(valid, 8, out / "valid.jsonl")
            started = time.monotonic()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jobs.kaggle.train_lora",
                    "--train",
                    str(train),
                    "--valid",
                    str(valid),
                    "--out",
                    str(out),
                    "--seed",
                    str(seed),
                    "--evals-per-epoch",
                    "1",
                ],
                check=True,
            )
            trained = time.monotonic() - started
            result = evaluate(check, SPEC, adapter=out / "adapter")
            answers = [e.answer.strip() for e in result.examples]
            cut = sum(
                a not in labels and any(lb.startswith(a) for lb in labels)
                for a in answers
            )
            results[key] = {
                "exact_match": result.value,
                "cut_answers": cut / len(answers),
                "no_eos": sum(not e.stopped_on_eos for e in result.examples)
                / len(answers),
                "train_s": round(trained),
                "eval_s": round(time.monotonic() - started - trained),
            }
            results_path.write_text(json.dumps(results, indent=2))
            print(key, results[key], flush=True)
    print(json.dumps(results, indent=2))
    summary(results)


def summary(results: dict) -> None:
    """Mean and spread of exact match per arm, and each fault's drop from healthy.

    The spread between seeds is this setup's own noise floor: the 0.06 threshold
    was measured on Token Factory and does not carry over to these runs.
    """
    arms: dict[str, list[float]] = {}
    for key, row in results.items():
        arms.setdefault(key.rsplit("-s", 1)[0], []).append(row["exact_match"])
    stats = {}
    for arm, values in arms.items():
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5
        stats[arm] = {"n": len(values), "mean": round(mean, 4), "sd": round(sd, 4)}
    healthy = stats.get("healthy", {}).get("mean")
    for arm, row in stats.items():
        if healthy is not None and arm != "healthy":
            row["drop_from_healthy"] = round(healthy - row["mean"], 4)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
