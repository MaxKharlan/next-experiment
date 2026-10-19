"""Turn the Kaggle output of the answer-cut runs into bundles for scripts.analyze_kaggle.

Reads the downloaded Kaggle output (the next-experiment/ folder of /kaggle/working)
and writes one bundle per fault strength in docs/kaggle-results.schema.json format:

  <out>/answers-cut-30.json  healthy, damaged (30%), target and control runs
  <out>/answers-cut-10.json  healthy and damaged (10%) runs

The metric of every run is exact match on the untouched check split, scored by
core.eval in jobs/kaggle/answer_cut_effect.py and answer_cut_pair.py. The runs'
summary.json files from train_lora.py score validation, so they are copied next to
the bundles as evidence but not referenced by summary_path: a validation score must
not stand for check. Training-file hashes come from the files the runs trained on, as saved in the output.

    python -m jobs.kaggle.export_answer_cut <kaggle output>/next-experiment \
        --out artifacts/kaggle/a5-a7 --env torch=2.11.0+cu128 --env gpu="Tesla T4"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL = "Qwen/Qwen3-0.6B"
ENVIRONMENT = {
    "transformers": "5.20.0",
    "peft": "0.21.2",
    "repository_commit": "2cb1661",
    "notebook": "jobs/kaggle/answer_cut.ipynb",
}
CONTROL = (
    "drop as many training examples as the target, drawn at random (seed 0) from "
    "the examples whose answers the answer_cut probe did not flag; same "
    "hyperparameters and training seeds as the target"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def run(root: Path, key: str, role: str, metric: float, train: Path) -> dict:
    summary = json.loads((root / "runs" / key / "summary.json").read_text())
    if summary["model"] != MODEL:
        raise ValueError(f"{key}: trained {summary['model']}, not {MODEL}")
    seed = int(key.rsplit("-s", 1)[1])
    if summary["seed"] != seed:
        raise ValueError(f"{key}: summary seed {summary['seed']} is not {seed}")
    if summary["train_examples"] != count(train):
        raise ValueError(f"{key}: {train} is not the file it trained on")
    return {
        "run_id": key,
        "role": role,
        "seed": seed,
        "metric": metric,
        "train_sha256": sha256(train),
        "hyperparameters": summary["hyperparameters"],
    }


def bundle(root: Path, strength: int, environment: dict[str, str]) -> dict:
    tasks = root / "artifacts" / "tasks"
    check = tasks / "banking77" / "check.jsonl"
    effect = json.loads((root / "runs" / "results.json").read_text())
    pair_path = root / "runs" / "pair.json"
    pair = json.loads(pair_path.read_text()) if pair_path.exists() else {}
    arms = {
        "healthy": ("healthy", tasks / "banking77" / "train.jsonl", effect),
        "damaged": (
            f"answers-cut-{strength}",
            tasks / f"banking77-answers-cut-{strength}" / "train.jsonl",
            effect,
        ),
        "target": (f"cut{strength}-target", root / "runs" / "target-train.jsonl", pair),
        "control": (
            f"cut{strength}-control",
            root / "runs" / "control-train.jsonl",
            pair,
        ),
    }
    runs = []
    for role, (prefix, train, results) in arms.items():
        for key, row in sorted(results.items()):
            if key.rsplit("-s", 1)[0] == prefix:
                runs.append(run(root, key, role, row["exact_match"], train))
    out = {
        "version": 1,
        "backend": "kaggle",
        "model": MODEL,
        "fault": "answer_truncation",
        "evaluation": {
            "split": "check",
            "sha256": sha256(check),
            "examples": count(check),
            "metric": "exact_match",
        },
        "runs": runs,
        "environment": environment,
    }
    if any(r["role"] in ("target", "control") for r in runs):
        out["control_description"] = CONTROL
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="next-experiment/ from the Kaggle output"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env", action="append", default=[], help="name=value")
    args = parser.parse_args()
    environment = dict(ENVIRONMENT)
    for item in args.env:
        name, _, value = item.partition("=")
        environment[name] = value
    args.out.mkdir(parents=True, exist_ok=False)
    for strength in (30, 10):
        data = bundle(args.root, strength, environment)
        path = args.out / f"answers-cut-{strength}.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        roles = sorted({r["role"] for r in data["runs"]})
        print(path, len(data["runs"]), "runs:", ", ".join(roles))
    evidence = args.out / "evidence"
    for name in ("results.json", "pair.json"):
        if (args.root / "runs" / name).exists():
            evidence.mkdir(exist_ok=True)
            shutil.copy(args.root / "runs" / name, evidence / name)
    for summary in sorted((args.root / "runs").glob("*/summary.json")):
        target = evidence / "summaries" / summary.parent.name / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(summary, target)
    print(evidence)


if __name__ == "__main__":
    main()
