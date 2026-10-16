"""Turn a finished Token Factory job into a benchmark case.

Reads the job's hyperparameters and the train_loss and valid_loss of each checkpoint,
scores the last checkpoint on the held-out set on this MacBook and writes
bench/cases/<split>/<id>.json. --truth names the injected faults; leave it out for a
healthy job. The symptom quotes the measured exact match and a few wrong answers,
so a healthy job comes with a complaint too: the methods have to tell a fault from
a hard task.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, get_args

from bench.cases import BenchCase
from core.agent.schemas import FaultClass
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.tf_runner import TF_LORA, download_last_checkpoint
from scripts.retrain_tf import count_rows

TASKS: dict[str, dict[str, Any]] = {
    "banking77": {
        "task": "classify banking queries into 77 intent labels, answer with the label",
        "findings": [
            "examples are 45 to 103 tokens long, far below context_length",
            "no duplicates between train and validation",
        ],
    },
}
FAULT_CLASSES = get_args(FaultClass)


def checkpoints(tf: TokenFactory, job_id: str) -> list[dict[str, Any]]:
    found = sorted(tf.checkpoints(job_id), key=lambda c: c.step_number)
    return [
        {
            "epoch": i,
            "train_loss": round(c.metrics.train_loss, 4),
            "valid_loss": round(c.metrics.valid_loss, 4),
        }
        for i, c in enumerate(found, start=1)
    ]


def symptom(metric: float, examples: int, wrong: list[dict[str, str]]) -> str:
    return (
        f"After fine-tuning the model gives the right label for {metric:.0%} of "
        f"{examples} held-out queries. We expected much better. "
        f"{len(wrong)} of the wrong answers are attached."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--id", required=True, help="e.g. fault6-lr1e-5")
    parser.add_argument("--truth", nargs="*", default=[], choices=FAULT_CLASSES)
    parser.add_argument("--split", default="dev", choices=["dev", "heldout"])
    parser.add_argument("--task", default="banking77", choices=TASKS)
    parser.add_argument("--train", default="artifacts/b6/train.jsonl")
    parser.add_argument("--valid", default="artifacts/b6/valid.jsonl")
    parser.add_argument("--heldout", default="artifacts/b18/heldout.jsonl")
    parser.add_argument("--errors", type=int, default=5, help="wrong answers to show")
    parser.add_argument("--report", action="append", default=[])
    args = parser.parse_args()

    from jobs.eval_local.exact_match import evaluate_adapter

    tf, storage = TokenFactory.from_settings(), LocalStorage("artifacts")
    job = tf.job(args.job_id)
    if job.status != "succeeded":
        raise SystemExit(f"{args.job_id} is {job.status}")
    hp = job.hyperparameters.model_dump(exclude_none=True)
    hp = {k: v for k, v in hp.items() if k not in TF_LORA}
    adapter = download_last_checkpoint(
        tf, storage, args.job_id, f"bench/cases/{args.id}"
    )
    wrong: list[dict[str, str]] = []
    metric = evaluate_adapter(adapter, args.heldout, wrong=wrong)
    examples = count_rows(Path(args.heldout))
    train_examples = count_rows(Path(args.train))
    valid_examples = count_rows(Path(args.valid))
    shown = wrong[: args.errors]
    case = BenchCase(
        id=args.id,
        split=args.split,
        task=args.task,
        model=job.model,
        truth=args.truth,
        hyperparameters=hp,
        case={
            "symptom": symptom(metric, examples, shown),
            "task": TASKS[args.task]["task"],
            "job": {
                "model": job.model,
                "status": job.status,
                "train_examples": train_examples,
                "validation_examples": valid_examples,
                "hyperparameters": hp,
            },
            "checkpoints": checkpoints(tf, args.job_id),
            "findings": TASKS[args.task]["findings"],
            "wrong_answers": shown,
        },
        train=args.train,
        valid=args.valid,
        heldout=args.heldout,
        job_id=args.job_id,
        heldout_metric=metric,
        reports=args.report,
    )
    print(f"{args.id}: exact match {metric:.3f} on {examples} held-out queries")
    print(f"Case: {case.save()}")


if __name__ == "__main__":
    main()
