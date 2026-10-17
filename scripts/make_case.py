"""Turn finished Token Factory jobs into benchmark cases.

Reads a job's hyperparameters and the train_loss and valid_loss of each checkpoint,
scores the last checkpoint on the held-out set on this MacBook and writes
bench/cases/<split>/<id>.json. The symptom quotes the measured exact match and a few
wrong answers, so a healthy job comes with a complaint too: the methods have to tell
a fault from a hard task.

One job: make_case <job id> --id <case id> [--truth <fault class>...]; leave out
--truth for a healthy job. A whole fault sweep: make_case --from-sweep
<dir>/results.jsonl (scripts/tf_fault_sweep.py), which names the cases after the
fault number and strength. A case must be clean on either side of the task's
reference, or it is reported and not saved:
- a fault that leaves the held-out metric within the tolerance of the reference does
  not hurt on this task, so the job is not a failure to diagnose;
- a healthy job that falls below the reference by more than the tolerance is held
  back by something the catalog does not name, so calling it healthy would count a
  real improvement as a false alarm.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from bench.cases import BenchCase
from bench.fixes import TOLERANCE, reference_metric
from core.agent.catalog import FAULTS
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


# (adapter directory, data file, list for wrong answers) -> exact match
Score = Callable[[Path, str, list[dict[str, str]]], float]


def build_case(
    tf: TokenFactory,
    storage: LocalStorage,
    score: Score,
    job_id: str,
    case_id: str,
    truth: list[str],
    args: argparse.Namespace,
) -> BenchCase:
    job = tf.job(job_id)
    if job.status != "succeeded":
        raise ValueError(f"{job_id} is {job.status}")
    hp = job.hyperparameters.model_dump(exclude_none=True)
    hp = {k: v for k, v in hp.items() if k not in TF_LORA}
    adapter = download_last_checkpoint(tf, storage, job_id, f"bench/cases/{case_id}")
    wrong: list[dict[str, str]] = []
    metric = score(adapter, args.heldout, wrong)
    shown = wrong[: args.errors]
    return BenchCase(
        id=case_id,
        split=args.split,
        task=args.task,
        model=job.model,
        truth=truth,
        hyperparameters=hp,
        case={
            "symptom": symptom(metric, count_rows(Path(args.heldout)), shown),
            "task": TASKS[args.task]["task"],
            "job": {
                "model": job.model,
                "status": job.status,
                "train_examples": count_rows(Path(args.train)),
                "validation_examples": count_rows(Path(args.valid)),
                "hyperparameters": hp,
            },
            "checkpoints": checkpoints(tf, job_id),
            "findings": TASKS[args.task]["findings"],
            "wrong_answers": shown,
        },
        train=args.train,
        valid=args.valid,
        heldout=args.heldout,
        job_id=job_id,
        heldout_metric=metric,
        reports=args.report,
    )


def sweep_cases(
    rows: list[dict[str, Any]], prefix: str = ""
) -> list[tuple[str, str, list[str]]]:
    """(case ID, job ID, truth) for each finished run of tf_fault_sweep.py."""
    numbers = {f.fault_class: f.number for f in FAULTS}
    out = []
    for r in rows:
        if r["status"] != "succeeded":
            continue
        if r["fault"] is None:
            name, truth = "healthy", []
        else:
            level = r["run"].split("-")[1]  # e.g. lr_too_low-mild-s0
            name, truth = f"fault{numbers[r['fault']]}-{level}", [r["fault"]]
        out.append((f"{prefix}{name}-s{r['seed']}", r["job_id"], truth))
    return out


def case_problem(case: BenchCase, reference: float | None) -> str | None:
    """Why the case is not clean, or None: a fault must fall below the reference by
    more than the tolerance, and a healthy job must not."""
    if reference is None:
        return None
    below = case.heldout_metric < reference - TOLERANCE
    if not case.healthy and not below:
        return "the fault does not hurt on this task"
    if case.healthy and below:
        return "the healthy job is well below the reference, so it is not clean"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--id", help="case ID for one job, e.g. fault6-lr1e-5")
    parser.add_argument("--truth", nargs="*", default=[], choices=FAULT_CLASSES)
    parser.add_argument("--from-sweep", help="results.jsonl of tf_fault_sweep.py")
    parser.add_argument("--prefix", default="", help="for sweep case IDs, e.g. ref-")
    parser.add_argument("--split", default="dev", choices=["dev", "heldout"])
    parser.add_argument("--task", default="banking77", choices=TASKS)
    parser.add_argument("--train", default="artifacts/b6/train.jsonl")
    parser.add_argument("--valid", default="artifacts/b6/valid.jsonl")
    parser.add_argument("--heldout", default="artifacts/b18/heldout.jsonl")
    parser.add_argument("--errors", type=int, default=5, help="wrong answers to show")
    parser.add_argument("--report", action="append", default=[])
    args = parser.parse_args()
    if args.from_sweep:
        lines = Path(args.from_sweep).read_text(encoding="utf-8").splitlines()
        todo = sweep_cases([json.loads(x) for x in lines if x], args.prefix)
    elif args.job_id and args.id:
        todo = [(args.id, args.job_id, args.truth)]
    else:
        parser.error("give a job ID with --id, or --from-sweep")

    from jobs.eval_local.exact_match import evaluate_adapter

    def score(adapter: Path, data: str, wrong: list[dict[str, str]]) -> float:
        return evaluate_adapter(adapter, data, wrong=wrong)

    tf, storage = TokenFactory.from_settings(), LocalStorage("artifacts")
    for case_id, job_id, truth in todo:
        case = build_case(tf, storage, score, job_id, case_id, truth, args)
        reference = reference_metric(case.task, case.model)
        line = f"{case_id}: exact match {case.heldout_metric:.3f}"
        if reference is not None:
            line += f", reference {reference:.3f}"
        problem = case_problem(case, reference)
        if problem:
            print(f"{line}: {problem}, not saved")
            continue
        print(f"{line}. Case: {case.save()}")


if __name__ == "__main__":
    main()
