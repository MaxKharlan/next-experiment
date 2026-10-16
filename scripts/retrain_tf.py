"""B18: retrain a confirmed investigation on Token Factory after the user approves.

Reads the report that scripts/investigate_tf.py saved, shows the retrain of the
fault-6 case with its price and asks before starting it. One Post-training job trains
the whole training file with the fix. Then the base model, the failed job and the
retrained job answer the held-out set on this MacBook. The steps go into the report,
and with --demo into a saved investigation that the screen replays.

Make the held-out set first with scripts/make_heldout.py. Without it the comparison
uses the validation file and says so.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import Intervention
from core.interventions.retrain import (
    Retrainer,
    RetrainPlan,
    failed_job,
    plan_retrain,
)
from core.planner.job_cost import estimate_job
from scripts.smoke_agent_steps import JOB_HP

RETRAIN_KINDS = {"retrain_proposal", "retrain", "comparison", "retrain_declined"}
TRAIN, VALID = Path("artifacts/b6/train.jsonl"), Path("artifacts/b6/valid.jsonl")
HELDOUT = Path("artifacts/b18/heldout.jsonl")
MODEL = "Qwen/Qwen3-0.6B"
# chat template tokens per BANKING77 example for Qwen3 (docs/b6-post-training.md)
TOKENS_PER_EXAMPLE = 57
# the Cyrillic "у" looks like "y" in a terminal, and "да" is yes
YES = {"y", "yes", "у", "д", "да"}


def count_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def confirmed_pair(report: dict[str, Any]) -> tuple[Intervention, Intervention]:
    for step in reversed(report["steps"]):
        if step["kind"] == "intervention" and step["result"]["status"] == "confirmed":
            return Intervention(**step["target"]), Intervention(**step["control"])
    raise ValueError("the report has no confirmed causal check")


def propose(
    target: Intervention,
    control: Intervention,
    job_id: str = "local",
    model: str = MODEL,
    train: Path = TRAIN,
) -> RetrainPlan:
    """The retrain of the fault-6 case for a confirmed pair."""
    return plan_retrain(
        target,
        JOB_HP,
        model,
        count_rows(train),
        TOKENS_PER_EXAMPLE,
        failed_job(job_id, control),
    )


def heldout_set(heldout: Path = HELDOUT, valid: Path = VALID) -> tuple[Path, str]:
    if heldout.exists():
        return heldout, ""
    return valid, (
        f"no held-out set at {heldout}: the validation file stands in, and the "
        "causal checks were scored on it too"
    )


def make_retrainer(
    tf: TokenFactory,
    storage: LocalStorage,
    investigation_id: str,
    train: Path = TRAIN,
    valid: Path = VALID,
    heldout: Path = HELDOUT,
) -> Retrainer:
    from jobs.eval_local.exact_match import evaluate_adapter, evaluate_base

    path, note = heldout_set(heldout, valid)
    return Retrainer(
        tf=tf,
        storage=storage,
        train_path=train,
        valid_path=valid,
        heldout_path=path,
        score_adapter=evaluate_adapter,
        score_base=evaluate_base,
        prefix=f"investigations/{investigation_id}/retrain",
        heldout_note=note,
    )


def approved(answer: str) -> bool:
    # a terminal can send a stray byte with the answer, e.g. "\udcd0y" when the
    # keyboard layout was switched; drop what is not text
    text = answer.encode("utf-8", "ignore").decode("utf-8")
    return text.strip().lower() in YES


def describe(plan: RetrainPlan) -> str:
    changes = ", ".join(
        f"{k} {plan.base_hyperparameters.get(k)} -> {v}" for k, v in plan.patch.items()
    )
    failed = (
        f"the failed model is job {plan.failed_job_id}"
        if plan.failed_job_id
        else "the failed recipe trains as a second job for the comparison"
    )
    return (
        f"Full retrain of {plan.hypothesis_id} on {plan.model}: "
        f"{plan.train_examples} examples, {changes}\n"
        f"{len(plan.jobs)} job(s), about {plan.tokens} tokens, ${plan.cost_usd:.3f} "
        f"and {plan.seconds / 60:.0f} min once a GPU is free; {failed}"
    )


def add_retrain_steps(
    report: dict[str, Any], plan: RetrainPlan, steps: list[dict[str, Any]]
) -> None:
    """Put the proposal, retrain and comparison steps at the end of a report."""
    kept = [s for s in report["steps"] if s["kind"] not in RETRAIN_KINDS]
    examples = plan.train_examples
    for step in kept:
        if step["kind"] == "intervention" and "cost_usd" not in step:
            # runs from before the loop recorded it: the estimate of the pair
            step["cost_usd"] = sum(
                len(arm["metrics"])
                * estimate_job(
                    plan.model,
                    round(examples * arm["patch"].get("data_fraction", 1.0)),
                    TOKENS_PER_EXAMPLE,
                    {**JOB_HP, **arm["patch"]}["n_epochs"],
                ).cost_usd
                for arm in (step["target"], step["control"])
            )
    proposal = {"kind": "retrain_proposal", **asdict(plan)}
    report["steps"] = [*kept, proposal, *steps]
    report["retrain_plan"] = asdict(plan)
    report["comparison"] = steps[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", default="artifacts/investigations/tf-fault6/report.json"
    )
    parser.add_argument("--heldout", type=Path, default=HELDOUT)
    parser.add_argument("--failed-job", help="the user's failed job, if there is one")
    parser.add_argument("--yes", action="store_true", help="approve without asking")
    parser.add_argument("--demo", help="also add the steps to this demo case file")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("retrain_plan"):
        plan = RetrainPlan(**report["retrain_plan"])
    else:
        plan = propose(*confirmed_pair(report), job_id=args.failed_job or "local")
    print(describe(plan))
    if not args.yes:
        answer = input("Start the retrain? [y/N] ")
        if not approved(answer):
            print(f"Not started: the answer was {answer!r}, not y.")
            return

    retrainer = make_retrainer(
        TokenFactory.from_settings(),
        LocalStorage("artifacts"),
        report["investigation_id"],
        heldout=args.heldout,
    )
    if retrainer.heldout_note:
        print(retrainer.heldout_note)
    steps: list[dict[str, Any]] = []
    comparison = retrainer(plan, on_step=steps.append)
    steps.append(comparison)
    print(f"Held-out exact match on {comparison['heldout_examples']} queries:")
    for m in comparison["models"]:
        print(f"  {m['name']:<7} {m['score']:.3f}  {m.get('job_id', m['label'])}")
    print(f"fixed against failed: {comparison['gain']:+.3f}")

    add_retrain_steps(report, plan, steps)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    if args.demo:
        demo_path = Path(args.demo)
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        add_retrain_steps(demo["report"], plan, steps)
        demo_path.write_text(
            json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Demo case: {demo_path}")


if __name__ == "__main__":
    main()
