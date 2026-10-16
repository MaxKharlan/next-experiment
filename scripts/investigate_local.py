"""First end-to-end investigation: fault 6 on a local Qwen3-0.6B job.

The failed job is the one from scripts/smoke_agent_steps.py: 200 BANKING77 examples,
learning rate 1e-5, 3 epochs. Super proposes hypotheses, the planner runs level-0
probes, the leader gets a causal check as two local mini-jobs per seed on the
job's own data (about 10 minutes on an M3), and Ultra writes the verdict. The trace
goes to the terminal and the report to
artifacts/investigations/<id>/report.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.agent.context import InvestigationContext
from core.agent.loop import investigate
from core.agent.router import Router
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import Intervention
from core.telemetry.calls import CallLog
from jobs.kaggle.run_pair import train_arm
from scripts.smoke_agent_steps import CASE, JOB_HP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="artifacts/b6/train.jsonl")
    parser.add_argument("--valid", default="artifacts/b6/valid.jsonl")
    parser.add_argument("--id", default="local-fault6")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    runs = Path("artifacts/investigations") / args.id / "runs"
    ctx = InvestigationContext(
        investigation_id=args.id,
        mode="A",
        job_id="local",
        job_config={"hyperparameters": JOB_HP, "checkpoints": CASE["checkpoints"]},
        budget_left_usd=1.0,
    )
    router = Router(TokenFactory.from_settings(), CallLog(args.id))

    def run_pair(target: Intervention, control: Intervention) -> None:
        out = runs / target.hypothesis_id
        for arm in (target, control):
            train_arm(arm, JOB_HP, args.train, args.valid, out, args.seeds)

    result = investigate(ctx, CASE, router, run_pair)

    print("\nTrace:")
    for step in result.steps:
        if step["kind"] == "hypotheses":
            for h in result.hypotheses:
                print(f"  {h.id} {h.fault_class} prior {h.prior:.2f}")
        elif step["kind"] == "probe":
            print(f"  probe {step['probe']} -> {step['outcome']}: {step['probs']}")
            if "card" in step:
                print(f"    {step['card']}")
        elif step["kind"] == "intervention":
            r = step["result"]
            print(
                f"  causal check of {step['hypothesis']}: {r['status']}, "
                f"effect {r['effect']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            )
        else:
            print(f"  stop: {step['reason']}")
    print(f"\nStatus: {result.status}")
    if result.verdict:
        print(f"Verdict: {result.verdict.summary}")
        print(result.verdict.patch)
    print(f"LLM cost ${result.llm_cost_usd:.4f}")
    print(f"Report: {result.save(LocalStorage('artifacts'))}")


if __name__ == "__main__":
    main()
