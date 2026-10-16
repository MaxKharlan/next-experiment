"""End-to-end investigation of the fault-6 case with causal checks on Token Factory.

Same case and loop as scripts/investigate_local.py, but each arm of a causal check
trains as a Post-training job on Token Factory. The adapter of the last checkpoint
is downloaded to artifacts/ and scored by exact match on this MacBook. For
Qwen3-0.6B with 200 examples and 3 epochs a pair with 2 seeds is 4 jobs, about $0.06
and 10 to 20 minutes when Token Factory has free capacity. A confirmed fix ends with
the proposal of the full retrain; scripts/retrain_tf.py runs it after approval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from core.agent.context import InvestigationContext
from core.agent.loop import investigate
from core.agent.router import Router
from core.config import get_settings
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import Intervention
from core.interventions.retrain import RetrainPlan
from core.interventions.tf_runner import TokenFactoryPairRunner
from core.planner.job_cost import estimate_job
from core.telemetry.calls import CallLog
from jobs.eval_local.exact_match import evaluate_adapter
from scripts.retrain_tf import describe, propose
from scripts.smoke_agent_steps import CASE, JOB_HP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="artifacts/b6/train.jsonl")
    parser.add_argument("--valid", default="artifacts/b6/valid.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--id", default="tf-fault6")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    # chat template tokens per BANKING77 example for Qwen3 (docs/b6-post-training.md)
    parser.add_argument("--tokens-per-example", type=float, default=57)
    args = parser.parse_args()

    storage = LocalStorage("artifacts")
    tf = TokenFactory.from_settings()
    router = Router(tf, CallLog(args.id))
    lines = Path(args.train).read_text(encoding="utf-8").splitlines()
    examples = sum(1 for line in lines if line.strip())
    one_job = estimate_job(
        args.model, examples, args.tokens_per_example, JOB_HP["n_epochs"]
    )
    pair_cost = 2 * len(args.seeds) * one_job.cost_usd
    print(f"one causal check: {2 * len(args.seeds)} jobs, about ${pair_cost:.3f}")

    ctx = InvestigationContext(
        investigation_id=args.id,
        mode="A",
        job_id="local",
        job_config={"hyperparameters": JOB_HP, "checkpoints": CASE["checkpoints"]},
        budget_left_usd=1.0,
    )

    def run_pair(target: Intervention, control: Intervention) -> None:
        TokenFactoryPairRunner(
            tf=tf,
            storage=storage,
            model=args.model,
            train_path=Path(args.train),
            valid_path=Path(args.valid),
            base_hp=JOB_HP,
            evaluate=lambda adapter: evaluate_adapter(adapter, args.valid),
            prefix=f"investigations/{args.id}/tf/{target.hypothesis_id}",
            seeds=args.seeds,
        )(target, control)

    tavily = Tavily.from_settings(storage) if get_settings().tavily_api_key else None
    if tavily is None:
        print("TAVILY_API_KEY is not set: no known-issues step")
    result = investigate(
        ctx,
        CASE,
        router,
        run_pair,
        intervention_cost_usd=pair_cost,
        tavily=tavily,
        propose_retrain=lambda target, control: asdict(
            propose(target, control, model=args.model, train=Path(args.train))
        ),
    )
    for step in result.steps:
        if step["kind"] == "known_issues":
            for source in step.get("sources", []):
                print(f"known issue: {source['title']} — {source['url']}")
            if "error" in step:
                print(f"known issues failed: {step['error']}")
        elif step["kind"] == "probe":
            print(f"probe {step['probe']} -> {step['outcome']}: {step['probs']}")
        elif step["kind"] == "intervention":
            r = step["result"]
            print(
                f"causal check of {step['hypothesis']}: {r['status']}, "
                f"effect {r['effect']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], "
                f"jobs {step['target']['ft_job_ids'] + step['control']['ft_job_ids']}"
            )
        elif step["kind"] == "stop":
            print(f"stop: {step['reason']}")
    print(f"Status: {result.status}")
    if result.verdict:
        print(f"Verdict: {result.verdict.summary}")
    print(
        f"LLM cost ${result.llm_cost_usd:.4f}, training about ${pair_cost:.3f} a pair"
    )
    report = result.save(storage)
    print(f"Report: {report}")
    if result.retrain_plan:
        print(describe(RetrainPlan(**result.retrain_plan)))
        print(
            f"Approve and run it: python -m scripts.retrain_tf --report artifacts/{report}"
        )


if __name__ == "__main__":
    main()
