"""B19: the agent and the comparison methods on the fault-6 case.

- single_ultra: one Ultra call with the whole case and no training, about one cent.
- react: Super picks probes and causal checks itself, without the planner.
- checklist: every level-0 probe, then a causal check for each recipe fault.
- next_experiment: the agent, as in scripts/investigate_tf.py.

Causal checks go through a cache that all methods share
(artifacts/bench/fault6/pairs.json). It starts with the pairs of the 2026-10-16
investigation, read from its report. A pair that is not in the cache trains on
Token Factory: 4 Qwen3-0.6B jobs, about $0.06. With --no-train such a pair stops
that method instead. Each Diagnosis goes to artifacts/bench/fault6/<method>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bench.baselines.checklist import diagnose_checklist
from bench.baselines.react_super import diagnose_react
from bench.baselines.single_ultra import diagnose_single
from bench.diagnosis import Diagnosis, from_investigation
from bench.pair_cache import PairCache, PairNotCached, pair_key
from core.agent.context import InvestigationContext
from core.agent.loop import investigate
from core.agent.router import Router
from core.config import get_settings
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import Intervention
from core.interventions.tf_runner import TokenFactoryPairRunner
from core.planner.job_cost import estimate_job
from core.telemetry.calls import CallLog
from scripts.smoke_agent_steps import CASE, JOB_HP

METHODS = ("single_ultra", "react", "checklist", "next_experiment")
MODEL = "Qwen/Qwen3-0.6B"
TRAIN, VALID = Path("artifacts/b6/train.jsonl"), Path("artifacts/b6/valid.jsonl")


def seed_cache(cache: PairCache, report_path: Path) -> int:
    """Add the pairs of a saved investigation that the cache does not have."""
    if not report_path.exists():
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    known = cache.load()
    added = 0
    for step in report["steps"]:
        if step["kind"] != "intervention":
            continue
        target = Intervention(**step["target"])
        control = Intervention(**step["control"])
        if pair_key(target.patch, control.patch) not in known:
            cache.add(target, control)
            added += 1
    return added


def tf_pairs(tf: TokenFactory, storage: LocalStorage, seeds: list[int]):
    from jobs.eval_local.exact_match import evaluate_adapter

    def run_pair(target: Intervention, control: Intervention) -> None:
        digest = hashlib.sha1(pair_key(target.patch, control.patch).encode())
        TokenFactoryPairRunner(
            tf=tf,
            storage=storage,
            model=MODEL,
            train_path=TRAIN,
            valid_path=VALID,
            base_hp=JOB_HP,
            evaluate=lambda adapter: evaluate_adapter(adapter, VALID),
            prefix=f"bench/fault6/tf/{digest.hexdigest()[:10]}",
            seeds=seeds,
        )(target, control)

    return run_pair


def run_method(
    method: str,
    tf: TokenFactory,
    run_pair: PairCache,
    pair_cost: float,
    tavily: Tavily | None,
) -> Diagnosis:
    router = Router(tf, CallLog(f"bench-fault6-{method}"))
    ctx = InvestigationContext(
        investigation_id=f"bench-fault6-{method}",
        mode="A",
        job_id="local",
        job_config={"hyperparameters": JOB_HP, "checkpoints": CASE["checkpoints"]},
        budget_left_usd=1.0,
    )
    if method == "single_ultra":
        return diagnose_single(router, CASE)
    if method == "react":
        return diagnose_react(ctx, CASE, router, run_pair, pair_cost_usd=pair_cost)
    if method == "checklist":
        return diagnose_checklist(ctx, run_pair, pair_cost_usd=pair_cost)
    result = investigate(
        ctx, CASE, router, run_pair, intervention_cost_usd=pair_cost, tavily=tavily
    )
    return from_investigation(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--report", default="artifacts/investigations/tf-fault6/report.json"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    storage = LocalStorage("artifacts")
    tf = TokenFactory.from_settings()
    cache = PairCache(
        storage,
        "bench/fault6/pairs.json",
        None if args.no_train else tf_pairs(tf, storage, args.seeds),
    )
    print(f"pairs from {args.report}: {seed_cache(cache, Path(args.report))} added")
    examples = sum(1 for line in TRAIN.read_text(encoding="utf-8").splitlines() if line)
    one_job = estimate_job(MODEL, examples, 57, JOB_HP["n_epochs"])
    pair_cost = 2 * len(args.seeds) * one_job.cost_usd
    tavily = Tavily.from_settings(storage) if get_settings().tavily_api_key else None

    for method in args.methods:
        try:
            d = run_method(method, tf, cache, pair_cost, tavily)
        except PairNotCached as err:
            print(f"{method}: stopped, {err} is not cached; run without --no-train")
            continue
        storage.put_text(f"bench/fault6/{method}.json", json.dumps(asdict(d), indent=2))
        top = ", ".join(f"{c} {p:.2f}" for c, p in d.ranking[:3])
        print(
            f"{method}: {d.status}, cause {d.cause}; top 3: {top}; "
            f"LLM ${d.llm_cost_usd:.4f}, {d.mini_jobs} jobs, "
            f"training about ${d.training_cost_usd:.3f}"
        )


if __name__ == "__main__":
    main()
