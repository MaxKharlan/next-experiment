"""The agent and the comparison methods behind one call, for any benchmark case.

Every method gets the same case, the same tools and the same budget, and returns a
Diagnosis. Causal checks train on Token Factory with the case's model, files and
recipe; the caller wraps them in a PairCache shared by the methods.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bench.baselines.checklist import diagnose_checklist
from bench.baselines.react_super import diagnose_react
from bench.baselines.single_ultra import diagnose_single
from bench.cases import BenchCase
from bench.diagnosis import Diagnosis, from_investigation
from bench.pair_cache import pair_key
from core.agent.context import InvestigationContext
from core.agent.loop import RunPair, investigate
from core.agent.router import Router
from core.connectors.storage import LocalStorage
from core.connectors.tavily import Tavily
from core.connectors.token_factory import TokenFactory
from core.interventions.pairs import Intervention
from core.interventions.tf_runner import TokenFactoryPairRunner
from core.planner.job_cost import estimate_job

METHODS = ("single_ultra", "react", "checklist", "next_experiment")


def pair_cost(case: BenchCase, seeds: int = 2) -> float:
    """Price of one causal check: both arms, one job per seed, at the case's size."""
    one = estimate_job(
        case.model,
        case.train_examples,
        case.tokens_per_example,
        case.hyperparameters["n_epochs"],
    )
    return 2 * seeds * one.cost_usd


def tf_pair_runner(
    tf: TokenFactory,
    storage: LocalStorage,
    case: BenchCase,
    seeds: list[int],
    prefix: str,
) -> RunPair:
    """Pairs as Token Factory jobs on the case's files, scored on its validation file."""
    from jobs.eval_local.exact_match import evaluate_adapter

    def run_pair(target: Intervention, control: Intervention) -> None:
        digest = hashlib.sha1(pair_key(target.patch, control.patch).encode())
        TokenFactoryPairRunner(
            tf=tf,
            storage=storage,
            model=case.model,
            train_path=Path(case.train),
            valid_path=Path(case.valid),
            base_hp=case.hyperparameters,
            evaluate=lambda adapter: evaluate_adapter(adapter, case.valid),
            prefix=f"{prefix}/{digest.hexdigest()[:10]}",
            seeds=seeds,
        )(target, control)

    return run_pair


def run_method(
    method: str,
    case: BenchCase,
    router: Router,
    run_pair: RunPair,
    *,
    pair_cost_usd: float,
    tavily: Tavily | None = None,
) -> Diagnosis:
    ctx = InvestigationContext(
        investigation_id=f"bench-{case.id}-{method}",
        mode="A",
        job_id=case.job_id,
        job_config=case.job_config(),
        budget_left_usd=1.0,
    )
    if method == "single_ultra":
        return diagnose_single(router, case.case)
    if method == "react":
        return diagnose_react(
            ctx, case.case, router, run_pair, pair_cost_usd=pair_cost_usd
        )
    if method == "checklist":
        return diagnose_checklist(ctx, run_pair, pair_cost_usd=pair_cost_usd)
    if method == "next_experiment":
        result = investigate(
            ctx,
            case.case,
            router,
            run_pair,
            intervention_cost_usd=pair_cost_usd,
            tavily=tavily,
        )
        return from_investigation(result)
    raise ValueError(f"unknown method {method!r}")
