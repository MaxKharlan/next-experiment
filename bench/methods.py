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
from core.probes.weight_cache import investigation_probes

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
    eval_path: str | None = None,
) -> RunPair:
    """Pairs as Token Factory jobs on the case's files, scored on eval_path, the
    case's validation file by default."""
    from jobs.eval_local.exact_match import evaluate_adapter

    scored_on = eval_path or case.valid

    def run_pair(target: Intervention, control: Intervention) -> None:
        digest = hashlib.sha1(pair_key(target.patch, control.patch).encode())
        TokenFactoryPairRunner(
            tf=tf,
            storage=storage,
            model=case.model,
            train_path=Path(case.train),
            valid_path=Path(case.valid),
            base_hp=case.hyperparameters,
            evaluate=lambda adapter: evaluate_adapter(adapter, scored_on),
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
    noise_sd: float | None = None,
    min_effect: float | None = None,
) -> Diagnosis:
    """noise_sd and min_effect replace the methods' defaults in every causal check."""
    checks = {
        k: v
        for k, v in (("noise_sd", noise_sd), ("min_effect", min_effect))
        if v is not None
    }
    ctx = InvestigationContext(
        investigation_id=f"bench-{case.id}-{method}",
        mode="A",
        job_id=case.job_id,
        job_config={"model": case.model, **case.job_config()},
        budget_left_usd=1.0,
    )
    if method == "single_ultra":
        return diagnose_single(router, case.case)
    probes = investigation_probes(ctx)
    if method == "react":
        return diagnose_react(
            ctx,
            case.case,
            router,
            run_pair,
            pair_cost_usd=pair_cost_usd,
            probes=probes,
            **checks,
        )
    if method == "checklist":
        return diagnose_checklist(
            ctx, run_pair, pair_cost_usd=pair_cost_usd, probes=probes, **checks
        )
    if method == "next_experiment":
        result = investigate(
            ctx,
            case.case,
            router,
            run_pair,
            intervention_cost_usd=pair_cost_usd,
            probes=probes,
            tavily=tavily,
            **checks,
        )
        return from_investigation(result)
    raise ValueError(f"unknown method {method!r}")
