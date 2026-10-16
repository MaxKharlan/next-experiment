"""The tools the comparison methods share with the agent: probes and causal checks.

Steps come out in the shape the agent's loop records, so bench/diagnosis.py counts
jobs and cost the same way for every method.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from core.agent.context import InvestigationContext
from core.agent.loop import RunPair
from core.agent.schemas import Hypothesis
from core.interventions.pairs import Intervention, job_hyperparameters, make_pair
from core.interventions.stats import causal_check
from core.probes.base import Probe


def probe_step(probe: Probe, ctx: InvestigationContext) -> dict[str, Any]:
    run = probe.run(ctx)
    return {
        "kind": "probe",
        "probe": probe.name,
        "outcome": run.outcome,
        "raw": json.loads(run.summary),
    }


def check_step(
    fault_class: str,
    ctx: InvestigationContext,
    run_pair: RunPair,
    *,
    min_effect: float,
    noise_sd: float | None,
    cost_usd: float,
) -> dict[str, Any]:
    """Train the pair for a fault class and test it, as the agent's loop does."""
    hypothesis = Hypothesis(
        id=fault_class, fault_class=fault_class, location="", prior=0.0, rationale=""
    )
    target, control = make_pair(hypothesis, ctx)
    run_pair(target, control)
    result = causal_check(
        target.metrics, control.metrics, min_effect=min_effect, noise_sd=noise_sd
    )
    return {
        "kind": "intervention",
        "hypothesis": fault_class,
        "target": asdict(target),
        "control": asdict(control),
        "result": asdict(result),
        "cost_usd": cost_usd,
    }


def patch_of(step: dict[str, Any], ctx: InvestigationContext) -> str:
    return Intervention(**step["target"]).diff(job_hyperparameters(ctx))
