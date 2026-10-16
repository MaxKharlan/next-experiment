"""Comparison method 1: every check in a fixed order and a diagnosis by the signature table.

All level-0 probes run first and update a uniform prior over the catalog classes of
the mode through the same signature table the agent uses. Then every fault class
that has a causal check gets one, in catalog order, whatever the probes said. A
confirmed check makes the status confirmed; with more than one, the largest effect
wins. A rejected class drops out of the ranking. No LLM is called.
"""

from __future__ import annotations

from collections.abc import Sequence

from bench.baselines.tools import check_step, patch_of, probe_step
from bench.diagnosis import Diagnosis, pair_jobs, ranked
from core.agent.catalog import FAULTS
from core.agent.context import InvestigationContext
from core.agent.loop import RunPair
from core.interventions.pairs import BUILDERS
from core.planner.eig import update
from core.probes.base import Probe
from core.probes.level0.recipe import LEVEL0


def diagnose_checklist(
    ctx: InvestigationContext,
    run_pair: RunPair,
    *,
    probes: Sequence[Probe] = LEVEL0,
    min_effect: float = 0.03,
    noise_sd: float | None = 0.0146,
    pair_cost_usd: float = 0.0,
    mode: str = "A",
) -> Diagnosis:
    order = [f.fault_class for f in FAULTS if mode in f.modes]
    probs = {c: 1 / len(order) for c in order}
    identity = {c: c for c in order}
    steps = []
    for probe in probes:
        step = probe_step(probe, ctx)
        probs = update(probs, identity, probe.name, step["outcome"])
        steps.append({**step, "probs": probs})

    checks = [
        check_step(
            fault_class,
            ctx,
            run_pair,
            min_effect=min_effect,
            noise_sd=noise_sd,
            cost_usd=pair_cost_usd,
        )
        for fault_class in order
        if fault_class in BUILDERS
    ]
    steps += checks
    for check in checks:
        if check["result"]["status"] == "rejected":
            probs[check["hypothesis"]] = 0.0
    total = sum(probs.values())
    confirmed = sorted(
        (c for c in checks if c["result"]["status"] == "confirmed"),
        key=lambda c: c["result"]["effect"],
        reverse=True,
    )
    return Diagnosis(
        method="checklist",
        ranking=ranked((c, p / total) for c, p in probs.items() if p > 0),
        status="confirmed" if confirmed else "inconclusive",
        cause=confirmed[0]["hypothesis"] if confirmed else None,
        patch=patch_of(confirmed[0], ctx) if confirmed else "",
        mini_jobs=sum(pair_jobs(c) for c in checks),
        training_cost_usd=pair_cost_usd * len(checks),
        steps=steps,
    )
