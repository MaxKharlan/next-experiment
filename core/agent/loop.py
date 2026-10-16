"""B17: the investigation loop, first version.

0. With a Tavily connector, the known-issues step searches for the model, recipe
   and symptom, and the sources go into the case.
1. Super proposes hypotheses for the case.
2. While the leader is below the threshold, the planner picks the probe with the
   highest information gain per dollar, the code runs it and updates the
   probabilities by Bayes, and Super writes the probe card.
3. The leader gets a causal check: make_pair, run_pair (local mini-jobs or Token
   Factory), causal_check. A leader that is not confirmed leaves the list and the
   loop goes on.
4. Ultra writes the verdict for a confirmed hypothesis.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from core.agent.context import InvestigationContext
from core.agent.known_issues import find_known_issues
from core.agent.router import Router
from core.agent.schemas import Hypothesis, Verdict
from core.agent.steps import interpret_probe, propose_hypotheses, write_verdict
from core.connectors.storage import Storage
from core.connectors.tavily import Tavily, TavilyBudgetError
from core.interventions.pairs import BUILDERS, Intervention, make_pair
from core.interventions.stats import causal_check
from core.planner.eig import best_probe, expected_gain, update
from core.planner.stopping import next_action
from core.probes.base import Probe
from core.probes.level0.recipe import LEVEL0

# trains both arms of a pair and fills their metrics, one value per seed
RunPair = Callable[[Intervention, Intervention], None]


@dataclass
class InvestigationResult:
    investigation_id: str
    status: str  # confirmed, inconclusive
    hypotheses: list[Hypothesis]
    steps: list[dict[str, Any]] = field(default_factory=list)
    verdict: Verdict | None = None
    llm_cost_usd: float = 0.0
    llm_calls: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        data = {
            "investigation_id": self.investigation_id,
            "status": self.status,
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "steps": self.steps,
            "verdict": self.verdict.model_dump() if self.verdict else None,
            "llm_cost_usd": self.llm_cost_usd,
            "llm_calls": self.llm_calls,
        }
        return json.dumps(data, indent=2)

    def save(self, storage: Storage) -> str:
        key = f"investigations/{self.investigation_id}/report.json"
        return storage.put_text(key, self.to_json())


def _rounded(probs: dict[str, float]) -> dict[str, float]:
    return {h: round(p, 4) for h, p in probs.items()}


def investigate(
    ctx: InvestigationContext,
    case: dict[str, Any],
    router: Router,
    run_pair: RunPair,
    *,
    probes: Sequence[Probe] = LEVEL0,
    threshold: float = 0.6,
    max_interventions: int = 2,
    min_effect: float = 0.03,
    noise_sd: float | None = 0.0146,
    intervention_cost_usd: float = 0.0,
    explain: bool = True,
    tavily: Tavily | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> InvestigationResult:
    """Run the loop; on_step gets every step as soon as it is recorded."""
    steps: list[dict[str, Any]] = []

    def add(step: dict[str, Any]) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    if tavily is not None:
        job = case.get(
            "job", {"hyperparameters": ctx.job_config.get("hyperparameters")}
        )
        try:
            found = find_known_issues(
                tavily, ctx.investigation_id, job, case.get("symptom", "")
            )
            case = {
                **case,
                "known_issues": [
                    {k: s[k] for k in ("title", "url", "snippet")}
                    for s in found["sources"]
                ],
            }
            add({"kind": "known_issues", **found})
        except (TavilyBudgetError, httpx.HTTPError) as err:
            add({"kind": "known_issues", "error": str(err)})
    hypotheses = propose_hypotheses(router, case, mode=ctx.mode)
    by_id = {h.id: h for h in hypotheses}
    classes = {h.id: h.fault_class for h in hypotheses}
    probs = {h.id: h.prior for h in hypotheses}
    rest = 1 - sum(probs.values())
    if rest > 1e-9:  # causes outside the list keep their share of the mass
        probs["unlisted"], classes["unlisted"] = rest, "unlisted"
    testable = {h.id for h in hypotheses if h.fault_class in BUILDERS}
    remaining = {p.name: p for p in probes}
    base_hp = dict(ctx.job_config.get("hyperparameters", {}))
    evidence: list[str] = []
    result = InvestigationResult(
        ctx.investigation_id, "inconclusive", hypotheses, steps=steps
    )
    add(
        {
            "kind": "hypotheses",
            "hypotheses": [h.model_dump() for h in hypotheses],
            "probs": _rounded(probs),
        }
    )
    interventions = 0

    while True:
        costs = {name: p.estimate_cost(ctx) for name, p in remaining.items()}
        probe_name, _ = best_probe(probs, classes, costs)
        gain = expected_gain(probs, classes, probe_name) if probe_name else 0.0
        budget = ctx.budget_left_usd - router.log.cost_usd
        action, why = next_action(
            probs,
            threshold=threshold,
            gain_bits=gain,
            budget_left_usd=budget,
            intervention_cost_usd=intervention_cost_usd,
            testable=testable,
        )
        if action == "stop":
            add({"kind": "stop", "reason": why})
            break

        if action == "probe":
            probe = remaining.pop(probe_name)
            run = probe.run(ctx)
            probs = update(probs, classes, probe_name, run.outcome)
            step = {
                "kind": "probe",
                "probe": probe_name,
                "outcome": run.outcome,
                "gain_bits": round(gain, 4),
                "raw": json.loads(run.summary),
                "probs": _rounded(probs),
            }
            if explain:
                card = interpret_probe(
                    router, hypotheses, probe_name, run.outcome, step["raw"]
                )
                step["card"] = card.explanation
            add(step)
            evidence.append(f"probe:{probe_name}")
            continue

        leader = by_id[max(testable, key=probs.get)]
        target, control = make_pair(leader, ctx)
        run_pair(target, control)
        check = causal_check(
            target.metrics, control.metrics, min_effect=min_effect, noise_sd=noise_sd
        )
        interventions += 1
        ctx.budget_left_usd -= intervention_cost_usd
        evidence.append(f"intervention:{leader.id}")
        add(
            {
                "kind": "intervention",
                "hypothesis": leader.id,
                "reason": why,
                "target": asdict(target),
                "control": asdict(control),
                "result": asdict(check),
            }
        )
        if check.status == "confirmed":
            result.verdict = write_verdict(
                router, leader, target, control, check, base_hp, list(evidence)
            )
            result.status = "confirmed"
            add({"kind": "verdict", **result.verdict.model_dump()})
            break
        # not confirmed: drop the leader and spread its mass over the rest
        testable.discard(leader.id)
        probs[leader.id] = 0.0
        total = sum(probs.values())
        probs = {h: p / total for h, p in probs.items()} if total else probs
        if interventions >= max_interventions:
            add({"kind": "stop", "reason": "intervention limit"})
            break

    result.llm_cost_usd = router.log.cost_usd
    result.llm_calls = router.log.totals("model")
    return result
