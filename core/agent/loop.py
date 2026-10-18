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
5. With propose_retrain, the confirmed fix comes with a proposal of the full retrain
   and its price. The loop stops there: the retrain is a level-3 step and starts
   only after the user approves it (core/interventions/retrain.py).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from core.agent.context import InvestigationContext
from core.agent.ensemble import sample_hypotheses
from core.agent.known_issues import find_known_issues
from core.agent.label_checks import check_training_labels
from core.agent.router import Router
from core.agent.sandbox_check import generated_check
from core.agent.schemas import Hypothesis, Verdict
from core.agent.steps import interpret_probe, propose_hypotheses, write_verdict
from core.connectors.storage import Storage
from core.connectors.tavily import Tavily, TavilyBudgetError
from core.interventions.pairs import BUILDERS, Intervention, make_pair
from core.interventions.stats import causal_check
from core.planner.causal_belief import MODEL, update_causal
from core.planner.eig import best_probe, expected_gain, update
from core.planner.stopping import next_action
from core.probes.base import Probe
from core.probes.level0.recipe import LEVEL0

# trains both arms of a pair and fills their metrics, one value per seed
RunPair = Callable[[Intervention, Intervention], None]
# (target, control) of the confirmed check -> the retrain plan to propose, or None
ProposeRetrain = Callable[[Intervention, Intervention], dict[str, Any] | None]


@dataclass
class InvestigationResult:
    investigation_id: str
    status: str  # confirmed, inconclusive
    hypotheses: list[Hypothesis]
    steps: list[dict[str, Any]] = field(default_factory=list)
    verdict: Verdict | None = None
    llm_cost_usd: float = 0.0
    llm_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrain_plan: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None

    def to_json(self) -> str:
        data = {
            "investigation_id": self.investigation_id,
            "status": self.status,
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "steps": self.steps,
            "verdict": self.verdict.model_dump() if self.verdict else None,
            "llm_cost_usd": self.llm_cost_usd,
            "llm_calls": self.llm_calls,
            "retrain_plan": self.retrain_plan,
            "protocol": self.protocol,
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
    propose_retrain: ProposeRetrain | None = None,
    policy: str = "eig",
    random_seed: int = 0,
    initial_hypotheses: Sequence[Hypothesis] | None = None,
    estimate_pair_cost: Callable[[Intervention, Intervention], float] | None = None,
    pair_factory: Callable = make_pair,
    testable_classes: set[str] | None = None,
    initial_evidence: Sequence[str] = (),
    causal_checks: bool = True,
    hypothesis_samples: int = 1,
    sandbox_check: dict[str, Any] | None = None,
) -> InvestigationResult:
    """Run the loop; on_step gets every step as soon as it is recorded."""
    if type(hypothesis_samples) is not int or not 1 <= hypothesis_samples <= 5:
        raise ValueError("hypothesis_samples must be an integer from 1 to 5")
    if initial_hypotheses is not None and hypothesis_samples != 1:
        raise ValueError("Frozen hypotheses cannot be combined with repeated sampling")
    if policy not in {"eig", "fixed", "random"}:
        raise ValueError(f"Unknown policy: {policy}")
    if max_interventions < 1:
        raise ValueError("max_interventions must be positive")
    rng = random.Random(random_seed)
    steps: list[dict[str, Any]] = []

    def add(step: dict[str, Any]) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    label_checks = None
    if "training_labels" in case:
        if ctx.mode != "B":
            raise ValueError("training_labels are supported only in mode B")
        label_checks = check_training_labels(case["training_labels"])
        case = {k: v for k, v in case.items() if k != "training_labels"}
        case["training_label_checks"] = label_checks
        add({"kind": "training_label_checks", "raw": label_checks, "cost_usd": 0.0})

    sandbox_evidence = None
    if sandbox_check is not None:
        from core.config import get_settings

        if get_settings().public_demo:
            raise ValueError("Generated sandbox checks are disabled in public demo")
        if ctx.budget_left_usd <= router.log.cost_usd:
            raise ValueError("No remaining budget for check generation")
        generation_cost_before = router.log.cost_usd
        sandbox_evidence = generated_check(router, sandbox_check)
        add(
            {
                "kind": "sandbox_check",
                "raw": sandbox_evidence,
                "cost_usd": router.log.cost_usd - generation_cost_before,
            }
        )
        case = {**case, "sandbox_check": sandbox_evidence}

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
    if hypothesis_samples > 1:
        hypotheses, sampling = sample_hypotheses(
            router,
            case,
            mode=ctx.mode,
            samples=hypothesis_samples,
            budget_usd=ctx.budget_left_usd,
        )
        add(sampling)
    else:
        hypotheses = (
            propose_hypotheses(router, case, mode=ctx.mode)
            if initial_hypotheses is None
            else list(initial_hypotheses)
        )
    by_id = {h.id: h for h in hypotheses}
    classes = {h.id: h.fault_class for h in hypotheses}
    probs = {h.id: h.prior for h in hypotheses}
    rest = 1 - sum(probs.values())
    if rest > 1e-9:  # causes outside the list keep their share of the mass
        probs["unlisted"], classes["unlisted"] = rest, "unlisted"
    supported = set(BUILDERS) if testable_classes is None else testable_classes
    testable = {h.id for h in hypotheses if h.fault_class in supported}
    remaining = {p.name: p for p in probes if p.available(ctx)}
    base_hp = dict(ctx.job_config.get("hyperparameters", {}))
    evidence: list[str] = list(initial_evidence)
    if sandbox_evidence is not None:
        evidence.append("sandbox_check")
    if label_checks is not None:
        evidence.append("training_label_checks")
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
    tested: set[str] = set()

    while True:
        candidates = [h.id for h in hypotheses if h.id in testable]
        # Scheduling conditions on causes that have not been tested.
        # The reported belief still includes tested causes and unlisted mass.
        planning_probs = {h: p for h, p in probs.items() if h not in tested}
        planning_mass = sum(planning_probs.values())
        if planning_mass > 0:
            planning_probs = {h: p / planning_mass for h, p in planning_probs.items()}
        budget = ctx.budget_left_usd - router.log.cost_usd
        if policy == "eig":
            costs = {name: p.estimate_cost(ctx) for name, p in remaining.items()}
            probe_name, _ = best_probe(planning_probs, classes, costs)
            gain = (
                expected_gain(planning_probs, classes, probe_name)
                if probe_name
                else 0.0
            )
            action, why = next_action(
                planning_probs,
                threshold=threshold,
                gain_bits=gain,
                budget_left_usd=budget,
                intervention_cost_usd=intervention_cost_usd,
                testable=testable,
            )
        else:
            names = list(remaining)
            probe_name = (
                rng.choice(names)
                if policy == "random" and names
                else next(iter(names), None)
            )
            gain = (
                expected_gain(probs, classes, probe_name)
                if probe_name is not None
                else 0.0
            )
            if not candidates:
                action, why = "stop", "no hypothesis left to test"
            elif probe_name is not None:
                action, why = "probe", f"{policy} probe order"
            elif budget < intervention_cost_usd:
                action, why = "stop", "the budget does not cover a causal check"
            else:
                action, why = "intervene", f"{policy} hypothesis order"
        if action == "stop":
            add({"kind": "stop", "reason": why})
            break

        if action == "probe":
            probe = remaining.pop(probe_name)
            estimated_cost = probe.estimate_cost(ctx)
            if not math.isfinite(estimated_cost) or estimated_cost < 0:
                raise ValueError(
                    "The probe cost estimate must be finite and nonnegative"
                )
            remaining_budget = ctx.budget_left_usd - router.log.cost_usd
            if estimated_cost > remaining_budget:
                add(
                    {
                        "kind": "stop",
                        "reason": "the budget does not cover this probe",
                        "probe": probe_name,
                        "estimated_cost_usd": estimated_cost,
                        "budget_left_usd": remaining_budget,
                    }
                )
                break
            run = probe.run(ctx)
            if not math.isfinite(run.cost_usd) or run.cost_usd < 0:
                raise ValueError("The probe cost must be finite and nonnegative")
            ctx.budget_left_usd -= run.cost_usd
            probs = update(probs, classes, probe_name, run.outcome)
            step = {
                "kind": "probe",
                "probe": probe_name,
                "estimated_cost_usd": estimated_cost,
                "cost_usd": run.cost_usd,
                "budget_left_usd": ctx.budget_left_usd - router.log.cost_usd,
                "outcome": run.outcome,
                "gain_bits": round(gain, 4),
                "raw": json.loads(run.summary),
                "probs": _rounded(probs),
            }
            if explain and ctx.budget_left_usd > router.log.cost_usd:
                card = interpret_probe(
                    router, hypotheses, probe_name, run.outcome, step["raw"]
                )
                step["card"] = card.explanation
            add(step)
            evidence.append(f"probe:{probe_name}")
            continue

        if not causal_checks:
            add(
                {
                    "kind": "stop",
                    "reason": "causal checks disabled for this ablation",
                    "probs": _rounded(probs),
                }
            )
            break

        if policy == "random":
            leader_id = rng.choice(candidates)
        elif policy == "fixed":
            leader_id = candidates[0]
        else:
            leader_id = max(candidates, key=probs.get)
        leader = by_id[leader_id]
        target, control = pair_factory(leader, ctx)
        check_cost = (
            estimate_pair_cost(target, control)
            if estimate_pair_cost is not None
            else intervention_cost_usd
        )
        if not math.isfinite(check_cost) or check_cost < 0:
            raise ValueError("The pair cost must be finite and nonnegative")
        remaining_budget = ctx.budget_left_usd - router.log.cost_usd
        if check_cost > remaining_budget:
            add(
                {
                    "kind": "stop",
                    "reason": "the budget does not cover this causal check",
                    "estimated_cost_usd": check_cost,
                    "budget_left_usd": remaining_budget,
                }
            )
            break
        run_pair(target, control)
        check = causal_check(
            target.metrics, control.metrics, min_effect=min_effect, noise_sd=noise_sd
        )
        prior_probs = dict(probs)
        probs = update_causal(probs, classes, leader.id, check.status)
        interventions += 1
        ctx.budget_left_usd -= check_cost
        evidence.append(f"intervention:{leader.id}")
        add(
            {
                "kind": "intervention",
                "hypothesis": leader.id,
                "reason": why,
                "target": asdict(target),
                "control": asdict(control),
                "result": asdict(check),
                "prior_probs": prior_probs,
                "planning_probs": dict(planning_probs),
                "probs": dict(probs),
                "belief_model": MODEL,
                "cost_usd": check_cost,
            }
        )
        if check.status == "confirmed":
            result.verdict = write_verdict(
                router, leader, target, control, check, base_hp, list(evidence)
            )
            result.status = "confirmed"
            add({"kind": "verdict", **result.verdict.model_dump()})
            plan = propose_retrain(target, control) if propose_retrain else None
            if plan is not None:
                result.retrain_plan = plan
                add({"kind": "retrain_proposal", **plan})
            break
        # Do not repeat a tested hypothesis; retain its updated belief.
        testable.discard(leader.id)
        tested.add(leader.id)
        if interventions >= max_interventions:
            add({"kind": "stop", "reason": "intervention limit"})
            break

    result.llm_cost_usd = router.log.cost_usd
    result.llm_calls = router.log.totals("model")
    return result
