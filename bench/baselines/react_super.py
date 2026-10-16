"""B19, comparison method 3: a ReAct agent on Super with the agent's tools and no planner.

Each turn Super writes a thought, its ranking of causes and one action: run a probe,
run a causal check for a fault class, or give the final answer. The tools, the
causal check statistics and the limit on checks are the agent's own. What is
missing is the planner: Super chooses the order itself instead of the expected
information gain per dollar. As in the agent, only a confirmed causal check makes
the status confirmed; the code decides it, not Super.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from bench.baselines.single_ultra import RankedCause, normalized
from bench.baselines.tools import check_step, patch_of, probe_step
from bench.diagnosis import Diagnosis, pair_jobs
from core.agent.catalog import catalog_text
from core.agent.context import InvestigationContext
from core.agent.loop import RunPair
from core.agent.router import Router
from core.agent.schemas import FaultClass
from core.agent.steps import load_prompt
from core.interventions.pairs import BUILDERS
from core.probes.base import Probe
from core.probes.level0.recipe import LEVEL0


class ReactTurn(BaseModel):
    thought: str
    action: Literal["run_probe", "causal_check", "final_answer"]
    probe: str | None = None
    fault_class: FaultClass | None = None
    causes: list[RankedCause] = Field(min_length=1)


def _user(title: str, payload: Any) -> dict[str, str]:
    return {"role": "user", "content": f"{title}:\n{json.dumps(payload, indent=2)}"}


def diagnose_react(
    ctx: InvestigationContext,
    case: dict[str, Any],
    router: Router,
    run_pair: RunPair,
    *,
    probes: Sequence[Probe] = LEVEL0,
    max_turns: int = 8,
    max_checks: int = 2,
    seeds: int = 2,
    min_effect: float = 0.03,
    noise_sd: float | None = 0.0146,
    pair_cost_usd: float = 0.0,
    mode: str = "A",
    version: str = "v1",
) -> Diagnosis:
    prompt = load_prompt("react", version)
    system = prompt.render(
        probes=", ".join(p.name for p in probes),
        seeds=str(seeds),
        pair_cost=f"{pair_cost_usd:.2f}",
        testable=", ".join(BUILDERS),
        max_checks=str(max_checks),
        catalog=catalog_text(mode),
    )
    messages = [{"role": "system", "content": system}, _user("Case", case)]
    by_name = {p.name: p for p in probes}
    before = len(router.log.calls)
    steps: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    turn = None

    for _ in range(max_turns):
        turn = router.structured("react", messages, ReactTurn, prompt_version=prompt.id)
        messages.append({"role": "assistant", "content": turn.model_dump_json()})
        record = {"kind": "react", **turn.model_dump()}
        steps.append(record)
        if turn.action == "final_answer":
            break
        if turn.action == "run_probe":
            if turn.probe not in by_name:
                observation = {"error": f"no check {turn.probe!r} left to run"}
            else:
                step = probe_step(by_name.pop(turn.probe), ctx)
                steps.append(step)
                observation = {k: step[k] for k in ("probe", "outcome", "raw")}
        elif len(checks) >= max_checks:
            observation = {"error": "no causal checks left"}
        elif turn.fault_class not in BUILDERS:
            observation = {"error": f"no causal check for {turn.fault_class!r}"}
        elif turn.fault_class in {c["hypothesis"] for c in checks}:
            observation = {"error": f"{turn.fault_class} was checked already"}
        else:
            step = check_step(
                turn.fault_class,
                ctx,
                run_pair,
                min_effect=min_effect,
                noise_sd=noise_sd,
                cost_usd=pair_cost_usd,
            )
            steps.append(step)
            checks.append(step)
            observation = {
                "fault_class": turn.fault_class,
                "target": {
                    "change": step["target"]["description"],
                    "metric_per_seed": step["target"]["metrics"],
                },
                "control": {
                    "change": step["control"]["description"],
                    "metric_per_seed": step["control"]["metrics"],
                },
                "result": step["result"],
            }
        record["observation"] = observation
        messages.append(_user("Result", observation))

    confirmed = [c for c in checks if c["result"]["status"] == "confirmed"]
    calls = router.log.calls[before:]
    return Diagnosis(
        method="react_super",
        ranking=normalized(turn.causes) if turn else [],
        status="confirmed" if confirmed else "inconclusive",
        cause=confirmed[0]["hypothesis"] if confirmed else None,
        patch=patch_of(confirmed[0], ctx) if confirmed else "",
        llm_cost_usd=sum(c.cost_usd for c in calls),
        llm_tokens=sum(c.tokens_in + c.tokens_out for c in calls),
        mini_jobs=sum(pair_jobs(c) for c in checks),
        training_cost_usd=pair_cost_usd * len(checks),
        steps=steps,
    )
