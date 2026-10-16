"""B19, comparison method 2: one Ultra call with the whole case and no runs.

Ultra gets the case the agent starts from: the config, the loss per checkpoint, the
findings and, when the case has them, data samples and wrong answers. It ranks the
causes and says whether it is sure. Without experiments, "confirmed" here is Ultra's
own claim, so on a healthy case it counts as a false alarm like any other.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from bench.diagnosis import Diagnosis, ranked
from core.agent.catalog import catalog_text
from core.agent.router import Router
from core.agent.schemas import FaultClass
from core.agent.steps import load_prompt


class RankedCause(BaseModel):
    fault_class: FaultClass
    probability: float = Field(ge=0.0, le=1.0)
    rationale: str


class SingleShotDiagnosis(BaseModel):
    causes: list[RankedCause] = Field(min_length=1)
    status: Literal["confirmed", "inconclusive"]
    patch: str = ""


def normalized(causes: list[RankedCause]) -> list[tuple[str, float]]:
    """Ranking with probabilities scaled down when they add up to more than 1."""
    total = sum(c.probability for c in causes)
    scale = 1 / total if total > 1 else 1.0
    return ranked((c.fault_class, c.probability * scale) for c in causes)


def diagnose_single(
    router: Router, case: dict[str, Any], mode: str = "A", version: str = "v1"
) -> Diagnosis:
    prompt = load_prompt("single_shot", version)
    messages = [
        {"role": "system", "content": prompt.render(catalog=catalog_text(mode))},
        {"role": "user", "content": f"Case:\n{json.dumps(case, indent=2)}"},
    ]
    before = len(router.log.calls)
    answer = router.structured(
        "single_shot", messages, SingleShotDiagnosis, prompt_version=prompt.id
    )
    calls = router.log.calls[before:]
    ranking = normalized(answer.causes)
    confirmed = answer.status == "confirmed"
    return Diagnosis(
        method="single_ultra",
        ranking=ranking,
        status=answer.status,
        cause=ranking[0][0] if confirmed else None,
        patch=answer.patch if confirmed else "",
        llm_cost_usd=sum(c.cost_usd for c in calls),
        llm_tokens=sum(c.tokens_in + c.tokens_out for c in calls),
        steps=[{"kind": "single_shot", **answer.model_dump()}],
    )
