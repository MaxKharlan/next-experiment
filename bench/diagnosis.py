"""What every method in the benchmark returns for a case (section 10 of the project doc).

The metrics read only this record: the ranking gives top-1, top-3 and the probability
of the true cause; status "confirmed" on a healthy case is a false alarm; "confirmed"
lists every cause the method declared proven, so a confirmed cause outside the
truth counts as a wrong confirmation; the cost fields and the number of mini-jobs
give the price of the diagnosis.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from core.agent.loop import InvestigationResult


@dataclass
class Diagnosis:
    method: str
    # fault classes with probabilities, most likely first
    ranking: list[tuple[str, float]]
    status: Literal["confirmed", "inconclusive"]
    cause: str | None = None  # the confirmed fault class
    patch: str = ""
    llm_cost_usd: float = 0.0
    llm_tokens: int = 0
    mini_jobs: int = 0  # Post-training jobs or local runs, one per arm and seed
    training_cost_usd: float = 0.0
    steps: list[dict[str, Any]] = field(default_factory=list)
    # every cause the method declared proven, the chosen one included
    confirmed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # a confirmed cause is the method's answer, whatever the ranking said
        if self.cause is not None:
            p = dict(self.ranking).get(self.cause, 0.0)
            rest = [(c, q) for c, q in self.ranking if c != self.cause]
            self.ranking = [(self.cause, p), *rest]

    @property
    def top(self) -> list[str]:
        return [c for c, _ in self.ranking]


def ranked(pairs: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Fault classes by probability; a class named twice gets the sum."""
    total: dict[str, float] = {}
    for fault_class, p in pairs:
        total[fault_class] = total.get(fault_class, 0.0) + p
    return sorted(total.items(), key=lambda kv: kv[1], reverse=True)


def pair_jobs(step: dict[str, Any]) -> int:
    """Jobs behind one causal check: one per seed in each arm."""
    return len(step["target"]["metrics"]) + len(step["control"]["metrics"])


def llm_tokens(calls: dict[str, dict[str, Any]]) -> int:
    return sum(row["tokens_in"] + row["tokens_out"] for row in calls.values())


def from_investigation(result: InvestigationResult) -> Diagnosis:
    """Next Experiment's report as a Diagnosis: the last probabilities of the loop."""
    classes = {h.id: h.fault_class for h in result.hypotheses}
    probs: dict[str, float] = {}
    for step in result.steps:
        probs = step.get("probs", probs)
        if (
            step["kind"] == "intervention"
            and "probs" not in step
            and step["result"]["status"] != "confirmed"
        ):
            # Preserve the historical behavior for reports without posteriors.
            probs = {**probs, step["hypothesis"]: 0.0}
            total = sum(probs.values())
            probs = {h: p / total for h, p in probs.items()} if total else probs
    checks = [s for s in result.steps if s["kind"] == "intervention"]
    proven = [s for s in checks if s["result"]["status"] == "confirmed"]
    return Diagnosis(
        method="next_experiment",
        ranking=ranked((classes[h], p) for h, p in probs.items() if h in classes),
        status="confirmed" if result.status == "confirmed" else "inconclusive",
        cause=classes[result.verdict.hypothesis_id] if result.verdict else None,
        patch=result.verdict.patch if result.verdict else "",
        llm_cost_usd=result.llm_cost_usd,
        llm_tokens=llm_tokens(result.llm_calls),
        mini_jobs=sum(pair_jobs(s) for s in checks),
        training_cost_usd=sum(s.get("cost_usd", 0.0) for s in checks),
        steps=result.steps,
        confirmed=[classes[s["hypothesis"]] for s in proven],
    )
