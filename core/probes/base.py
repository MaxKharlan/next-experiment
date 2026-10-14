"""Contract for probes: cheap experiments that test hypotheses about a failed fine-tune."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.agent.context import InvestigationContext


class ProbeResult(BaseModel):
    outcome: str = Field(description="One of Probe.outcomes")
    raw_ref: str = Field(description="Storage key with raw data for the UI card")
    cost_usd: float = 0.0
    duration_s: float = 0.0
    summary: str = ""


class Probe(ABC):
    name: ClassVar[str]
    level: ClassVar[int]  # 0: CPU and Nano, 1: checkpoint inference, 2: mini fine-tune
    outcomes: ClassVar[tuple[str, ...]]

    @abstractmethod
    def estimate_cost(self, ctx: InvestigationContext) -> float: ...

    @abstractmethod
    def run(self, ctx: InvestigationContext) -> ProbeResult: ...

    def check_outcome(self, result: ProbeResult) -> ProbeResult:
        if result.outcome not in self.outcomes:
            raise ValueError(f"{self.name}: unknown outcome {result.outcome!r}")
        return result
