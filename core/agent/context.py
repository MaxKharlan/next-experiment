"""Everything a probe or an intervention needs to know about the current investigation."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class InvestigationContext:
    investigation_id: str
    mode: Literal["A", "B"]
    job_id: str
    job_config: dict[str, Any]
    # role -> storage key, e.g. "train", "validation", "eval_set", "checkpoint:120"
    files: dict[str, str] = field(default_factory=dict)
    budget_left_usd: float = 0.0
    # connectors bundle (TokenFactory, Storage, Compute, Tavily); typed later
    connectors: Any = None
