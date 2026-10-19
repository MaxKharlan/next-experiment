"""B2: log of every LLM call: step, model, tokens, cost, latency and prompt version.

The totals go into the report and the video. save() writes the whole log as JSONL
to storage under investigations/<id>/llm_calls.jsonl.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.connectors.storage import Storage
from core.connectors.token_factory import CallResult


@dataclass(frozen=True)
class LLMCall:
    id: str
    investigation_id: str
    purpose: str  # the agent step, e.g. "hypotheses" or "verdict"
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    prompt_version: str
    attempt: int  # 1 for the first try, more for structured-output retries
    created_at: float
    provider_request_id: str | None = None
    provider_response_id: str | None = None
    spending_operation_id: str | None = None


@dataclass
class CallLog:
    investigation_id: str
    calls: list[LLMCall] = field(default_factory=list)
    on_add: Callable[[LLMCall], None] | None = field(default=None, repr=False)

    def add(
        self, purpose: str, prompt_version: str, result: CallResult, attempt: int = 1
    ) -> LLMCall:
        call = LLMCall(
            id=uuid.uuid4().hex[:12],
            investigation_id=self.investigation_id,
            purpose=purpose,
            model=result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            prompt_version=prompt_version,
            attempt=attempt,
            created_at=time.time(),
            provider_request_id=result.provider_request_id,
            provider_response_id=result.provider_response_id,
            spending_operation_id=result.spending_operation_id,
        )
        self.calls.append(call)
        if self.on_add is not None:
            self.on_add(call)
        return call

    def totals(self, by: str = "model") -> dict[str, dict[str, Any]]:
        """Calls, tokens, cost and latency grouped by "model" or "purpose"."""
        out: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                     "latency_ms": 0}
        )  # fmt: skip
        for c in self.calls:
            row = out[getattr(c, by)]
            row["calls"] += 1
            row["tokens_in"] += c.tokens_in
            row["tokens_out"] += c.tokens_out
            row["cost_usd"] += c.cost_usd
            row["latency_ms"] += c.latency_ms
        return dict(out)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def save(self, storage: Storage) -> str:
        key = f"investigations/{self.investigation_id}/llm_calls.jsonl"
        lines = (json.dumps(asdict(c)) + "\n" for c in self.calls)
        return storage.put_text(key, "".join(lines))
