"""B2: which Nemotron 3 model handles each agent step; every call goes to the log.

Per-example calls go to Nano and loop steps to Super. Ultra writes the verdict and
the patch. Once Super has twice failed to square its hypotheses with probe results,
the reasoning steps of the loop move to Ultra as well (project doc, section 06).
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from core.connectors.token_factory import (
    CallResult,
    StructuredOutputError,
    TokenFactory,
)
from core.telemetry.calls import CallLog

T = TypeVar("T", bound=BaseModel)

STEP_ROLES: dict[str, str] = {
    # one call per data example or per event
    "audit_example": "nano",
    "label_errors": "nano",
    "parse_events": "nano",
    # the investigation loop
    "hypotheses": "super",
    "interpret_probe": "super",
    "known_issues": "super",
    "probe_code": "super",
    # final decisions
    "verdict": "ultra",
    "patch": "ultra",
}
# loop steps that move to Ultra after repeated mismatches
ESCALATING = {"hypotheses", "interpret_probe"}


class Router:
    def __init__(self, tf: TokenFactory, log: CallLog, escalate_after: int = 2):
        self.tf = tf
        self.log = log
        self.escalate_after = escalate_after
        self.mismatches = 0

    def role(self, step: str) -> str:
        if step not in STEP_ROLES:
            raise KeyError(f"unknown agent step {step!r}")
        if step in ESCALATING and self.mismatches >= self.escalate_after:
            return "ultra"
        return STEP_ROLES[step]

    def model(self, step: str) -> str:
        return self.tf.model(self.role(step))

    def mismatch(self) -> None:
        """Super's hypotheses disagreed with a probe result."""
        self.mismatches += 1

    def chat(
        self,
        step: str,
        messages: list[dict[str, str]],
        *,
        prompt_version: str,
        **kwargs: Any,
    ) -> CallResult:
        result = self.tf.chat(self.model(step), messages, **kwargs)
        self.log.add(step, prompt_version, result)
        return result

    def structured(
        self,
        step: str,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        prompt_version: str,
        **kwargs: Any,
    ) -> T:
        try:
            parsed, calls = self.tf.structured(
                self.model(step), messages, schema, **kwargs
            )
        except StructuredOutputError as err:
            for i, call in enumerate(err.calls, start=1):
                self.log.add(step, prompt_version, call, attempt=i)
            raise
        for i, call in enumerate(calls, start=1):
            self.log.add(step, prompt_version, call, attempt=i)
        return parsed
