"""What an evaluation keeps: one record per example and the score over all of them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExampleResult:
    line: int
    expected: str
    answer: str
    score: float  # from the task metric, 0 to 1
    prompt_tokens: int
    answer_tokens: int  # generated tokens without the final EOS
    stopped_on_eos: bool  # the model ended its answer itself
    hit_limit: bool  # generation stopped at max_new_tokens, without EOS
    source: str | None = None

    @property
    def correct(self) -> bool:
        return self.score >= 1.0


@dataclass
class EvalResult:
    metric: str
    value: float  # mean score over the examples
    model: str  # the base model
    data: str  # the file the examples came from
    max_new_tokens: int
    adapter: str | None = None  # None for the base model alone
    examples: list[ExampleResult] = field(default_factory=list)

    details: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_examples(
        cls, metric: str, examples: list[ExampleResult], **meta: Any
    ) -> EvalResult:
        if not examples:
            raise ValueError("no examples to score")
        value = sum(e.score for e in examples) / len(examples)
        return cls(metric=metric, value=value, examples=examples, **meta)

    def wrong(self) -> list[ExampleResult]:
        return [e for e in self.examples if not e.correct]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
