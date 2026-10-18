"""Structured outputs that the Nemotron models must return."""

from typing import Literal

from pydantic import BaseModel, Field

FaultClass = Literal[
    "answer_truncation",
    "truncation",
    "chat_template_mismatch",
    "contradictory_labels",
    "train_val_overlap",
    "distribution_shift",
    "lr_too_low",
    "too_many_epochs",
    "lora_rank_too_small",
    "loss_on_prompt",
    "pad_eos_masking",
    "double_label_shift",
    "grad_accum_normalization",
    "other",
]


class Hypothesis(BaseModel):
    id: str
    fault_class: FaultClass
    location: str = Field(
        description="Data file, config field or code path where it shows"
    )
    prior: float = Field(ge=0.0, le=1.0)
    rationale: str
    sources: list[str] = Field(
        default_factory=list, description="URLs from known_issues that support it"
    )


class HypothesisList(BaseModel):
    hypotheses: list[Hypothesis] = Field(min_length=1)


class ProbeInterpretation(BaseModel):
    probe_name: str
    outcome: str
    explanation: str


class Verdict(BaseModel):
    hypothesis_id: str
    status: Literal["confirmed", "rejected", "inconclusive"]
    evidence: list[str] = Field(description="ProbeRun and Intervention ids")
    patch: str = Field(
        description="Unified diff for data prep, config or training code"
    )
    summary: str = Field("", description="Two or three sentences for the report")
