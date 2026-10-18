"""The user input that starts an investigation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetRef(ContractModel):
    ref: str = Field(description="Storage key, local path or provider file id")
    format: Literal["chat_jsonl"] = "chat_jsonl"
    examples: int | None = Field(default=None, ge=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("ref")
    @classmethod
    def ref_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset reference must not be blank")
        return value


class DatasetSplits(ContractModel):
    train: DatasetRef
    validation: DatasetRef
    check: DatasetRef
    test: DatasetRef

    @model_validator(mode="after")
    def references_are_distinct(self) -> DatasetSplits:
        refs = [self.train.ref, self.validation.ref, self.check.ref, self.test.ref]
        if len(set(refs)) != len(refs):
            raise ValueError(
                "train, validation, check and test must be different files"
            )
        return self


class FineTuningJob(ContractModel):
    id: str
    model: str
    hyperparameters: dict[str, Any]
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("id", "model")
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value


class EvaluatorSpec(ContractModel):
    name: str
    metric: str
    higher_is_better: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "metric")
    @classmethod
    def names_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value


class InvestigationCase(ContractModel):
    task: str
    symptom: str
    mode: Literal["A", "B"] = "A"
    job: FineTuningJob
    datasets: DatasetSplits
    evaluator: EvaluatorSpec
    budget_usd: float = Field(default=5.0, gt=0)

    @field_validator("task", "symptom")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value
