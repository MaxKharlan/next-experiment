"""Bounded, opt-in mode-B label checks supplied as evidence to the agent."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from bench.faults.labels import audit_eos_labels, audit_label_shift, audit_prompt_labels


class LabelRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_ids: list[StrictInt] = Field(max_length=32768)
    labels: list[StrictInt] = Field(max_length=32768)

    @model_validator(mode="after")
    def validate_row(self):
        if len(self.input_ids) != len(self.labels):
            raise ValueError("Input IDs and labels must have equal lengths")
        if any(token < 0 for token in self.input_ids):
            raise ValueError("Input token IDs must be nonnegative")
        if any(label < 0 and label != -100 for label in self.labels):
            raise ValueError("Labels must be token IDs or -100")
        return self


class LabelCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: list[LabelRow] = Field(min_length=1, max_length=1024)
    observed: list[LabelRow] = Field(min_length=1, max_length=1024)
    eos_token_id: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_reference(self):
        if sum(len(r.input_ids) for r in self.reference + self.observed) > 131072:
            raise ValueError("Label checks allow at most 131072 token positions")
        for row in self.reference:
            if any(
                label not in (-100, token)
                for token, label in zip(row.input_ids, row.labels)
            ):
                raise ValueError("Reference labels must be unshifted token IDs or -100")
        return self


def check_training_labels(payload: dict) -> dict:
    """Reference rows must be unpadded, with only prompt positions ignored."""
    data = LabelCheckInput.model_validate(payload)
    reference = [row.model_dump() for row in data.reference]
    observed = [row.model_dump() for row in data.observed]
    # This also validates that both sides describe the same token sequences.
    shift = audit_label_shift(reference, observed)
    try:
        prompt = audit_prompt_labels(reference, observed)
    except ValueError:
        prompt = {
            "status": "not_applicable",
            "reason": "labels are not prompt-only additions",
        }
    if data.eos_token_id is None:
        eos = {"status": "unavailable", "reason": "EOS token ID was not supplied"}
    else:
        try:
            eos = audit_eos_labels(reference, observed, data.eos_token_id)
        except ValueError:
            eos = {
                "status": "not_applicable",
                "reason": "labels are not masking-only changes",
            }
    return {
        "scope": "supplied unpadded labels; reference correctness is caller-owned",
        "rows": len(reference),
        "alignment": shift,
        "prompt": prompt,
        "eos": eos,
        "causal_confirmation": False,
    }
