"""Local-only label audit requests; never accept server paths or executable code."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from core.agent.label_checks import LabelCheckInput, check_training_labels


class LabelAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Mode B label audit", min_length=1, max_length=120)
    training_labels: LabelCheckInput


def prepare_audit(request):
    payload = request.training_labels.model_dump()
    result = check_training_labels(payload)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "kind": "training_label_checks",
        "raw": result,
        "input_sha256": digest,
        "cost_usd": 0.0,
    }
