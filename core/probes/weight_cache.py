"""Cached B25 measurements for a specific completed Token Factory job."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from core.probes.base import ProbeResult
from core.probes.level0.recipe import LEVEL0
from core.probes.level0.weights import LoraWeights

CACHE_DIR = Path("artifacts/probes/weights")
SUPPORTED_MODEL = "Qwen/Qwen3-0.6B"
CACHE_VERSION = 1


def weights_enabled():
    value = os.environ.get("NEXT_EXPERIMENT_WEIGHTS", "0")
    if value not in {"0", "1"}:
        raise ValueError("NEXT_EXPERIMENT_WEIGHTS must be 0 or 1")
    return value == "1"


def cache_path(job_id):
    digest = hashlib.sha256(job_id.encode()).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def load_measurement(ctx):
    path = cache_path(ctx.job_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No B25 measurement for {ctx.job_id}. "
            "Run python -m scripts.prepare_weight_probes first."
        )
    record = json.loads(path.read_text())
    expected_model = ctx.job_config.get("model")
    if (
        record.get("version") != CACHE_VERSION
        or record.get("job_id") != ctx.job_id
        or record.get("model") != expected_model
        or record.get("hyperparameters") != ctx.job_config.get("hyperparameters")
    ):
        raise ValueError(f"B25 cache does not match the investigation: {path}")
    if expected_model != SUPPORTED_MODEL:
        raise ValueError("The B25 threshold is calibrated only for Qwen3-0.6B")
    raw = record["raw"]
    if (
        raw.get("model") != expected_model
        or not math.isfinite(raw["rel_median"])
        or raw["rel_median"] < 0
        or raw["modules"] < 1
    ):
        raise ValueError(f"Invalid B25 measurement: {path}")
    return record


class CachedLoraWeights(LoraWeights):
    """The existing weight probe using previously calculated statistics."""

    def __init__(self, record):
        self.record = record

    def available(self, ctx):
        return self.record["job_id"] == ctx.job_id

    def run(self, ctx):
        if not self.available(ctx):
            raise ValueError("B25 measurement belongs to another job")
        raw = {
            **self.record["raw"],
            "small_below": self.SMALL,
            "source_job_id": self.record["job_id"],
            "adapter_sha256": self.record["adapter_sha256"],
        }
        outcome = "small" if raw["rel_median"] < self.SMALL else "normal"
        return self.check_outcome(
            ProbeResult(
                outcome=outcome,
                raw_ref=str(cache_path(ctx.job_id)),
                summary=json.dumps(raw),
            )
        )


def investigation_probes(ctx):
    if not weights_enabled():
        return LEVEL0
    return (*LEVEL0, CachedLoraWeights(load_measurement(ctx)))
