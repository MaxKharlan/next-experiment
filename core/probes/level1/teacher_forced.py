"""Opt-in local level-1 diagnostics without invented fault likelihoods."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from core.eval.data import read_examples
from core.eval.generate import free_cache, load_model
from core.eval.runtime import file_digest, freeze_runtime, verify_runtime
from core.interventions.tf_runner import weights_digest
from core.probes.base import Probe, ProbeResult
from core.probes.level1.logits import aggregate, inspect_example


class TeacherForcedLogits(Probe):
    name = "teacher_forced_logits"
    level = 1
    outcomes = ("measured",)

    def available(self, ctx):
        return all(
            ctx.files.get(key) and Path(ctx.files[key]).exists()
            for key in ("logit_data", "model_snapshot")
        ) and isinstance(ctx.job_config.get("logit_context_length"), int)

    def estimate_cost(self, ctx):
        return 0.0

    def run(self, ctx):
        if not self.available(ctx):
            raise ValueError(
                "Explicit local data, snapshot and context length required"
            )
        started = time.monotonic()
        data = Path(ctx.files["logit_data"])
        snapshot = Path(ctx.files["model_snapshot"])
        adapter = (
            Path(ctx.files["logit_adapter"]) if ctx.files.get("logit_adapter") else None
        )
        runtime = freeze_runtime(snapshot, ctx.job_config["model"])
        data_sha = file_digest(data)
        adapter_sha = weights_digest(adapter) if adapter else None
        device = ctx.job_config.get("logit_device", "cpu")
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("Unsupported local device")
        examples = read_examples(data)
        if len(examples) > 200:
            raise ValueError("Local diagnostic is bounded to 200 explicit examples")
        model, tokenizer, _ = load_model(adapter, base_model=str(snapshot))
        model.to(device).eval()
        rows = []
        try:
            for example in examples:
                rows.append(
                    inspect_example(
                        model,
                        tokenizer,
                        example,
                        device,
                        context_length=ctx.job_config["logit_context_length"],
                    )
                )
                free_cache(device)
        finally:
            del model, tokenizer
            gc.collect()
            free_cache(device)
        verify_runtime(runtime)
        if (
            file_digest(data) != data_sha
            or adapter
            and weights_digest(adapter) != adapter_sha
        ):
            raise ValueError("Diagnostic inputs changed")
        raw = {
            "runtime": runtime,
            "data_sha256": data_sha,
            "adapter_sha256": adapter_sha,
            "rows": rows,
            "summary": aggregate(rows),
            "likelihood_status": "Neutral diagnostic outcome; no fitted fault discrimination.",
        }
        return self.check_outcome(
            ProbeResult(
                outcome="measured",
                raw_ref=f"investigations/{ctx.investigation_id}/probes/{self.name}.json",
                duration_s=time.monotonic() - started,
                summary=json.dumps(raw),
            )
        )
