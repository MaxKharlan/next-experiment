"""Download completed-job adapters and cache B25 statistics. Never train."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from bench.cases import load_cases
from core.agent.context import InvestigationContext
from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from core.probes.level0.weights import base_norms, update_stats
from core.probes.weight_cache import (
    CACHE_VERSION,
    SUPPORTED_MODEL,
    cache_path,
    load_measurement,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()

    cases = load_cases(args.split)
    if args.cases:
        unknown = set(args.cases) - {case.id for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case.id in args.cases]
    if not cases:
        parser.error("No cases selected")

    storage = LocalStorage("artifacts")
    tf = None
    for case in cases:
        if case.model != SUPPORTED_MODEL:
            raise ValueError(f"Unsupported B25 calibration model: {case.model}")
        ctx = InvestigationContext(
            investigation_id=f"prepare-weights-{case.id}",
            mode="A",
            job_id=case.job_id,
            job_config={"model": case.model, **case.job_config()},
        )
        path = cache_path(case.job_id)
        if path.exists():
            load_measurement(ctx)
            print(f"Reusing {case.id}: {path}", flush=True)
            continue

        if tf is None:
            tf = TokenFactory.from_settings()
        job = tf.job(case.job_id)
        if job.status != "succeeded":
            raise ValueError(f"{case.job_id} has status {job.status}")
        if job.model != case.model:
            raise ValueError(f"Model mismatch for {case.job_id}")

        adapter = download_last_checkpoint(
            tf,
            storage,
            case.job_id,
            f"probes/weight-adapters/{path.stem}",
        )
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != case.model:
            raise ValueError(f"Adapter model mismatch: {adapter}")

        stats = update_stats(adapter, base_norms(case.model))
        digest = weights_digest(adapter)
        if not digest:
            raise ValueError(f"Missing adapter weights: {adapter}")

        record = {
            "version": CACHE_VERSION,
            "job_id": case.job_id,
            "model": case.model,
            "hyperparameters": case.hyperparameters,
            "adapter": str(adapter),
            "adapter_sha256": digest,
            "raw": asdict(stats),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)
        load_measurement(ctx)
        print(
            f"Saved {case.id}: relative norm={stats.rel_median:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
