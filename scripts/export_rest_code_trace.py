"""Verify and export the completed REST trace without executing a model."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from apps.api.code_investigations import verify_prepared
from core.eval.runtime import file_digest
from core.interventions.stats import causal_check

ROOT = Path("docs/results/rest-code-investigation-20261020")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_events(text, report):
    events = []
    for block in text.split("\n\n"):
        if not block.startswith("id:"):
            continue
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        events.append(
            {
                "seq": int(fields["id"]),
                "type": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    if [e["seq"] for e in events] != list(range(len(events))):
        raise ValueError("SSE sequence is incomplete")
    if events != report["events"] or events[-1]["type"] != "finished":
        raise ValueError("SSE differs from the persisted finished report")
    return events


def main():
    protocol = read(ROOT / "protocol.json")
    prepared = read(ROOT / "prepared.json")
    assert file_digest(ROOT / "prepared.json") == protocol["prepared_sha256"]
    assert read(ROOT / "request.json") == prepared["request"]
    verify_prepared(prepared)
    status = read(ROOT / "execution-status.json")
    assert status["status"] == "trace_complete" and status["restart_verified"]
    inv_id = status["id"]
    accepted = read(ROOT / "accepted.json")
    assert accepted == {"http_status": 202, "body": {"id": inv_id, "reused": False}}
    restart = read(ROOT / "restart-verification.json")
    assert restart == {
        "report_equal": True,
        "idempotent_post": {"id": inv_id, "reused": True},
        "reused_without_launch": True,
    }
    report = read(ROOT / "report.json")
    assert report["investigation_id"] == inv_id
    assert report["execution_status"] == "finished" and not report["partial"]
    assert report["error"] is None
    events = verify_events((ROOT / "events.sse").read_text(), report)
    interventions = [s for s in report["steps"] if s["kind"] == "intervention"]
    assert len(interventions) == 1
    step = interventions[0]
    source = Path("artifacts/code-investigations") / inv_id
    expected_dirs = {
        f"{arm}-s{seed}" for arm in ("target", "control") for seed in protocol["seeds"]
    }
    assert {p.name for p in source.iterdir() if p.is_dir()} == expected_dirs
    measurements = {}
    scores = {"target": [], "control": []}
    source_hashes = {}
    for arm, arm_scores in scores.items():
        for seed in protocol["seeds"]:
            name = f"{arm}-s{seed}"
            directory = source / name
            summary = read(directory / "summary.json")
            score = read(directory / "check-score.json")
            fault = None if arm == "target" else prepared["request"]["code_fault"]
            assert score["seed"] == summary["seed"] == seed
            assert score["code_fault"] == summary["code_fault"] == fault
            assert summary["hyperparameters"] == prepared["request"]["recipe"]
            assert (
                summary["train_examples"] == prepared["datasets"]["train"]["examples"]
            )
            assert (
                summary["valid_examples"]
                == prepared["datasets"]["validation"]["examples"]
            )
            assert summary["device"] == prepared["request"]["device"]
            assert score["datasets"] == {
                spec["path"]: spec["sha256"] for spec in prepared["datasets"].values()
            }
            assert summary["parameter_audit"]["trainable_non_lora_parameters"] == 0
            assert summary["gradient_audit"]["tensor_counts"]["nonfinite"] == 0
            count = prepared["datasets"]["check"]["examples"]
            assert (score["exact_match"] * count).is_integer()
            measurements[name] = {
                "seed": seed,
                "code_fault": fault,
                "exact_match": score["exact_match"],
                "correct": int(score["exact_match"] * count),
                "check_examples": count,
                "training_steps": summary["steps"],
                "training_wall_seconds": summary["wall_seconds"],
                "weights_sha256": file_digest(
                    directory / "adapter/adapter_model.safetensors"
                ),
            }
            arm_scores.append(score["exact_match"])
            for filename in (
                "summary.json",
                "check-score.json",
                "metrics.jsonl",
                "parameter-audit.json",
                "gradient-audit.json",
            ):
                src = directory / filename
                target = ROOT / "measurements" / name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert target.read_bytes() == src.read_bytes()
                else:
                    target.write_bytes(src.read_bytes())
                source_hashes[str(src)] = file_digest(src)
        assert arm_scores == step[arm]["metrics"]
    computed = asdict(
        causal_check(
            scores["target"],
            scores["control"],
            min_effect=prepared["request"]["min_effect"],
        )
    )
    assert computed == step["result"]
    calls = report["llm_usage"]["calls"]
    assert len(calls) == 1 and calls[0]["attempt"] == 1
    assert calls[0]["provider_request_id"] and calls[0]["provider_response_id"]
    save(
        ROOT / "verification.json",
        {
            "investigation_id": inv_id,
            "status": "verified",
            "http_status": 202,
            "sse_events": len(events),
            "trained_adapters": len(measurements),
            "distinct_adapter_hashes": len(
                {m["weights_sha256"] for m in measurements.values()}
            ),
            "restart_verified": True,
            "idempotent_reuse_verified": True,
            "diagnosis_status": report["diagnosis_status"],
            "comparison": computed,
            "arm_means": {
                arm: statistics.fmean(values) for arm, values in scores.items()
            },
            "measurements": measurements,
            "llm_usage": report["llm_usage"],
            "source_files_sha256": source_hashes,
            "prediction_scope": "The frozen REST executor saved eight aggregate check scores over 64 examples each. It did not persist individual generated answers; verification recomputes the comparison from recorded seed scores and checks their dataset and training bindings. No generation was repeated.",
            "scope": protocol["scope"],
        },
    )
    files = [
        ROOT / name
        for name in (
            "protocol.json",
            "prepared.json",
            "request.json",
            "submission-intent.json",
            "accepted.json",
            "execution-status.json",
            "events.sse",
            "report.json",
            "restart-verification.json",
            "verification.json",
        )
    ] + [p for p in (ROOT / "measurements").rglob("*") if p.is_file()]
    save(
        ROOT / "manifest.json",
        {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sorted(files)},
    )
    print(json.dumps({"status": "verified", "adapters": 8, "events": len(events)}))


if __name__ == "__main__":
    main()
