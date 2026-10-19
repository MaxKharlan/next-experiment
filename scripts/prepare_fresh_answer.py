"""Count cached-tokenizer tokens and write a non-executable fresh-run plan.

No model weights, provider requests or final-test scoring are performed. A local
Hugging Face snapshot is required, so this command cannot silently follow main.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from bench.faults.recipe import HEALTHY_HP
from bench.tasks.fresh_extraction import review
from bench.tasks.structured import sha, write_json
from core.interventions.answer_check import bind_pair
from core.probes.level0.answers import file_answers
from core.probes.level0.lengths import count_tokens


def prepare(cohort, snapshot, out):
    from dataclasses import asdict

    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if len(snapshot.name) != 40 or any(
        c not in "0123456789abcdef" for c in snapshot.name
    ):
        raise ValueError("Use an explicit local commit snapshot")
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    clean = cohort / "evaluator-only" / "clean"
    if review(clean, Path("datasets/structured-mixed-v1")) != json.loads(
        (cohort / "review.json").read_text()
    ):
        raise ValueError("Review changed after cohort preparation")
    summaries, split_hashes = {}, {}
    for name, folder in (
        ("clean", clean),
        ("mild", cohort / "observed/mild"),
        ("strong", cohort / "observed/strong"),
    ):
        manifest = json.loads((folder / "manifest.json").read_text())
        for role, spec in manifest["datasets"].items():
            path = folder / spec["ref"]
            if sha(path) != spec["sha256"]:
                raise ValueError("Dataset changed since preparation")
            if role in ("check", "test") and sha(path) != sha(clean / f"{role}.jsonl"):
                raise ValueError("Evaluation split was modified by injection")
        for role in ("train", "validation"):
            path = folder / f"{role}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            tokens = [count_tokens(tok, row["messages"]) for row in rows]
            info = file_answers(path)
            summaries[f"{name}/{role}"] = {
                "examples": len(rows),
                "rendered_tokens": sum(tokens),
                "max_tokens": max(tokens),
                "flagged": info.cut,
                "sha256": sha(path),
            }
        split_hashes[name] = {
            role: sha(folder / f"{role}.jsonl")
            for role in ("train", "validation", "check", "test")
        }
    development = Path("datasets/structured-mixed-v1/json_extraction")
    for role in ("train", "validation"):
        path = development / f"{role}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        tokens = [count_tokens(tok, row["messages"]) for row in rows]
        summaries[f"development/{role}"] = {
            "examples": len(rows),
            "rendered_tokens": sum(tokens),
            "max_tokens": max(tokens),
            "sha256": sha(path),
        }
    source = cohort / "observed/strong/train.jsonl"
    raw = {"files": {"train": asdict(file_answers(source))}}
    arms = bind_pair("answer_truncation", raw, sha(source))
    for arm in arms:
        rows = arm.rows(source)
        tokens = [count_tokens(tok, json.loads(row)["messages"]) for row in rows]
        summaries[arm.kind] = {
            "examples": len(rows),
            "rendered_tokens": sum(tokens),
            "max_tokens": max(tokens),
            "selected_lines_sha256": hashlib.sha256(
                json.dumps(arm.drop_lines).encode()
            ).hexdigest(),
        }
    tokenizer_files = {
        p.name: sha(p)
        for p in sorted(snapshot.iterdir())
        if p.name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "config.json",
            "generation_config.json",
        }
    }
    max_row = max(item["max_tokens"] for item in summaries.values())
    # Price is the repository estimate; provider verification is a submission gate.
    jobs = 6 + 3 + 2 * 8 + 1
    training_upper = jobs * 432 * max_row * 0.40 / 1e6
    plan = {
        "version": 1,
        "status": "prepared_not_executable",
        "task": "json_extraction",
        "model": "Qwen/Qwen3-0.6B",
        "local_evaluator_model_revision": snapshot.name,
        "provider_model_revision": "unverified; provider job metadata required",
        "tokenizer_files_sha256": tokenizer_files,
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        "tokenizer_runtime": {
            n: importlib.metadata.version(n)
            for n in ("transformers", "tokenizers", "jinja2")
        },
        "source_sha256": {
            str(p): sha(p)
            for p in (
                Path("scripts/prepare_fresh_answer.py"),
                Path("core/agent/prompts/hypotheses.v4.md"),
                Path("core/agent/prompts/interpret_probe.v2.md"),
                Path("core/agent/prompts/verdict.v3.md"),
                Path("core/agent/router.py"),
                Path("core/probes/signatures.yaml"),
                Path("core/agent/catalog.py"),
                Path("core/eval/structured.py"),
                Path("core/eval/generate.py"),
                Path("core/probes/level0/answers.py"),
                Path("core/interventions/stats.py"),
                Path("bench/tasks/fresh_extraction.py"),
                Path("bench/tasks/structured_review.py"),
                Path("uv.lock"),
            )
        },
        "datasets": split_hashes,
        "token_counts": summaries,
        "recipe": {**HEALTHY_HP, "lora": True, "packing": False},
        "policy": {
            "temperature": 1,
            "hypotheses": "hypotheses-v4",
            "verdict": "verdict-v3",
            "answer_probe": "answer-cut-v2",
            "experimental_classifier": False,
        },
        "statistical_plan": {
            "min_effect": 0.08,
            "anticipated_effect_for_power_only": 0.16,
            "noise_source": "pending six development-only jobs, three healthy and three damaged on historical structured data; score historical check only",
            "development_shuffle_seeds": [1101, 1102, 1103],
            "candidate_seeds_per_arm": [3, 4, 5, 6, 7, 8],
            "fresh_shuffle_seeds": list(range(2101, 2109)),
            "selection": "Before fresh jobs, use pooled development SD and simulate the full causal rule; choose the smallest candidate with lower 95% Monte Carlo power bound >= 0.80 at effect 0.16 over 1000 trials, seed 3101. If none passes, stop and revise before opening fresh test.",
            "zero_noise": "Stop for identical-weight or zero-SD review; do not substitute historical BANKING77 noise.",
            "repeated_looks": False,
        },
        "budget": {
            "cap_usd": 5.0,
            "approval_owner": "user permission to run necessary paid experiments in source chat",
            "max_jobs": jobs,
            "development_jobs": 6,
            "fresh_reference_jobs": 3,
            "maximum_pair_jobs": 16,
            "final_retrain_jobs": 1,
            "training_upper_estimate_usd": training_upper,
            "input_tokens_upper_per_row": max_row,
            "inference_reserve_usd": 1.0,
            "contingency_usd": 0.50,
            "estimated_total_upper_usd": training_upper + 1.5,
            "price_usd_per_m": 0.4,
            "price_status": "repository estimate, verify with provider before submission",
            "invoice_reconciled": False,
        },
        "stopping_rule": "One strong-damage persisted API investigation, one approved target/control pair and at most one final retrain. No automatic retries after ambiguous submission. Stop on cap, changed hashes, failed power gate or resource conflict. Final test opens only after stopping and recipe lock; report inconclusive outcomes without selecting another seed or recipe.",
        "reference_scope": "One healthy, mild-damaged and strong-damaged reference each; base untrained. Single reference jobs do not estimate reference variance or close the 3x3 matrix.",
        "control_scope": "Equal removed row count. Token counts are reported separately. No specificity claim without a prespecified matched-compute sensitivity if imbalance could explain the result.",
        "submission_blockers": [
            "development noise and seed count not measured",
            "provider price and immutable model metadata not verified",
            "runtime evaluator does not yet enforce the recorded model revision",
            "API execution configuration and approved protocol hash not frozen",
            "B9 MPS handoff pending",
        ],
        "paid_jobs_submitted": 0,
        "final_test_model_evaluations": 0,
    }
    if plan["budget"]["estimated_total_upper_usd"] > 5:
        raise ValueError("Conservative budget exceeds selected cap")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, plan)
    print(
        json.dumps(
            {
                "plan": str(out),
                "sha256": sha(out),
                "budget": plan["budget"],
                "token_counts": summaries,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cohort, args.snapshot, args.out)


if __name__ == "__main__":
    main()
