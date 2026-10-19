"""Four-arm development experiment; format learning is measured separately."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import random
import sqlite3
import subprocess
from pathlib import Path

from bench.tasks.format_content import ARMS, build
from bench.tasks.structured import sha
from core.connectors.storage import LocalStorage
from core.db.spending import SpendingLedger
from core.eval.data import read_examples
from core.eval.format_content import (
    evaluate_example,
    generation_metrics,
    selected_score,
    summary,
)
from core.eval.generate import free_cache, load_model
from core.eval.runtime import verify_runtime
from core.interventions.tf_runner import download_last_checkpoint, weights_digest
from core.probes.level0.lengths import count_tokens
from scripts.fresh_answer_calibration import provider, save
from scripts.fresh_answer_power import decide as power_decide
from scripts.ticket_update_development import SOURCES as PRIOR_SOURCES

SEEDS = (1301, 1302, 1303)
SOURCES = (
    *PRIOR_SOURCES,
    "bench/tasks/format_content.py",
    "core/eval/format_content.py",
    "scripts/format_content_development.py",
    "core/connectors/token_factory.py",
    "core/interventions/tf_runner.py",
)


def prepare(root, old):
    from transformers import AutoTokenizer

    root.mkdir(parents=True, exist_ok=False)
    prior = json.loads((old / "protocol.json").read_text())
    if json.loads((old / "decision.json").read_text())["status"] != "stopped":
        raise ValueError("Predecessor must remain terminal")
    runtime = prior["evaluation_runtime"]
    verify_runtime(runtime)
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["snapshot"], local_files_only=True
    )
    datasets = build(root / "data", tokenizer)
    recipe = {**prior["recipe"], "n_epochs": 2}
    jobs = {}
    for arm in ARMS:
        rows = Path(datasets[f"{arm}_train"]["path"]).read_text().splitlines()
        for seed in SEEDS:
            order = list(rows)
            random.Random(seed).shuffle(order)
            path = root / f"{arm}-s{seed}-train.jsonl"
            path.write_text("\n".join(order) + "\n")
            tokens = [
                count_tokens(tokenizer, json.loads(row)["messages"]) for row in order
            ]
            jobs[f"{arm}-s{seed}"] = {
                "train": str(path.resolve()),
                "train_sha256": sha(path),
                "validation": datasets[f"{arm}_validation"]["path"],
                "validation_sha256": datasets[f"{arm}_validation"]["sha256"],
                "seed": seed,
                "examples": len(rows),
                "max_tokens": max(tokens),
                "rendered_tokens": sum(tokens),
                "reservation_usd": len(rows) * max(tokens) * 2 * 0.4 / 1e6,
            }
    source = sqlite3.connect(
        f"file:{(old / 'spending.sqlite3').resolve()}?mode=ro", uri=True
    )
    destination = sqlite3.connect(root / "spending.sqlite3")
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    inherited = SpendingLedger(root / "spending.sqlite3", 5).snapshot()
    if any(r["status"] != "settled" for r in inherited["operations"]):
        raise ValueError("Unresolved inherited spending")
    save(root / "inherited-spending.json", inherited)
    reservation = sum(j["reservation_usd"] for j in jobs.values())
    if reservation + 1.5 > inherited["available_usd"]:
        raise ValueError("Series and reserves exceed inherited cap")
    protected = [
        *sorted((root / "data").glob("*.json")),
        root / "inherited-spending.json",
    ]
    plan = {
        "id": "format-content-development-20261019-v1",
        "phase": "development_only",
        "model": prior["model"],
        "evaluation_runtime": runtime,
        "recipe": recipe,
        "source_sha256": {p: sha(Path(p)) for p in SOURCES},
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "datasets": datasets,
        "jobs": jobs,
        "protected_sha256": {str(p.resolve()): sha(p) for p in protected},
        "price": {
            **prior["price"],
            "rechecked_at": "2026-10-19",
            "source": "https://tokenfactory.nebius.com/organization/prices",
        },
        "cap_usd": 5,
        "pilot_reservation_usd": reservation,
        "budget": {
            "inherited_committed_usd": inherited["committed_usd"],
            "future_inference_reserve_usd": 1,
            "contingency_usd": 0.5,
            "source": str((old / "spending.sqlite3").resolve()),
            "policy": "Full-history copy; this ledger is the sole successor. No new submissions against predecessors.",
        },
        "predecessor_protocol_sha256": sha(old / "protocol.json"),
        "metrics": {
            "format": "24 direct field serializations: strict exact and raw schema validity.",
            "content": "48 histories: strict full-object exact; canonical fenced-json unwrap diagnostic with all examples denominator; four-choice conditional logprob discrimination of product/priority with all other fields fixed to reference.",
            "choice_scoring": "Canonical JSON candidates equal in tokenizer length. Prompt tokens then separately tokenized candidate, sum conditional logprobs excluding EOS. Unique highest wins, ties within 1e-6 count incorrect. Candidate set fixed before evaluation.",
            "generation": "Greedy, thinking off, max_new_tokens=192; same runtime for all arms and base.",
        },
        "gates": {
            "format_exact_minimum": 0.9,
            "healthy_choice_minimum": 0.35,
            "healthy_choice_maximum": 0.95,
            "healthy_minus_format_choice": 0.10,
            "healthy_minus_wrong_choice": 0.10,
            "healthy_minus_cut_strict": 0.08,
        },
        "power_rule": {**prior["power_rule"], "random_seed": 3301},
        "stopping": "No base admission gate. Run all 12 planned trainings once, 3 order shuffles per arm; no adaptive repeats or replacements. Evaluate base plus all successful adapters. Any failed gate stops confirmation. Distinct weights per arm and nonzero pooled healthy/cut strict-score SD required before power selection. Stop on hash drift, ambiguous submission, budget failure or failed training; reconcile holds. A pass only prepares a separate confirmatory protocol, never opens a final test.",
        "scope": "Development only; no final data. Shared synthetic grammar. Format-only arm has different input lengths and is not compute matched. Choice task measures two fields with others supplied; not equivalent to open-ended reasoning. T=1; classifiers offline.",
    }
    save(root / "protocol.json", plan)
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")
    print(
        json.dumps(
            {
                "protocol_sha256": sha(root / "protocol.json"),
                "reservation_usd": reservation,
                "inherited_spend": inherited["committed_usd"],
            }
        ),
        flush=True,
    )


def load(root):
    digest = (root / "protocol.sha256").read_text().strip()
    if sha(root / "protocol.json") != digest:
        raise ValueError("Protocol drift")
    plan = json.loads((root / "protocol.json").read_text())
    for path, expected in {**plan["source_sha256"], **plan["protected_sha256"]}.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen source/evidence drift: {path}")
    for entry in plan["datasets"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Data drift")
    for job in plan["jobs"].values():
        for role in ("train", "validation"):
            if sha(Path(job[role])) != job[f"{role}_sha256"]:
                raise ValueError("Training data drift")
    verify_runtime(plan["evaluation_runtime"])
    state = (
        json.loads((root / "state.json").read_text())
        if (root / "state.json").exists()
        else {}
    )
    return plan, state, digest


def submit(root, env):
    plan, state, digest = load(root)
    if (root / "decision.json").exists():
        raise ValueError("Terminal experiment")
    ledger = SpendingLedger(root / "spending.sqlite3", 5)
    remaining = sum(
        j["reservation_usd"] for n, j in plan["jobs"].items() if n not in state
    )
    if ledger.snapshot()["available_usd"] < remaining + 1.5:
        raise ValueError("Training series plus reserves no longer fits")
    tf = provider(env)
    for name, job in plan["jobs"].items():
        if name in state:
            if state[name].get("job_id"):
                continue
            raise ValueError("Ambiguous submission: reconcile, do not retry")
        operation = f"{plan['id']}:{name}"
        ledger.reserve(
            operation, "format_content_development_training", job["reservation_usd"]
        )
        state[name] = {
            "status": "submitting",
            "operation_id": operation,
            "protocol_sha256": digest,
        }
        save(root / "state.json", state)
        train_id = tf.upload(Path(job["train"]))
        state[name]["training_file"] = train_id
        save(root / "state.json", state)
        valid_id = tf.upload(Path(job["validation"]))
        state[name]["validation_file"] = valid_id
        save(root / "state.json", state)
        job_id = tf.create_job(
            plan["model"],
            train_id,
            validation_file=valid_id,
            hyperparameters=plan["recipe"],
            seed=job["seed"],
            suffix=f"ne-fc-{name}",
        )
        state[name].update(job_id=job_id, status="submitted")
        save(root / "state.json", state)
        print(name, job_id, flush=True)
    save(root / "spending.json", ledger.snapshot())


def collect(root, env):
    plan, state, _ = load(root)
    tf = provider(env)
    ledger = SpendingLedger(root / "spending.sqlite3", 5)
    for name, entry in state.items():
        if not entry.get("job_id"):
            raise ValueError("Unresolved submission")
        job = tf.job(entry["job_id"])
        entry["status"] = job.status
        save(root / "provider" / f"{name}.json", job.model_dump())
        if job.status == "succeeded":
            if not entry.get("adapter"):
                adapter = download_last_checkpoint(
                    tf, LocalStorage(root), job.id, "checkpoints"
                )
                entry.update(
                    adapter=str(adapter.resolve()),
                    weights_sha256=weights_digest(adapter),
                )
            if not entry.get("settled"):
                if type(job.trained_tokens) is not int or job.trained_tokens < 0:
                    raise ValueError("Missing provider token count")
                cost = job.trained_tokens * plan["price"]["usd_per_million"] / 1e6
                # A crash after settlement is reconciled against the existing operation.
                operation = next(
                    r
                    for r in ledger.snapshot()["operations"]
                    if r["id"] == entry["operation_id"]
                )
                if operation["status"] == "pending":
                    ledger.settle(entry["operation_id"], cost)
                elif operation["actual_usd"] != cost:
                    raise ValueError("Settled cost mismatch")
                entry.update(
                    settled=True,
                    trained_tokens=job.trained_tokens,
                    estimated_cost_usd=cost,
                )
        save(root / "state.json", state)
        print(name, job.status, flush=True)
    save(root / "spending.json", ledger.snapshot())


def checked_score(path, plan, digest, entry):
    record = json.loads(path.read_text())
    expected_binding = {
        "protocol_sha256": digest,
        "runtime": plan["evaluation_runtime"],
        "job_id": entry.get("job_id"),
        "weights_sha256": entry.get("weights_sha256"),
    }
    if any(record.get(k) != v for k, v in expected_binding.items()):
        raise ValueError("Score binding differs")
    choices = json.loads(
        Path(plan["datasets"]["check_content"]["path"])
        .with_name("choices.json")
        .read_text()
    )
    for kind in ("format", "content"):
        examples = read_examples(plan["datasets"][f"check_{kind}"]["path"])
        rows = record[kind]["rows"]
        if (
            len(rows) != len(examples)
            or record[kind]["data_sha256"]
            != plan["datasets"][f"check_{kind}"]["sha256"]
        ):
            raise ValueError("Incomplete or mismatched score")
        for i, (e, row) in enumerate(zip(examples, rows, strict=True)):
            if e.expected != row["expected"] or e.line != row["line"]:
                raise ValueError("Example binding changed")
            if any(
                row[k] != v
                for k, v in generation_metrics(row["answer"], e.expected).items()
            ):
                raise ValueError("Generation metric changed")
            if kind == "content" and (
                len(row["logprobs"]) != 4
                or row["choice_exact"]
                != selected_score(row["logprobs"], choices[i]["correct_index"])
            ):
                raise ValueError("Choice metric changed")
        if record[kind]["summary"] != summary(rows):
            raise ValueError("Summary differs")
    return record


def score(root, base=False):
    plan, state, digest = load(root)
    if (root / "decision.json").exists():
        raise ValueError("Terminal experiment")
    if not base and (
        set(state) != set(plan["jobs"])
        or any(e["status"] != "succeeded" for e in state.values())
    ):
        raise ValueError("Require all twelve successful jobs")
    choices = json.loads((root / "data/choices.json").read_text())
    for name, entry in ({"base": {}} if base else state).items():
        output = root / "scores" / f"{name}.json"
        if (
            entry.get("adapter")
            and weights_digest(Path(entry["adapter"])) != entry["weights_sha256"]
        ):
            raise ValueError("Adapter drift")
        if output.exists():
            checked_score(output, plan, digest, entry)
            continue
        verify_runtime(
            plan["evaluation_runtime"],
            model=plan["model"],
            template_dir=entry.get("adapter"),
        )
        model, tokenizer, _ = load_model(
            entry.get("adapter"), base_model=plan["evaluation_runtime"]["snapshot"]
        )
        model.to("mps").eval()
        record = {
            "protocol_sha256": digest,
            "runtime": plan["evaluation_runtime"],
            "job_id": entry.get("job_id"),
            "weights_sha256": entry.get("weights_sha256"),
        }
        for kind in ("format", "content"):
            data = plan["datasets"][f"check_{kind}"]
            rows = []
            for i, example in enumerate(read_examples(data["path"])):
                rows.append(
                    evaluate_example(
                        model,
                        tokenizer,
                        example,
                        "mps",
                        choices[i] if kind == "content" else None,
                    )
                )
                if (i + 1) % 12 == 0:
                    print(name, kind, i + 1, flush=True)
                    save(
                        root / "progress.json",
                        {"name": name, "kind": kind, "completed": i + 1},
                    )
                free_cache("mps")
            record[kind] = {
                "rows": rows,
                "summary": summary(rows),
                "data_sha256": data["sha256"],
            }
        save(output, record)
        checked_score(output, plan, digest, entry)
        print(
            name, {k: record[k]["summary"] for k in ("format", "content")}, flush=True
        )
        del model, tokenizer
        gc.collect()
        free_cache("mps")


def gate_failures(groups, rules):
    def mean(arm, kind, metric):
        values = groups[arm][kind][metric]
        return sum(values) / len(values)

    failures = []
    for arm in ("format", "healthy"):
        if mean(arm, "format", "strict_exact") < rules["format_exact_minimum"]:
            failures.append(f"{arm}_serialization_not_learned")
    healthy = mean("healthy", "content", "choice_exact")
    if (
        not rules["healthy_choice_minimum"]
        <= healthy
        <= rules["healthy_choice_maximum"]
    ):
        failures.append("healthy_choice_floor_or_ceiling")
    for arm in ("format", "wrong"):
        if (
            healthy - mean(arm, "content", "choice_exact")
            < rules[f"healthy_minus_{arm}_choice"]
        ):
            failures.append(f"insufficient_choice_gap_{arm}")
    if (
        mean("healthy", "content", "strict_exact")
        - mean("cut", "content", "strict_exact")
        < rules["healthy_minus_cut_strict"]
    ):
        failures.append("insufficient_cut_sensitivity")
    return failures


def analyze(root):
    plan, state, digest = load(root)
    if (root / "decision.json").exists() or set(state) != set(plan["jobs"]):
        raise ValueError("Terminal or incomplete experiment")
    base = checked_score(root / "scores/base.json", plan, digest, {})
    groups = {a: {k: {} for k in ("format", "content")} for a in ARMS}
    digests = {a: [] for a in ARMS}
    for name, entry in state.items():
        if (
            entry["status"] != "succeeded"
            or not entry.get("settled")
            or weights_digest(Path(entry["adapter"])) != entry["weights_sha256"]
        ):
            raise ValueError("Incomplete training provenance")
        record = checked_score(root / "scores" / f"{name}.json", plan, digest, entry)
        arm = name.split("-", 1)[0]
        digests[arm].append(entry["weights_sha256"])
        for kind in ("format", "content"):
            for metric, value in record[kind]["summary"].items():
                groups[arm][kind].setdefault(metric, []).append(value)
    failures = gate_failures(groups, plan["gates"])
    if any(len(set(values)) != 3 for values in digests.values()):
        failures.append("identical_weights")
    power = None
    if not failures:
        power = power_decide(
            {
                "healthy": groups["healthy"]["content"]["strict_exact"],
                "damaged": groups["cut"]["content"]["strict_exact"],
            },
            {"healthy": digests["healthy"], "damaged": digests["cut"]},
            plan["power_rule"],
        )
    result = {
        "protocol_sha256": digest,
        "groups": groups,
        "base": {k: base[k]["summary"] for k in ("format", "content")},
        "weight_digests": digests,
        "failures": failures,
        "power": power,
        "status": "stopped" if failures else power["status"],
        "selected_seeds": [] if failures else power["selected_seeds"],
        "next_phase": "New confirmatory protocol only if all gates pass; historical final test stays protected.",
    }
    save(root / "decision.json", result)
    save(
        root / "completion.json",
        {
            "protocol_sha256": digest,
            "status": result["status"],
            "paid_jobs": len(state),
            "scored_models": len(state) + 1,
        },
    )
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "submit", "collect", "base", "score", "analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    # One process may mutate a cohort at a time; stale lockfiles do not hold flock.
    args.root.parent.mkdir(parents=True, exist_ok=True)
    with (args.root.parent / f".{args.root.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            prepare(args.root, args.old)
        elif args.action in ("submit", "collect"):
            {"submit": submit, "collect": collect}[args.action](
                args.root, args.env_file
            )
        elif args.action in ("base", "score"):
            score(args.root, args.action == "base")
        else:
            analyze(args.root)


if __name__ == "__main__":
    main()
