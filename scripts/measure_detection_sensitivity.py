"""Measure frozen agent top-1 recipe recovery before any causal intervention."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from openai import APIError

from bench.diagnosis import from_investigation
from bench.sensitivity import detection_report
from core.agent.context import InvestigationContext
from core.agent.loop import investigate
from core.agent.router import Router
from core.connectors.token_factory import StructuredOutputError, TokenFactory
from core.telemetry.calls import CallLog


def observation(row, healthy):
    if "source" in row:
        path = Path(row["source"]).with_name("metrics.jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        train = {r["step"]: r["train_loss"] for r in records if "train_loss" in r}
        checkpoints = [
            {
                "step": r["step"],
                "train_loss": train[r["step"]],
                "valid_loss": r["valid_loss"],
            }
            for r in records
            if "valid_loss" in r and r["step"] in train
        ]
    else:
        checkpoints = [
            {"step": s, "train_loss": t, "valid_loss": v}
            for s, t, v in zip(
                row["steps"], row["train_loss"], row["valid_loss"], strict=True
            )
        ]
    return {
        "task": "BANKING77 intent classification",
        "backend": "local Transformers/PEFT"
        if "local" in row["study"]
        else "Token Factory",
        "symptom": "Assess the recipe and validation behavior. Quality may be unchanged from the reference; do not assume a harmful fault exists.",
        "model": "Qwen/Qwen3-0.6B",
        "job": {"hyperparameters": row["hyperparameters"]},
        "checkpoints": checkpoints,
        "validation_exact_match": row["exact_match"],
        "reference_validation_exact_match": healthy["exact_match"],
        "evaluation_examples": 100,
    }


def no_training(*args):
    raise AssertionError("Detection protocol must not train interventions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = Path("docs/results/b467-20261019")
    rows = []
    for name in ("local-sensitivity", "epoch-sensitivity"):
        rows.extend(json.loads((source / f"{name}.json").read_text())["inputs"])
    healthy = {(r["study"], r["seed"]): r for r in rows if r["fault"] is None}
    cases = [
        {
            "id": f"case-{i:02d}",
            "truth": r,
            "observation": observation(r, healthy[r["study"], r["seed"]]),
        }
        for i, r in enumerate(rows)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "frozen agent ranking before causal intervention; no confirmation claim",
        "case_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest(),
        "cases": len(cases),
        "budget_usd": 2.4,
        "per_case_reserve_usd": 0.1,
        "hypothesis_prompt": "hypotheses-v3",
        "causal_checks": False,
        "explain": False,
        "threshold": 0.6,
    }
    path = args.out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Detection inputs or protocol changed")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.execute:
        return
    tf = TokenFactory.from_settings()
    tf.client = tf.client.with_options(timeout=90, max_retries=0)
    measured = []
    for c in cases:
        output = args.out / f"{c['id']}.json"
        calls = args.out / f"{c['id']}-calls.jsonl"
        if output.exists():
            measured.append(json.loads(output.read_text()))
            continue
        if calls.exists():
            raise RuntimeError("Partial paid case exists; reconcile it before resuming")

        def persist(call, path=calls):
            with path.open("a") as handle:
                handle.write(json.dumps(asdict(call)) + "\n")

        router = Router(tf, CallLog(c["id"], on_add=persist))
        obs = c["observation"]
        truth = c["truth"]
        if len(json.dumps(obs).encode()) > 16000:
            raise ValueError("Case exceeds inference envelope")
        context = InvestigationContext(
            investigation_id=c["id"],
            mode="A",
            job_id="opaque",
            job_config={**obs["job"], "checkpoints": obs["checkpoints"]},
            budget_left_usd=0.1,
        )
        row = {
            "case_id": c["id"],
            "study": truth["study"],
            "seed": truth["seed"],
            "fault": truth["fault"],
            "hyperparameters": truth["hyperparameters"],
            "observation": obs,
        }
        try:
            result = investigate(
                context, obs, router, no_training, causal_checks=False, explain=False
            )
            diagnosis = from_investigation(result)
            row.update(
                ranking=diagnosis.ranking,
                trace=json.loads(result.to_json()),
                error=None,
            )
        except (APIError, StructuredOutputError) as exc:
            row.update(ranking=[], error=type(exc).__name__ + ": " + str(exc))
        row["cost_usd"] = router.log.cost_usd
        output.write_text(json.dumps(row, indent=2) + "\n")
        measured.append(row)
        print(c["id"], truth["fault"], row["ranking"][:1], row["cost_usd"], flush=True)
        if row["cost_usd"] > 0.1:
            raise RuntimeError("Per-case inference envelope exceeded; stop")
    (args.out / "report.json").write_text(
        json.dumps(detection_report(measured), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
