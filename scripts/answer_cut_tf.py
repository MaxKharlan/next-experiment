"""Token Factory check of answer_truncation: the Kaggle experiment's arms as real jobs.

Six LoRA jobs on Qwen/Qwen3-0.6B with the healthy recipe, no packing:
healthy s0, damaged (30% of answers cut) s0, target s0/s1 (drop the examples the
answer_cut probe flags), control s0/s1 (drop as many random unflagged ones).
Token Factory ignores the job seed, so each seed is its own shuffle of the file,
as core/interventions/tf_runner.py does.

  PYTHONPATH=. .venv/bin/python scripts/answer_cut_tf.py prepare   # free
  PYTHONPATH=. .venv/bin/python scripts/answer_cut_tf.py submit    # paid
  PYTHONPATH=. .venv/bin/python scripts/answer_cut_tf.py collect   # wait, download
  PYTHONPATH=. .venv/bin/python scripts/answer_cut_tf.py evaluate  # local, check

State goes to artifacts/tf_answer_cut/state/jobs.json; a rerun of submit never
creates a job that is already there.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from bench.faults.recipe import HEALTHY_HP
from core.agent.context import InvestigationContext
from core.connectors.storage import LocalStorage
from core.interventions.data_pairs import answer_cut_pair, flagged_lines
from core.interventions.tf_runner import TF_LORA
from core.planner.job_cost import FT_PRICES_USD_PER_M
from core.probes.level0.answers import AnswerCut

MODEL = "Qwen/Qwen3-0.6B"
HERE = Path("artifacts/tf_answer_cut")
TASKS = Path("artifacts/tasks")
CLEAN, CUT = TASKS / "banking77", TASKS / "banking77-answers-cut-30"
CHECK = CLEAN / "check.jsonl"
HP = {**HEALTHY_HP, **TF_LORA}
BUDGET_USD = 1.0
storage = LocalStorage(HERE / "state")
STATE = "jobs.json"


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text(encoding="utf-8").splitlines(True) if x.strip()]


def arms() -> dict[str, tuple[list[str], int, Path]]:
    """name -> (training rows before shuffling, seed, validation file)."""
    cut = lines(CUT / "train.jsonl")
    ctx = InvestigationContext(
        "tf", "A", "tf", {}, files={"train": str(CUT / "train.jsonl")}
    )
    raw = json.loads(AnswerCut().run(ctx).summary)
    target, control = answer_cut_pair("answer_truncation", flagged_lines(raw), len(cut))
    valid_cut = CUT / "validation.jsonl"
    return {
        "healthy-s0": (lines(CLEAN / "train.jsonl"), 0, CLEAN / "validation.jsonl"),
        "damaged-s0": (cut, 0, valid_cut),
        "target-s0": (target.apply(cut), 0, valid_cut),
        "target-s1": (target.apply(cut), 1, valid_cut),
        "control-s0": (control.apply(cut), 0, valid_cut),
        "control-s1": (control.apply(cut), 1, valid_cut),
    }


def prepare() -> None:
    from transformers import AutoTokenizer

    from core.probes.level0.lengths import count_tokens

    tok = AutoTokenizer.from_pretrained(MODEL)
    price = FT_PRICES_USD_PER_M[MODEL][0]
    plan, total = {}, 0
    for name, (rows, seed, valid) in arms().items():
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        path = HERE / "files" / f"{name}-train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(rows), encoding="utf-8")
        tokens = sum(count_tokens(tok, json.loads(r)["messages"]) for r in rows)
        total += tokens
        plan[name] = {
            "examples": len(rows),
            "tokens": tokens,
            "usd": round(tokens * price / 1e6, 4),
            "train": str(path),
            "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation": str(valid),
            "seed": seed,
        }
    summary = {
        "model": MODEL,
        "hyperparameters": HP,
        "price_usd_per_m": price,
        "jobs": plan,
        "tokens": total,
        "usd": round(total * price / 1e6, 4),
        "budget_usd": BUDGET_USD,
    }
    (HERE / "plan.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _state() -> dict:
    return json.loads(storage.get_text(STATE)) if storage.exists(STATE) else {}


def _save(state: dict) -> None:
    storage.put_text(STATE, json.dumps(state, indent=2))


def submit() -> None:
    from core.connectors.token_factory import TokenFactory

    plan = json.loads((HERE / "plan.json").read_text())
    if plan["usd"] > BUDGET_USD:
        raise SystemExit(f"estimate ${plan['usd']} is over the ${BUDGET_USD} budget")
    tf = TokenFactory.from_settings()
    state = _state()
    for name, job in plan["jobs"].items():
        if name in state:
            print(name, "already submitted", state[name])
            continue
        train = Path(job["train"])
        if hashlib.sha256(train.read_bytes()).hexdigest() != job["train_sha256"]:
            raise SystemExit(f"{train} changed since prepare")
        valid_key = f"_valid:{job['validation']}"
        if valid_key not in state:
            state[valid_key] = tf.upload(job["validation"])
            _save(state)
        file_id = tf.upload(train)
        state[name] = tf.create_job(
            MODEL,
            file_id,
            validation_file=state[valid_key],
            hyperparameters=HP,
            seed=job["seed"],
            suffix=f"ne-answer-cut-{name}",
        )
        _save(state)
        print(name, "submitted", state[name], flush=True)


def collect() -> None:
    from core.connectors.token_factory import TokenFactory
    from core.interventions.tf_runner import download_last_checkpoint, wait_for_jobs

    tf = TokenFactory.from_settings()
    state = _state()
    names = [n for n in state if not n.startswith("_")]
    finished = wait_for_jobs(
        tf, {state[n] for n in names}, poll_s=60, timeout_s=4 * 3600
    )
    jobs = state.setdefault("_jobs", {})
    for name in names:
        job = finished[state[name]]
        jobs[name] = {
            "id": job.id,
            "status": job.status,
            "trained_tokens": getattr(job, "trained_tokens", None),
            "error": str(getattr(job, "error", None)),
        }
        if job.status == "succeeded" and name not in state.get("_adapters", {}):
            adapter = download_last_checkpoint(tf, storage, job.id, "checkpoints")
            state.setdefault("_adapters", {})[name] = str(adapter)
        _save(state)
        print(name, jobs[name], flush=True)


def evaluate() -> None:
    from core.case import EvaluatorSpec
    from core.eval import evaluate as score

    state = _state()
    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    spec = EvaluatorSpec(name="intent_label", metric="exact_match")
    for name, adapter in state.get("_adapters", {}).items():
        if name in results:
            continue
        result = score(CHECK, spec, adapter=adapter, base_model=MODEL)
        results[name] = {
            "exact_match": result.value,
            "trained_tokens": state["_jobs"][name]["trained_tokens"],
            "job_id": state["_jobs"][name]["id"],
        }
        results_path.write_text(json.dumps(results, indent=2))
        print(name, results[name], flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "submit": submit, "collect": collect, "evaluate": evaluate}[
        sys.argv[1]
    ]()
