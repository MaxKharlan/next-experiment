"""Bounded access to the real Nemotron investigation loop and TF pair runner."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apps.api.live_protocol import estimate_pair_cost, inspect_datasets, verify_datasets
from core.config import Settings
from core.connectors.storage import LocalStorage
from core.db.spending import BudgetExceeded, SpendingLedger

ROOT = Path(os.environ.get("JUDGE_DATA", "/tmp/next-experiment-judge"))
RUN_CAP = 1.0
MAX_BODY = 300_000
MODEL = "Qwen/Qwen3-0.6B"
BOOT = secrets.token_hex(16)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def conversation(self):
        if self.messages[-1].role != "assistant" or not any(
            m.role == "user" for m in self.messages[:-1]
        ):
            raise ValueError(
                "Each row needs a user prompt and a final assistant answer"
            )
        if any(not m.content.strip() for m in self.messages):
            raise ValueError("Messages cannot be blank")
        if len(self.model_dump_json().encode()) > 4096:
            raise ValueError("Each row must be at most 4096 UTF-8 bytes")
        return self


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    train_loss: float = Field(gt=0)
    valid_loss: float = Field(gt=0)


class Upload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task: str = Field(min_length=1, max_length=300)
    symptom: str = Field(min_length=1, max_length=2000)
    train: list[Row] = Field(min_length=16, max_length=64)
    validation: list[Row] = Field(min_length=4, max_length=16)
    check: list[Row] = Field(min_length=4, max_length=8)
    test: list[Row] = Field(min_length=4, max_length=8)
    learning_rate: float = Field(default=0.00001, ge=0.000001, le=0.0003)
    n_epochs: int = Field(default=3, ge=1, le=12)
    lora_r: Literal[8, 16, 32] = 16
    checkpoints: list[Checkpoint] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def short_answers(self):
        for role in ("validation", "check", "test"):
            if any(len(row.messages[-1].content) > 100 for row in getattr(self, role)):
                raise ValueError(
                    "This hosted runner scores short answers (at most 100 characters)"
                )
        return self


def prepare(upload: Upload, run_id: str, root: Path):
    """No caller-controlled paths, model URLs, code, or decoding settings."""
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role in ("train", "validation", "check", "test"):
        path = folder / f"{role}.jsonl"
        path.write_text(
            "\n".join(r.model_dump_json() for r in getattr(upload, role)) + "\n"
        )
        paths[role] = path
    datasets = inspect_datasets(paths)
    # Bytes plus framing bound tokens for the fixed tokenizer; reject oversized
    # runs before calling Nemotron or uploading anything to Token Factory.
    token_bound = max(len(r.model_dump_json().encode()) + 256 for r in upload.train)
    hp = {
        "learning_rate": upload.learning_rate,
        "n_epochs": upload.n_epochs,
        "lora_r": upload.lora_r,
        "lora_alpha": upload.lora_r * 2,
        "batch_size": 4,
        "context_length": 8192,
        "lora": True,
        "packing": False,
    }
    pair_bound = 4 * len(upload.train) * token_bound * upload.n_epochs * 0.4 / 1e6 * 1.2
    if pair_bound > 0.75:
        raise ValueError(
            "The pair exceeds this hosted run's allowance. Use fewer/shorter training rows or fewer epochs."
        )
    return {
        "version": 1,
        "name": "hosted-judge-v1",
        "model": MODEL,
        "job_id": "submitted-recipe",
        "original_job_verified": False,
        "hyperparameters": hp,
        "datasets": datasets,
        "seeds": [0, 1],
        "tokens_per_example": token_bound * 1.2,
        "budget_usd": RUN_CAP,
        "pair_estimate_usd": pair_bound,
        "noise_sd": 0.0146,
        "min_effect": 0.03,
        "calibration_scope": "Exploratory: the benchmark noise threshold is not calibrated for these uploaded data.",
        "evaluator": {
            "name": "intent_label",
            "metric": "exact_match",
            "parameters": {"max_new_tokens": 32},
        },
        "task": upload.task,
        "source": "submitted recipe and losses; no verification of an original provider job",
    }


class HubStore:
    """A private repository holds wallet state across hosting restarts.

    Commits use the observed parent hash: a concurrent claim fails closed.
    The wallet must be provisioned separately; the server cannot reset it.
    """

    def __init__(self):
        self.repo = os.environ["JUDGE_PRIVATE_REPO"]
        self.token = os.environ["JUDGE_HF_TOKEN"]
        self.api = HfApi(token=self.token)
        if not self.api.repo_info(self.repo, repo_type="dataset").private:
            raise RuntimeError("Judge storage must be private")
        self.lock = threading.RLock()

    def read(self, name, revision=None):
        path = hf_hub_download(
            self.repo,
            name,
            repo_type="dataset",
            token=self.token,
            revision=revision,
            force_download=revision is None,
        )
        return json.loads(Path(path).read_text())

    def commit(self, files, parent):
        self.api.create_commit(
            repo_id=self.repo,
            repo_type="dataset",
            parent_commit=parent,
            commit_message="Save private investigation state",
            operations=[
                CommitOperationAdd(
                    path_in_repo=k,
                    path_or_fileobj=json.dumps(v, ensure_ascii=False).encode(),
                )
                for k, v in files.items()
            ],
        )

    def wallet(self):
        head = self.api.repo_info(self.repo, repo_type="dataset").sha
        return head, self.read("wallet.json", head)

    def available(self):
        with self.lock:
            _, wallet = self.wallet()
            used = sum(r["reserved_usd"] for r in wallet["runs"].values())
            active = any(r["status"] == "running" for r in wallet["runs"].values())
            return {
                "available": not active and wallet["cap_usd"] - used >= RUN_CAP,
                "busy": active,
                "run_allowance_usd": RUN_CAP,
                "funded_remaining_usd": max(0, round(wallet["cap_usd"] - used, 4)),
            }

    def claim(self, run_id, key, payload_hash):
        with self.lock:
            head, w = self.wallet()
            existing = w["runs"].get(run_id)
            if existing:
                if (
                    existing["key_hash"] != digest(key)
                    or existing["payload_hash"] != payload_hash
                ):
                    raise ValueError("Run identifier already used")
                return False
            if any(r["status"] == "running" for r in w["runs"].values()):
                raise BudgetExceeded(
                    "Another investigation is running. Try again after it finishes."
                )
            if (
                sum(r["reserved_usd"] for r in w["runs"].values()) + RUN_CAP
                > w["cap_usd"]
            ):
                raise BudgetExceeded(
                    "The live-run budget is exhausted. No charge was made."
                )
            w["runs"][run_id] = {
                "key_hash": digest(key),
                "payload_hash": payload_hash,
                "reserved_usd": RUN_CAP,
                "status": "running",
                "boot": BOOT,
                "started": time.time(),
            }
            self.commit(
                {
                    "wallet.json": w,
                    f"runs/{run_id}.json": {
                        "id": run_id,
                        "status": "running",
                        "events": [],
                        "started": time.time(),
                    },
                },
                head,
            )
            return True

    def authenticated(self, run_id, key):
        with self.lock:
            _, w = self.wallet()
            r = w["runs"].get(run_id)
            if not r or not secrets.compare_digest(r["key_hash"], digest(key)):
                raise HTTPException(404, "Investigation not found")
            result = self.read(f"runs/{run_id}.json")
            if r["status"] == "running" and r["boot"] != BOOT:
                result["status"] = "interrupted"
                result["message"] = (
                    "The host restarted. Saved evidence is retained; paid work is not restarted automatically."
                )
            return result

    def save(self, run_id, state, final_spending=None):
        with self.lock:
            head, w = self.wallet()
            files = {f"runs/{run_id}.json": state}
            if state["status"] != "running":
                r = w["runs"][run_id]
                r["status"] = state["status"]
                # Unknown calls/training outcomes keep the whole reservation.
                if (
                    state["status"] == "complete"
                    and final_spending is not None
                    and all(
                        o["status"] == "settled" for o in final_spending["operations"]
                    )
                ):
                    r["reserved_usd"] = final_spending["committed_usd"]
                files["wallet.json"] = w
            self.commit(files, head)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class MirroredStorage(LocalStorage):
    def __init__(self, root, publish):
        super().__init__(root)
        self.publish = publish

    def put_bytes(self, key, data):
        result = super().put_bytes(key, data)
        if key.endswith("jobs.json"):
            self.publish({"kind": "training_jobs", "jobs": json.loads(data)})
        return result


def execute(store, run_id, upload, protocol):
    state = {"id": run_id, "status": "running", "events": [], "protocol": protocol}
    spending = SpendingLedger(ROOT / run_id / "spending.sqlite3", RUN_CAP)

    def emit(event):
        state["events"].append({"at": time.time(), **event})
        store.save(run_id, state)

    try:
        from dataclasses import asdict

        from core.agent.context import InvestigationContext
        from core.agent.router import Router
        from core.connectors.token_factory import TokenFactory
        from core.interventions.answer_check import (
            DataAwareTFPairRunner,
            investigate_with_answers,
        )
        from core.probes.level0.recipe import LEVEL0, LossCurve
        from core.telemetry.calls import CallLog

        storage = MirroredStorage(ROOT / run_id / "artifacts", emit)
        tf = TokenFactory.from_settings(Settings(_env_file=None))
        tf.spending = spending
        original_job = tf.job
        seen_jobs = {}

        def tracked_job(job_id):
            job = original_job(job_id)
            progress = (job.status, job.trained_steps, job.total_steps)
            if seen_jobs.get(job_id) != progress:
                emit(
                    {
                        "kind": "training_progress",
                        "job_id": job_id,
                        "status": job.status,
                        "trained_steps": job.trained_steps,
                        "total_steps": job.total_steps,
                    }
                )
                seen_jobs[job_id] = progress
            return job

        tf.job = tracked_job
        # A run is never replayed after a crash; each paid call has a local hold
        # inside its durable, already-reserved global allowance.
        router = Router(
            tf,
            CallLog(run_id, on_add=lambda c: emit({"kind": "llm_call", **asdict(c)})),
        )
        ctx = InvestigationContext(
            run_id,
            "A",
            "submitted-recipe",
            {
                "model": MODEL,
                "hyperparameters": protocol["hyperparameters"],
                "checkpoints": [c.model_dump() for c in upload.checkpoints],
            },
            files={r: d["path"] for r, d in protocol["datasets"].items()},
            budget_left_usd=RUN_CAP,
        )

        def score(adapter):
            emit({"kind": "evaluation_started", "adapter": Path(adapter).name})
            import subprocess
            import sys

            request_path = ROOT / run_id / "evaluation-request.json"
            output_path = ROOT / run_id / "evaluation-result.json"
            output_path.unlink(missing_ok=True)
            request_path.write_text(
                json.dumps(
                    {
                        "data": protocol["datasets"]["check"]["path"],
                        "evaluator": protocol["evaluator"],
                        "adapter": str(adapter),
                        "output": str(output_path),
                    }
                )
            )
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in {"TOKEN_FACTORY_API_KEY", "JUDGE_HF_TOKEN"}
            }
            subprocess.run(
                [sys.executable, "-m", "apps.judge.score", str(request_path)],
                env=env,
                check=True,
                timeout=900,
            )
            result = json.loads(output_path.read_text())
            emit({"kind": "evaluation", "result": result})
            return result["value"]

        pair = DataAwareTFPairRunner(
            tf=tf,
            storage=storage,
            model=MODEL,
            train_path=Path(ctx.files["train"]),
            valid_path=Path(ctx.files["validation"]),
            base_hp=protocol["hyperparameters"],
            evaluate=score,
            seeds=[0, 1],
            prefix="pair",
            protocol=protocol,
            timeout_s=3 * 3600,
        )

        def run_pair(target, control):
            verify_datasets(protocol)
            emit(
                {
                    "kind": "training_started",
                    "target": asdict(target),
                    "control": asdict(control),
                }
            )
            pair(target, control)

        run_pair.supports_answer_drop = True

        def propose_retrain(target, control):
            return {
                "status": "proposal_only",
                "hyperparameters": target.hyperparameters(protocol["hyperparameters"]),
                "description": "Export this recipe for a full retrain. Hosted allowance covers the diagnostic pair.",
            }

        emit({"kind": "started", "model": MODEL, "source": protocol["source"]})
        result = investigate_with_answers(
            ctx,
            {
                "task": upload.task,
                "symptom": upload.symptom,
                "job": ctx.job_config,
                "original_job_verified": False,
            },
            router,
            run_pair,
            probes=tuple(
                p for p in LEVEL0 if upload.checkpoints or not isinstance(p, LossCurve)
            ),
            max_interventions=1,
            spending=spending,
            on_step=emit,
            estimate_pair_cost=lambda t, c: estimate_pair_cost(protocol, t, c),
            intervention_cost_usd=protocol["pair_estimate_usd"],
            noise_sd=protocol["noise_sd"],
            min_effect=protocol["min_effect"],
            propose_retrain=propose_retrain,
        )
        # Explicit proposal without pretending that a full retrain was performed.
        state["report"] = json.loads(result.to_json())
        state["report"]["protocol"] = protocol
        state["spending"] = spending.snapshot()
        state["status"] = "complete"
        store.save(run_id, state, state["spending"])
    except Exception as error:  # noqa: BLE001 — terminal worker receipt
        # Exception text and tracebacks may contain provider URLs or internal data.
        import traceback

        diagnostic = ROOT / run_id / "diagnostic.txt"
        diagnostic.write_text(traceback.format_exc())
        diagnostic.chmod(0o600)
        state["status"] = "failed"
        state["error"] = {
            "type": type(error).__name__,
            "message": "The investigation stopped. Partial evidence is available; no automatic paid retry will run.",
        }
        state["spending"] = spending.snapshot()
        store.save(run_id, state)


app = FastAPI(
    title="Next Experiment · live investigations", docs_url=None, redoc_url=None
)
_store = None


def store():
    global _store
    if _store is None:
        _store = HubStore()
    return _store


@app.middleware("http")
async def bounded_body(request: Request, call_next):
    if request.method == "POST":
        from fastapi.responses import JSONResponse

        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BODY:
                return JSONResponse({"detail": "Upload must be at most 300 KB"}, 413)
        request._body = bytes(data)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "frame-ancestors 'self' https://maxkharlan.github.io; base-uri 'none'"
    )
    return response


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/example.zip")
def example():
    return FileResponse(
        Path(__file__).with_name("example.zip"), filename="next-experiment-example.zip"
    )


@app.get("/api/status")
def status():
    return store().available()


@app.post("/api/runs")
def start(upload: Upload, request: Request):
    key = request.headers.get("X-Run-Key", "")
    run_id = request.headers.get("X-Run-Id", "")
    import re

    if not re.fullmatch(r"[a-f0-9]{32}", run_id) or not re.fullmatch(
        r"[a-f0-9]{64}", key
    ):
        raise HTTPException(
            400, "A browser-generated run ID and access key are required"
        )
    try:
        # Validate in a temporary folder; repeated requests cannot mutate a live run.
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            protocol = prepare(upload, run_id, Path(temp))
            new = store().claim(run_id, key, digest(upload.model_dump_json()))
            if new:
                destination = ROOT / run_id
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(Path(temp) / run_id, destination)
                for role, entry in protocol["datasets"].items():
                    entry["path"] = str((destination / f"{role}.jsonl").resolve())
                threading.Thread(
                    target=execute,
                    args=(store(), run_id, upload, protocol),
                    daemon=True,
                ).start()
        return {"id": run_id, "reused": not new}
    except BudgetExceeded as error:
        raise HTTPException(429, str(error)) from None
    except ValueError as error:
        raise HTTPException(400, str(error)) from None


@app.get("/api/runs/{run_id}")
def read(run_id: str, request: Request):
    return store().authenticated(run_id, request.headers.get("X-Run-Key", ""))
