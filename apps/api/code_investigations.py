"""Opt-in REST execution for the four supported local trainer code faults."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apps.api.live_protocol import inspect_datasets, verify_datasets
from core.connectors.token_factory import TokenFactory
from core.eval.data import read_examples
from core.eval.runtime import EvaluationRuntime, file_digest, verify_runtime
from core.interventions.code_faults import CODE_FAULTS
from core.probes.level0.overlap import normalize

SOURCES = (
    "apps/api/code_investigations.py",
    "core/interventions/code_faults.py",
    "jobs/kaggle/train_lora.py",
    "jobs/eval_local/exact_match.py",
    "core/eval/generate.py",
    "core/eval/metrics.py",
    "core/training/gradients.py",
    "core/training/parameters.py",
    "bench/faults/labels.py",
)


class SingleAttemptTokenFactory(TokenFactory):
    """Keep schema-repair calls inside the REST study's no-paid-retry policy."""

    def structured(self, *args, **kwargs):
        kwargs["attempts"] = 1
        return super().structured(*args, **kwargs)


class LocalCodeRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    n_epochs: int = Field(default=3, ge=1, le=6, strict=True)
    batch_size: int = Field(default=2, ge=1, le=8, strict=True)
    gradient_accumulation_steps: int = Field(default=4, ge=2, le=16, strict=True)
    learning_rate: float = Field(default=2e-4, gt=0, le=0.001)
    warmup_ratio: float = Field(default=0, ge=0, lt=1)
    weight_decay: float = Field(default=0, ge=0)
    max_grad_norm: float = Field(default=1, gt=0)
    lora_r: int = Field(default=16, ge=8, le=128, strict=True)
    lora_alpha: int = Field(default=16, ge=8, le=256, strict=True)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    context_length: int = Field(default=512, ge=32, le=8192, strict=True)


class CodeInvestigationStart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    title: str = Field(default="Local code investigation", min_length=1, max_length=200)
    symptom: str = Field(min_length=1, max_length=4000)
    train: str
    validation: str
    check: str
    evaluation_runtime: EvaluationRuntime
    recipe: LocalCodeRecipe = Field(default_factory=LocalCodeRecipe)
    code_fault: Literal["prompt_loss", "eos_as_pad", "double_shift", "microbatch_mean"]
    seeds: list[int] = Field(
        default_factory=lambda: [4201, 4202, 4203], min_length=2, max_length=6
    )
    min_effect: float = Field(default=0.05, gt=0, le=1)
    budget_usd: float = Field(default=1, gt=0, le=25)
    device: Literal["cpu", "mps", "cuda"] = "cpu"

    @field_validator("seeds", mode="before")
    @classmethod
    def valid_seeds(cls, value):
        if (
            not isinstance(value, list)
            or any(type(seed) is not int or seed < 0 for seed in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("Require distinct nonnegative integer seeds")
        return value


def prepare(req):
    verify_runtime(req.evaluation_runtime, model="Qwen/Qwen3-0.6B")
    datasets = inspect_datasets(
        {name: getattr(req, name) for name in ("train", "validation", "check")}
    )
    seen = {}
    for role, entry in datasets.items():
        if entry["examples"] > 2000:
            raise ValueError(
                "Local code investigations support at most 2000 rows per split"
            )
        for example in read_examples(entry["path"]):
            key = tuple(
                (message["role"], normalize(message["content"]))
                for message in example.prompt
            )
            if key in seen and seen[key] != role:
                raise ValueError("Prompt shared across dataset splits")
            seen[key] = role
    # Reject truncation before training. This does not load model weights.
    from transformers import AutoTokenizer

    from jobs.kaggle.train_lora import encode

    tokenizer = AutoTokenizer.from_pretrained(
        req.evaluation_runtime.snapshot, local_files_only=True
    )
    for role, entry in datasets.items():
        for line in Path(entry["path"]).read_text().splitlines():
            messages = json.loads(line)["messages"]
            ids = tokenizer.apply_chat_template(messages, tokenize=True)
            if len(ids) > req.recipe.context_length:
                raise ValueError(f"{role} example would be truncated")
            encode(tokenizer, messages, req.recipe.context_length)
    return {
        "model": "Qwen/Qwen3-0.6B",
        "datasets": datasets,
        "evaluation_runtime": req.evaluation_runtime.model_dump(),
        "request": req.model_dump(),
        "source_sha256": {name: file_digest(name) for name in SOURCES},
        "scope": "Local supported code switches; new target/control training, no final-test data. Three-seed defaults are exploratory, not a population recovery guarantee.",
    }


class AllowanceLedger:
    """Use the caller's global ledger, including its 20% allowance."""

    def __init__(self, ledger, request_ledger=None):
        self.ledger = ledger
        self.request_ledger = request_ledger

    def reserve(self, operation, purpose, amount):
        if self.request_ledger is not None:
            self.request_ledger.reserve(operation, purpose, amount * 1.2)
        # If the global reservation fails, the request hold stays conservative.
        return self.ledger.reserve(operation, purpose, amount * 1.2)

    def settle(self, operation, amount):
        if self.request_ledger is not None:
            self.request_ledger.settle(operation, amount * 1.2)
        return self.ledger.settle(operation, amount * 1.2)

    def record_provider_receipt(self, *args, **kwargs):
        return self.ledger.record_provider_receipt(*args, **kwargs)


def configured_ledger():
    from core.db.spending import SpendingLedger

    path = os.environ.get("NEXT_EXPERIMENT_SPENDING_LEDGER")
    cap = os.environ.get("NEXT_EXPERIMENT_SPENDING_CAP_USD")
    if not path or not cap:
        raise ValueError(
            "Configure the authorized global spending ledger and cap before local code investigations"
        )
    return SpendingLedger(path, float(cap))


@contextmanager
def model_execution_lock(path=Path("artifacts/.code-investigation-model.lock")):
    """Serialize model execution across local API workers without paid retries."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another local code investigation is running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_prepared(prepared):
    verify_datasets(prepared)
    verify_runtime(prepared["evaluation_runtime"], model=prepared["model"])
    for name, expected in prepared["source_sha256"].items():
        if file_digest(name) != expected:
            raise ValueError(f"Code investigation source changed: {name}")


def run(store, inv_id, prepared):
    from apps.api.runners import _fail

    try:
        with model_execution_lock():
            _run_locked(store, inv_id, prepared)
    except Exception as error:  # noqa: BLE001
        _fail(store, inv_id, error)


def _run_locked(store, inv_id, prepared):
    from apps.api.runners import _fail, _summary
    from core.agent.context import InvestigationContext
    from core.agent.router import Router
    from core.db.spending import SpendingLedger
    from core.interventions.code_faults import (
        LocalCodePairRunner,
        LocalLoRAExecutor,
        investigate_code,
    )
    from core.telemetry.calls import CallLog

    ledger = None
    try:
        ledger = configured_ledger()
        verify_prepared(prepared)
        req = CodeInvestigationStart.model_validate(prepared["request"])
        request_ledger = SpendingLedger(
            Path("artifacts/code-investigation-budgets") / f"{inv_id}.sqlite3",
            req.budget_usd,
        )
        tf = SingleAttemptTokenFactory.from_settings()
        tf.client = tf.client.with_options(max_retries=0, timeout=120)
        tf.spending = AllowanceLedger(ledger, request_ledger)
        tf.before_paid = lambda: store.assert_owner(inv_id)
        router = Router(
            tf,
            CallLog(
                inv_id,
                on_add=lambda call: store.emit(inv_id, "llm_call", asdict(call)),
            ),
        )
        hp = req.recipe.model_dump()
        ctx = InvestigationContext(
            inv_id,
            "B",
            "local:" + inv_id,
            {
                "model": prepared["model"],
                "hyperparameters": hp,
                "code_fault": req.code_fault,
            },
            files={key: value["path"] for key, value in prepared["datasets"].items()},
            budget_left_usd=min(req.budget_usd, ledger.snapshot()["available_usd"]),
            connectors=SimpleNamespace(),
        )
        case = {
            "mode": "B",
            "symptom": req.symptom,
            "job": {"model": prepared["model"], "hyperparameters": hp},
            "code_observation": {
                "supported_switch": req.code_fault,
                "fault_class": next(
                    k for k, v in CODE_FAULTS.items() if v == req.code_fault
                ),
            },
            "scope": prepared["scope"],
        }
        store.storage.put_text(
            f"investigations/{inv_id}/code-protocol.json",
            json.dumps(prepared, indent=2),
        )
        store.emit(
            inv_id, "started", {"case": case, "backend": "local", "protocol": prepared}
        )
        executor = LocalLoRAExecutor(
            Path(req.train),
            Path(req.validation),
            Path(req.check),
            req.evaluation_runtime.snapshot,
            req.device,
        )
        runner = LocalCodePairRunner(
            hp, req.seeds, Path("artifacts/code-investigations") / inv_id, executor
        )

        def owned_pair(target, control):
            store.assert_owner(inv_id)
            verify_prepared(prepared)
            result = runner(target, control)
            verify_prepared(prepared)
            return result

        result = investigate_code(
            ctx,
            case,
            router,
            owned_pair,
            probes=(),
            noise_sd=None,
            min_effect=req.min_effect,
            on_step=lambda step: store.emit(inv_id, "step", step),
            max_interventions=1,
        )
        result.protocol = prepared
        report = json.loads(result.to_json())
        report["evidence_scope"] = prepared["scope"]
        report["spending"] = ledger.snapshot()
        report["request_spending"] = request_ledger.snapshot()
        store.storage.put_text(
            f"investigations/{inv_id}/report.json", json.dumps(report, indent=2)
        )
        store.emit(
            inv_id, "finished", {**_summary(report), "spending": ledger.snapshot()}
        )
    except Exception as error:  # noqa: BLE001 - persist the failure and end the stream
        _fail(store, inv_id, error, spending=ledger)
