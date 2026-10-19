"""Nebius Token Factory: Nemotron calls with structured output and the Post-training API."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from openai import BadRequestError, OpenAI, UnprocessableEntityError
from pydantic import BaseModel, ValidationError

from core.config import Settings, get_settings
from core.connectors.storage import Storage

T = TypeVar("T", bound=BaseModel)

# $ per 1M tokens (input, output) from the Token Factory model cards
PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
# Post-training ranges from docs.tokenfactory.nebius.com/post-training/how-to-fine-tune;
# a job outside them is rejected with HTTP 422 when it is created
HP_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "n_epochs": (1, 20),
    "lora_r": (8, 128),
    "lora_alpha": (8, None),
    "learning_rate": (0, None),
    "warmup_ratio": (0, 1),
    "lora_dropout": (0, 1),
}
JSON_FORMATS = ["json_schema", "json_object", "plain text"]


def hyperparameter_problem(hp: dict[str, Any]) -> str | None:
    """Why Token Factory would reject these hyperparameters, or None."""
    for key, (low, high) in HP_LIMITS.items():
        value = hp.get(key)
        if value is None:
            continue
        if low is not None and value < low:
            return f"{key} {value} is below Token Factory's minimum of {low}"
        if high is not None and value > high:
            return f"{key} {value} is over Token Factory's limit of {high}"
    return None


@dataclass
class CallResult:
    model: str
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int


class StructuredOutputError(RuntimeError):
    """The model did not return JSON that matches the schema after all attempts."""

    def __init__(self, message: str, calls: list[CallResult] | None = None):
        super().__init__(message)
        self.calls = calls or []  # every attempt, so the cost still gets logged


def extract_json(text: str) -> str:
    """Pull the JSON out of a reply: drop reasoning tags, code fences, leading prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    return text[min(starts) :] if starts else text


class TokenFactory:
    def __init__(self, client: OpenAI, settings: Settings, *, spending=None):
        self.client = client
        self.spending = spending
        self.settings = settings
        # per model: index into JSON_FORMATS that the endpoint accepted last time
        self.format_level: dict[str, int] = {}

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> TokenFactory:
        s = s or get_settings()
        if not s.token_factory_api_key:
            raise ValueError("TOKEN_FACTORY_API_KEY must be set")
        client = OpenAI(
            api_key=s.token_factory_api_key, base_url=s.token_factory_base_url
        )
        return cls(client, s)

    def model(self, role: str) -> str:
        """Model ID for a router role: nano, super or ultra."""
        return {
            "nano": self.settings.nemotron_nano_model,
            "super": self.settings.nemotron_super_model,
            "ultra": self.settings.nemotron_ultra_model,
        }[role]

    # Inference

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> CallResult:
        extra: dict[str, Any] = {}
        if response_format is not None:
            extra["response_format"] = response_format
        started = time.monotonic()
        reservation = None
        client = self.client
        if self.spending is not None:
            if model not in PRICES_USD_PER_M:
                raise ValueError("Cannot reserve inference cost for an unpriced model")
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")
            # UTF-8 bytes overestimate content tokens; reserve additional framing.
            # This is a conservative estimate, not a provider-enforced cap.
            input_bound = (
                len(json.dumps([messages, extra], ensure_ascii=False).encode()) + 4096
            )
            price_in, price_out = PRICES_USD_PER_M[model]
            amount = (input_bound * price_in + max_tokens * price_out) / 1_000_000
            reservation = uuid.uuid4().hex
            self.spending.reserve(reservation, f"inference:{model}", amount)
            client = self.client.with_options(max_retries=0)
        reply = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
        usage = reply.usage
        if reservation is not None and usage is None:
            raise RuntimeError(
                "Provider omitted usage; inference reservation remains pending"
            )
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        price_in, price_out = PRICES_USD_PER_M.get(model, (0.0, 0.0))
        cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        if reservation is not None:
            self.spending.settle(reservation, cost)
        return CallResult(
            model=model,
            text=reply.choices[0].message.content or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def structured(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        attempts: int = 3,
        max_tokens: int = 4096,
        on_call: Callable[[CallResult], None] | None = None,
    ) -> tuple[T, list[CallResult]]:
        """Ask for JSON matching `schema`; on a mismatch, retry with the error attached."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        history = [
            {
                "role": "system",
                "content": "Reply with JSON only. It must match this JSON Schema:\n"
                + schema_json,
            },
            *messages,
        ]
        calls: list[CallResult] = []
        for _ in range(attempts):
            call = self._chat_json(model, history, schema, max_tokens)
            calls.append(call)
            if on_call is not None:
                on_call(call)
            try:
                return schema.model_validate_json(extract_json(call.text)), calls
            except ValidationError as err:
                history = [
                    *history,
                    {"role": "assistant", "content": call.text},
                    {
                        "role": "user",
                        "content": "The answer does not match the schema: "
                        + str(err)[:2000]
                        + "\nReply with corrected JSON only.",
                    },
                ]
        raise StructuredOutputError(
            f"{schema.__name__}: no valid JSON after {attempts} attempts", calls
        )

    def _chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        max_tokens: int,
    ) -> CallResult:
        # Not every model accepts json_schema: fall back to json_object, then plain text
        formats: list[dict[str, Any] | None] = [
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
            {"type": "json_object"},
            None,
        ]
        level = self.format_level.get(model, 0)
        while True:
            try:
                call = self.chat(
                    model,
                    messages,
                    max_tokens=max_tokens,
                    response_format=formats[level],
                )
            except (BadRequestError, UnprocessableEntityError):
                # 400 or 422: this endpoint rejects the format, try the next one
                if level == len(formats) - 1:
                    raise
                level += 1
                continue
            self.format_level[model] = level
            return call

    # Post-training

    def upload(self, path: str | Path) -> str:
        with open(path, "rb") as f:
            return self.client.files.create(file=f, purpose="fine-tune").id

    def create_job(
        self,
        model: str,
        training_file: str,
        *,
        validation_file: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        seed: int | None = None,
        suffix: str | None = None,
    ) -> str:
        problem = hyperparameter_problem(hyperparameters or {})
        if problem:
            raise ValueError(f"Token Factory would reject the job: {problem}")
        optional = {
            "validation_file": validation_file,
            "hyperparameters": hyperparameters,
            "seed": seed,
            "suffix": suffix,
        }
        # A timeout or server error may follow an accepted, billable submission.
        # The SDK must not replay this non-idempotent request automatically.
        job = self.client.with_options(max_retries=0).fine_tuning.jobs.create(
            model=model,
            training_file=training_file,
            **{k: v for k, v in optional.items() if v is not None},
        )
        return job.id

    def job(self, job_id: str) -> Any:
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def wait(
        self,
        job_id: str,
        *,
        poll_s: float = 30.0,
        timeout_s: float = 6 * 3600,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Any:
        deadline = time.monotonic() + timeout_s
        while True:
            job = self.job(job_id)
            if job.status in TERMINAL_STATUSES:
                return job
            if time.monotonic() > deadline:
                raise TimeoutError(f"{job_id} still {job.status} after {timeout_s} s")
            sleep(poll_s)

    def events(self, job_id: str) -> list[Any]:
        return list(self.client.fine_tuning.jobs.list_events(job_id))

    def checkpoints(self, job_id: str) -> list[Any]:
        return list(self.client.fine_tuning.jobs.checkpoints.list(job_id))

    def download_checkpoint(
        self, checkpoint: Any, storage: Storage, prefix: str
    ) -> list[str]:
        """Copy every file of a checkpoint into storage under prefix/<checkpoint id>/."""
        keys = []
        for file_id in checkpoint.result_files:
            name = Path(self.client.files.retrieve(file_id).filename).name
            data = self.client.files.content(file_id).read()
            keys.append(storage.put_bytes(f"{prefix}/{checkpoint.id}/{name}", data))
        return keys
