"""A8: a level-0 probe on example lengths in tokens against the job's context_length.

Examples longer than context_length. In mode A this is a check before a job, not
the cause of a silent failure: the tested Token Factory configuration fails
the job while processing the dataset ("Sequence length is too long: 9013 > 8192"),
and it refuses a context_length below 8192 (measured on 2026-10-18). A training
file used by that succeeded job fit its provider limits; the probe matters for files about
to be trained, such as a fix or a retrain. In mode B a trainer may cut long examples
silently, so the end of the answer never reaches the loss and the model learns to
stop early: that is fault 1 (truncation) as the catalog describes it. Answers cut by
the user's own preparation before the upload are a different fault
(bench/faults/data.py) that this probe does not see. The probe renders each chat
example with the base model's chat template, counts its tokens and compares the
count with the limit. No GPU, no LLM: only the base model's tokenizer.

The tokens come from apply_chat_template(..., tokenize=True), which already adds the
template's special tokens. Rendering the text first and tokenizing it again would
add BOS a second time for models such as Llama.

Limits:

- context_length comes from the job's hyperparameters; a job that did not set it
  trained with the documented Token Factory default of 8192, and the raw result
  records which of the two the probe used;
- with packing (on by default) several short examples share one sequence; the probe
  assumes that an example longer than context_length does not fit either way, and
  it does not measure how packing splits shorter ones;
- the probe counts the examples it sees. A long example shows the risk of a failed
  job or of truncation, not that truncation made the model fail: that is the causal
  check's job.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.probes.base import Probe, ProbeResult

if TYPE_CHECKING:
    from core.agent.context import InvestigationContext

# Token Factory docs (How to fine-tune): context_length defaults to 8192
TOKEN_FACTORY_CONTEXT_LENGTH = 8192
ROLES = ("train", "validation")
LONGEST = 5  # how many of the longest examples the raw result lists

Messages = list[dict[str, str]]


@dataclass
class FileLengths:
    examples: int
    over: int  # examples with more tokens than context_length
    over_share: float
    max_tokens: int
    p50: int
    p95: int
    p99: int
    longest: list[tuple[int, int]]  # (line number, tokens), longest first


def _is_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
    )


def read_chat_jsonl(path: Path) -> list[tuple[int, Messages]]:
    """Line number and messages of every record in a chat JSONL file, checked."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: not JSON ({e.msg})") from e
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no}: no messages list")
            if not all(_is_message(message) for message in messages):
                raise ValueError(f"{path}:{line_no}: a message needs role and content")
            records.append((line_no, messages))
    if not records:
        raise ValueError(f"{path}: no examples")
    return records


def count_tokens(tokenizer: Any, messages: Messages) -> int:
    """Tokens of one example as the chat template renders it, special tokens included."""
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if hasattr(ids, "keys"):  # a BatchEncoding when return_dict is the default
        ids = ids["input_ids"]
    return len(ids)


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile of a sorted, non-empty sequence."""
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def file_lengths(path: Path, tokenizer: Any, context_length: int) -> FileLengths:
    counted = [
        (line_no, count_tokens(tokenizer, messages))
        for line_no, messages in read_chat_jsonl(path)
    ]
    lengths = sorted(tokens for _, tokens in counted)
    over = sum(tokens > context_length for tokens in lengths)
    longest = sorted(counted, key=lambda item: (-item[1], item[0]))[:LONGEST]
    return FileLengths(
        examples=len(lengths),
        over=over,
        over_share=over / len(lengths),
        max_tokens=lengths[-1],
        p50=_percentile(lengths, 0.50),
        p95=_percentile(lengths, 0.95),
        p99=_percentile(lengths, 0.99),
        longest=longest,
    )


@lru_cache(maxsize=4)
def load_tokenizer(model: str) -> Any:
    """The base model's tokenizer from the Hugging Face Hub or its local cache."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def context_limit(ctx: InvestigationContext) -> tuple[int, str]:
    """The job's context_length and where it came from: the job or the default."""
    value = ctx.job_config.get("hyperparameters", {}).get("context_length")
    if value is None:
        return TOKEN_FACTORY_CONTEXT_LENGTH, "token_factory_default"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"context_length must be a positive integer, got {value!r}")
    return value, "job"


class ExampleLength(Probe):
    """Are training examples longer in tokens than the job's context_length?

    Needs the base model in ctx.job_config["model"] (a Hugging Face ID such as
    Qwen/Qwen3-0.6B) and the training file in ctx.files["train"]: a local chat
    JSONL file. A local ctx.files["validation"] is measured too and reported, but
    the outcome rests on the training file, the one that reaches the loss.
    """

    name = "example_length"
    level = 0
    outcomes = ("truncated", "ok")
    # Legacy categorical threshold; not calibrated and not an upload approval.
    TRUNCATED = 0.01

    def __init__(self, tokenizer: Callable[[str], Any] = load_tokenizer):
        self.tokenizer = tokenizer

    def available(self, ctx: InvestigationContext) -> bool:
        train = ctx.files.get("train")
        return (
            bool(ctx.job_config.get("model")) and bool(train) and Path(train).is_file()
        )

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        start = time.perf_counter()
        model = ctx.job_config["model"]
        limit, source = context_limit(ctx)
        tokenizer = self.tokenizer(model)
        files = {
            role: asdict(file_lengths(Path(ctx.files[role]), tokenizer, limit))
            for role in ROLES
            if ctx.files.get(role) and Path(ctx.files[role]).is_file()
        }
        train = files["train"]
        outcome = "truncated" if train["over_share"] >= self.TRUNCATED else "ok"
        template = getattr(tokenizer, "chat_template", None) or ""
        raw = {
            "semantics_version": "length-preflight-v1",
            "mode": ctx.mode,
            "over_limit_found": any(f["over"] > 0 for f in files.values()),
            "interpretation": (
                "Mode A: over-limit inputs risk explicit job rejection, not silent provider truncation."
                if ctx.mode == "A"
                else "Mode B: over-limit inputs require inspecting the trainer; length alone does not prove token removal."
            ),
            "context_length": limit,
            "context_length_source": source,
            "truncated_from_share": self.TRUNCATED,
            "tokenizer": getattr(tokenizer, "name_or_path", model),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "files": files,
        }
        return self.check_outcome(
            ProbeResult(
                outcome=outcome,
                raw_ref=f"investigations/{ctx.investigation_id}/probes/{self.name}.json",
                duration_s=time.perf_counter() - start,
                summary=json.dumps(raw),
            )
        )
