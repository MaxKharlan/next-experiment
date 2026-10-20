"""Evaluation examples from a chat JSONL file: prompt messages, reference, source."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class Example:
    line: int  # line number in the file, for the report and for a patch
    prompt: Messages  # every message but the last
    expected: str  # the final assistant message, the reference answer
    source: str | None = None  # where the input came from, when the record says


def _is_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
    )


def _source(record: dict[str, Any]) -> str | None:
    source = record.get("source")
    if source is None and isinstance(record.get("metadata"), dict):
        source = record["metadata"].get("source")
    return source if isinstance(source, str) else None


def read_examples(path: str | Path) -> list[Example]:
    """Every record of a chat JSONL file whose last message is the reference answer.

    A record may name its input's source in "source" or "metadata.source"; the
    error breakdown groups by it.
    """
    path = Path(path)
    examples = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: not JSON ({e.msg})") from e
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{path}:{line_no}: needs a prompt and an answer")
            if not all(_is_message(message) for message in messages):
                raise ValueError(f"{path}:{line_no}: a message needs role and content")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{line_no}: the last message is not assistant")
            examples.append(
                Example(
                    line_no, messages[:-1], messages[-1]["content"], _source(record)
                )
            )
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples
