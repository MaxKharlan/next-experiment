"""Lexical train/validation candidates; not calibrated semantic fault evidence."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from core.eval.data import Example, read_examples

VERSION = "lexical-overlap-v1"
THRESHOLD = 0.85
GRAM_SIZE = 5
MIN_CHARACTERS = 40
DEFAULT_MAX_PAIRS = 500_000


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class Prompt:
    messages: tuple[tuple[str, str], ...]
    grams: frozenset[tuple[int, str]]
    characters: int


def represent(example: Example) -> Prompt:
    messages = tuple((m["role"], normalize(m["content"])) for m in example.prompt)
    if not any(role == "user" and text for role, text in messages):
        raise ValueError(f"Line {example.line}: needs a nonempty user prompt")
    if any(role not in {"system", "user", "assistant"} for role, _ in messages):
        raise ValueError(f"Line {example.line}: unsupported prompt role")
    grams = frozenset(
        (index, text[start : start + GRAM_SIZE])
        for index, (role, text) in enumerate(messages)
        if role != "system"
        for start in range(max(0, len(text) - GRAM_SIZE + 1))
    )
    return Prompt(
        messages,
        grams,
        sum(len(text) for role, text in messages if role != "system"),
    )


def compare(left: Prompt, right: Prompt) -> tuple[str, float] | None:
    if left.messages == right.messages:
        return "normalized_exact", 1.0
    if (
        tuple(role for role, _ in left.messages)
        != tuple(role for role, _ in right.messages)
        or min(left.characters, right.characters) < MIN_CHARACTERS
    ):
        return None
    if any(
        ltext != rtext
        for (role, ltext), (_, rtext) in zip(left.messages, right.messages)
        if role == "system"
    ):
        return None
    union = left.grams | right.grams
    score = len(left.grams & right.grams) / len(union) if union else 0.0
    return ("character_5gram", score) if score >= THRESHOLD else None


def find_candidates(
    train: list[Example],
    validation: list[Example],
    *,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> dict:
    """Fail above the exhaustive-search cap; never return a sampled clean result."""
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    if not train or not validation:
        raise ValueError("Both splits must contain examples")
    for rows in (train, validation):
        if len({row.line for row in rows}) != len(rows):
            raise ValueError("Line numbers must be unique within each split")
    comparisons = len(train) * len(validation)
    if comparisons > max_pairs:
        raise ValueError(f"{comparisons} pairs exceed max_pairs={max_pairs}")
    train_repr, valid_repr = (
        [represent(r) for r in train],
        [represent(r) for r in validation],
    )
    candidates = []
    for row, represented in zip(validation, valid_repr):
        for reference, reference_repr in zip(train, train_repr):
            match = compare(reference_repr, represented)
            if match is None:
                continue
            kind, score = match
            candidates.append(
                {
                    "train_line": reference.line,
                    "validation_line": row.line,
                    "kind": kind,
                    "similarity": score,
                    "reference_answers_equal": reference.expected == row.expected,
                }
            )
    lines = sorted({pair["validation_line"] for pair in candidates})
    return {
        "method": VERSION,
        "threshold": THRESHOLD,
        "gram_size": GRAM_SIZE,
        "min_characters": MIN_CHARACTERS,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "pairs_compared": comparisons,
        "max_pairs": max_pairs,
        "candidate_pairs": candidates,
        "validation_lines": lines,
        "validation_candidate_share": len(lines) / len(validation),
        "scope": "Lexical candidates for review; no semantic or model-quality conclusion",
    }


def audit_files(train: Path, validation: Path, *, max_pairs=DEFAULT_MAX_PAIRS) -> dict:
    if train.resolve() == validation.resolve():
        raise ValueError("Train and validation must be different files")
    digests = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in (("train", train), ("validation", validation))
    }
    result = find_candidates(
        read_examples(train), read_examples(validation), max_pairs=max_pairs
    )
    after = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in (("train", train), ("validation", validation))
    }
    if digests != after:
        raise ValueError("Dataset changed during inspection")
    result["dataset_sha256"] = digests
    return result
