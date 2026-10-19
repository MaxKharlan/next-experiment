"""A5: data faults as deterministic changes to chat JSONL examples.

Fault 1 in mode A: answers cut by the user's own data preparation. Token Factory
does not cut long examples: a job with an example longer than context_length fails
while it processes the dataset ("Sequence length is too long: 9013 > 8192"), and a
context_length below 8192 is refused when the job is created (measured on
2026-10-18). A user who meets that error, or who trims data to save tokens, cuts
examples in their own code before the upload. A cut that keeps the start of an
example removes the end of the assistant answer; the job then succeeds, and the
model learns cut answers. Examples over the length limit are a different problem,
caught before the job by the A8 length probe.

truncate_answers() cuts the final assistant answer of a share of the examples;
the share is the fault's strength. build() applies it to the training and
validation files of a task directory, which the user prepares with the same code,
and copies check and test unchanged: they hold the answers the model should give.
The record of what was cut goes to a separate file for the evaluator only; the task
directory the agent reads has no trace of it.

    python -m bench.faults.data --task artifacts/tasks/banking77 \
        --out artifacts/tasks/banking77-answers-cut-30 --strength 0.3
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

# fault class -> named strengths: the share of examples whose answer is cut. A
# named injection strengths. answer_truncation is a separate catalog class from
# legacy custom-trainer truncation; manifests are evaluation-only ground truth.
ANSWER_TRUNCATION = "answer_truncation"
STRENGTHS: dict[str, dict[str, float]] = {
    ANSWER_TRUNCATION: {"mild": 0.1, "strong": 0.3},
}
FAULTED_ROLES = ("train", "validation")

Select = Literal["longest", "random"]


def example_chars(row: dict[str, Any]) -> int:
    return sum(len(m["content"]) for m in row["messages"])


def cut(answer: str, keep: float) -> str:
    """The first share keep of the answer, at least one character, never all of it."""
    if len(answer) < 2:
        raise ValueError("an answer shorter than two characters cannot be cut")
    size = min(len(answer) - 1, max(1, math.floor(len(answer) * keep)))
    return answer[:size]


def pick(
    rows: list[dict[str, Any]], count: int, seed: int, select: Select
) -> list[int]:
    """Indices of the examples to cut, in file order.

    "longest" takes the examples with the most characters, as a cut to a length
    budget would; ties break by a seeded shuffle. "random" takes a seeded sample.
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if select == "longest":
        order.sort(key=lambda i: -example_chars(rows[i]))
    elif select != "random":
        raise ValueError(f"unknown selection {select!r}")
    return sorted(order[:count])


def truncate_answers(
    rows: list[dict[str, Any]],
    strength: float,
    seed: int = 0,
    keep: float = 0.5,
    select: Select = "longest",
) -> tuple[list[dict[str, Any]], list[int]]:
    """A copy of rows with the answer cut in a share strength of them, and their indices.

    The answer is the final message, which must come from the assistant; keep is
    the share of its characters that stays.
    """
    if not 0 < strength <= 1:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if not 0 < keep < 1:
        raise ValueError(f"keep must be in (0, 1), got {keep}")
    if not rows:
        raise ValueError("no examples to change")
    for i, row in enumerate(rows):
        if row["messages"][-1]["role"] != "assistant":
            raise ValueError(f"example {i}: the final message is not the answer")
    count = max(1, round(len(rows) * strength))
    changed = pick(rows, count, seed, select)
    out = copy.deepcopy(rows)
    for i in changed:
        answer = out[i]["messages"][-1]
        answer["content"] = cut(answer["content"], keep)
    return out, changed


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    raw = data.encode("utf-8")
    path.write_bytes(raw)
    return {
        "ref": path.name,
        "format": "chat_jsonl",
        "examples": len(rows),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def record_path(out: Path) -> Path:
    """Where build() puts the record of the fault: next to the task directory."""
    return out.with_name(out.name + ".fault.json")


def build(
    task: Path,
    out: Path,
    strength: float,
    seed: int = 0,
    keep: float = 0.5,
    select: Select = "longest",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A new task directory with the fault in its training and validation files.

    Returns its manifest and the fault record. The manifest is the source task's
    with each file's new hash. The record, written to record_path(out), names the
    fault and its parameters, lists the changed examples by line number from 1 and
    the hash of each file it describes.
    """
    record_file = record_path(out)
    if record_file.exists():
        raise FileExistsError(f"{record_file} already exists")
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "class": ANSWER_TRUNCATION,
        "source": str(task),
        "task": str(out),
        "strength": strength,
        "seed": seed,
        "keep": keep,
        "select": select,
        "changed_lines": {},
        "sha256": {},
    }
    # A new directory keeps earlier benchmark files and manifests intact.
    out.mkdir(parents=True, exist_ok=False)
    for role, spec in manifest["datasets"].items():
        rows = read_rows(task / spec["ref"])
        if role in FAULTED_ROLES:
            # each file gets its own draw, so the same seed cuts unrelated examples
            role_seed = seed * 1000 + FAULTED_ROLES.index(role)
            rows, changed = truncate_answers(rows, strength, role_seed, keep, select)
            record["changed_lines"][role] = [i + 1 for i in changed]
        manifest["datasets"][role] = write_rows(rows, out / spec["ref"])
        if role in FAULTED_ROLES:
            record["sha256"][role] = manifest["datasets"][role]["sha256"]
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    record_file.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--select", choices=["longest", "random"], default="longest")
    args = parser.parse_args()
    if args.out.exists() or record_path(args.out).exists():
        parser.error("output directory or fault record exists; choose a new --out")
    manifest, record = build(
        args.task, args.out, args.strength, args.seed, args.keep, args.select
    )
    for role, lines in record["changed_lines"].items():
        total = manifest["datasets"][role]["examples"]
        print(f"{role}: {len(lines)} of {total} answers cut")
    print(args.out / "manifest.json")
    print(record_path(args.out))


if __name__ == "__main__":
    main()
