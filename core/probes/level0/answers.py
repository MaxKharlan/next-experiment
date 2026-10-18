"""A level-0 probe for answers cut by the user's own data preparation.

Fault answer_truncation (bench/faults/data.py): code that trims examples before the
upload keeps their start and drops the end of the assistant answer. Token Factory
trains on such a file without complaint, and the model learns to give cut answers.
The A8 length probe does not see it: the cut examples are short.

The probe reads the final assistant answer of every example and flags two kinds of
cut answer, with no model and no access to the original data:

- a prefix: an answer that is a strict prefix of another answer in the same file
  and stops inside a word ("card_a" next to "card_arrival"), or stops at a separator
  and is no more frequent than the longer answer. Labels and other repeated answers
  are whole words from a small set, so a part of one is a cut;
- broken JSON: when most answers in the file are JSON, an answer that opens like
  JSON and does not parse.

Limits:

- free-text answers that never repeat give the prefix rule nothing to compare, and
  a cut that ends on a full sentence looks like a short answer;
- if every example of a label was cut the same way, the whole label is missing and
  the prefix rule does not see it;
- a cut that stops at a separator and is used more often than the full answer looks
  like a label of its own. On BANKING77 with 30% of answers cut, the probe flags
  2308 of the 2536 cut answers and nothing else; the 228 it misses are
  "wrong_exchange_rate" and "cash_withdrawal", cut from longer labels;
- a flagged answer shows that the data is damaged, not that the damage made the model
  fail: that is the causal check's job.

The flagged line numbers are in the raw result, so a fix can drop those examples
without knowing what the answers should have been.
"""

from __future__ import annotations

import bisect
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.probes.base import Probe, ProbeResult
from core.probes.level0.lengths import read_chat_jsonl

if TYPE_CHECKING:
    from core.agent.context import InvestigationContext

ROLES = ("train", "validation")
EXAMPLES = 5  # how many flagged answers the raw result shows with their full form
JSON_MAJORITY = 0.9  # share of answers that must be JSON for the JSON rule to apply


@dataclass
class FileAnswers:
    examples: int
    distinct: int
    json_file: bool
    cut: int
    cut_share: float
    cut_lines: list[int]  # line numbers from 1, in file order
    shown: list[dict[str, str | int]]  # line, answer and why it looks cut


def answers(path: Path) -> list[tuple[int, str]]:
    """Line number and final answer of every example; the answer must be the assistant's."""
    out = []
    for line_no, messages in read_chat_jsonl(path):
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"{path}:{line_no}: the final message is not the answer")
        out.append((line_no, messages[-1]["content"].strip()))
    return out


def _mid_word(answer: str, longer: str) -> bool:
    """Whether answer stops inside a word of longer or right after a separator."""
    return not answer[-1].isalnum() or longer[len(answer)].isalnum()


def prefix_cuts(counts: Counter[str]) -> dict[str, str]:
    """Answers that look like a cut of a longer answer -> that answer.

    A prefix that stops inside a word ("card_a" of "card_arrival") is a cut. One that
    stops at a separator ("pay" of "pay_later") can be a label of its own, so it
    counts as a cut only when the longer answer is at least as frequent.
    """
    ordered = sorted(counts)
    found = {}
    for answer in ordered:
        if not answer:
            continue
        # every answer that starts with this one sorts right after it
        i = bisect.bisect_right(ordered, answer)
        while i < len(ordered) and ordered[i].startswith(answer):
            longer = ordered[i]
            if _mid_word(answer, longer) or counts[longer] >= counts[answer]:
                found[answer] = longer
                break
            i += 1
    return found


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _opens_like_json(text: str) -> bool:
    return text[:1] in ("{", "[")


def file_answers(path: Path) -> FileAnswers:
    rows = answers(path)
    counts = Counter(answer for _, answer in rows)
    json_like = sum(_opens_like_json(a) and _parses(a) for _, a in rows)
    json_file = json_like >= JSON_MAJORITY * len(rows)
    prefixes = prefix_cuts(counts)
    cut_lines, shown = [], []
    for line_no, answer in rows:
        if json_file and _opens_like_json(answer) and not _parses(answer):
            why = "JSON that does not parse"
        elif answer in prefixes:
            why = f"prefix of {prefixes[answer]!r}"
        else:
            continue
        cut_lines.append(line_no)
        if len(shown) < EXAMPLES:
            shown.append({"line": line_no, "answer": answer, "why": why})
    return FileAnswers(
        examples=len(rows),
        distinct=len(counts),
        json_file=json_file,
        cut=len(cut_lines),
        cut_share=len(cut_lines) / len(rows),
        cut_lines=cut_lines,
        shown=shown,
    )


class AnswerCut(Probe):
    """Are some final answers in the training data cut short?

    Needs the training file in ctx.files["train"]: a local chat JSONL file. A local
    ctx.files["validation"] is checked too and reported, but the outcome rests on
    the training file, the one that reaches the loss.
    """

    name = "answer_cut"
    level = 0
    outcomes = ("cut", "ok")
    # share of training answers that look cut; a first guess, calibrated in A14
    CUT = 0.01

    def available(self, ctx: InvestigationContext) -> bool:
        train = ctx.files.get("train")
        return bool(train) and Path(train).is_file()

    def estimate_cost(self, ctx: InvestigationContext) -> float:
        return 0.0

    def run(self, ctx: InvestigationContext) -> ProbeResult:
        start = time.perf_counter()
        files = {
            role: asdict(file_answers(Path(ctx.files[role])))
            for role in ROLES
            if ctx.files.get(role) and Path(ctx.files[role]).is_file()
        }
        outcome = "cut" if files["train"]["cut_share"] >= self.CUT else "ok"
        raw = {"cut_from_share": self.CUT, "files": files}
        return self.check_outcome(
            ProbeResult(
                outcome=outcome,
                raw_ref=f"investigations/{ctx.investigation_id}/probes/{self.name}.json",
                duration_s=time.perf_counter() - start,
                summary=json.dumps(raw),
            )
        )
