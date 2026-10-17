"""The metrics of section 10 for one method on one case, and the table per method.

- Top-1 and top-3: a true cause is first, or among the first three, in the ranking.
  Fault cases only.
- False alarm: the status is "confirmed" on a healthy case.
- Cost: LLM plus training dollars, LLM tokens and mini-jobs until the method stopped.
- Fix success: the method confirmed a cause, and the model retrained with the
  standard fix for it reaches the reference within the tolerance (bench/fixes.py).
  No confirmed cause, or a cause without a fix, is a failure. An untrained fix or a
  missing reference leaves it unknown.
- P(true cause): the probability the method gave the true cause when it stopped.
- Wrong confirmations: causes the method declared proven that are not in the truth.
  One-factor checks can prove a second cause on an undertrained job: a larger
  LoRA rank also helps a model that a low learning rate held back.

Rates are counts, such as 2/3, because the mini-benchmark has a handful of cases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bench.cases import BenchCase
from bench.diagnosis import Diagnosis
from bench.fixes import TOLERANCE


@dataclass
class Row:
    case: str
    method: str
    truth: list[str]
    status: str
    cause: str | None
    top: list[str]
    top1: bool | None  # None on a healthy case
    top3: bool | None
    p_true: float | None
    false_alarm: bool | None  # None on a fault case
    cost_usd: float
    llm_tokens: int
    mini_jobs: int
    fixed_metric: float | None = None
    reference: float | None = None
    fix_success: bool | None = None
    wrong_confirmations: int = 0


def fix_outcome(
    d: Diagnosis,
    has_fix: bool,
    fixed_metric: float | None,
    reference: float | None,
    tolerance: float = TOLERANCE,
) -> bool | None:
    if d.status != "confirmed" or not has_fix:
        return False
    if fixed_metric is None or reference is None:
        return None
    return fixed_metric >= reference - tolerance


def score(
    case: BenchCase,
    d: Diagnosis,
    *,
    has_fix: bool = False,
    fixed_metric: float | None = None,
    reference: float | None = None,
) -> Row:
    probs = dict(d.ranking)
    fault = not case.healthy
    return Row(
        case=case.id,
        method=d.method,
        truth=case.truth,
        status=d.status,
        cause=d.cause,
        top=d.top[:3],
        top1=bool(d.top) and d.top[0] in case.truth if fault else None,
        top3=any(t in d.top[:3] for t in case.truth) if fault else None,
        p_true=max(probs.get(t, 0.0) for t in case.truth) if fault else None,
        false_alarm=None if fault else d.status == "confirmed",
        cost_usd=d.llm_cost_usd + d.training_cost_usd,
        llm_tokens=d.llm_tokens,
        mini_jobs=d.mini_jobs,
        fixed_metric=fixed_metric,
        reference=reference,
        fix_success=fix_outcome(d, has_fix, fixed_metric, reference) if fault else None,
        wrong_confirmations=sum(1 for c in d.confirmed if c not in case.truth),
    )


def count(values: Sequence[bool | None]) -> tuple[int, int]:
    """(true, known): values that are None do not count."""
    known = [v for v in values if v is not None]
    return sum(known), len(known)


def mean(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def summarize(rows: Sequence[Row]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for method in dict.fromkeys(r.method for r in rows):
        mine = [r for r in rows if r.method == method]
        out[method] = {
            "cases": len(mine),
            "top1": count([r.top1 for r in mine]),
            "top3": count([r.top3 for r in mine]),
            "false_alarms": count([r.false_alarm for r in mine]),
            "fix_success": count([r.fix_success for r in mine]),
            "wrong_confirmations": sum(r.wrong_confirmations for r in mine),
            "p_true": mean([r.p_true for r in mine]),
            "cost_usd": mean([r.cost_usd for r in mine]),
            "llm_tokens": mean([r.llm_tokens for r in mine]),
            "mini_jobs": mean([r.mini_jobs for r in mine]),
        }
    return out


def _rate(pair: tuple[int, int]) -> str:
    k, n = pair
    return f"{k}/{n}" if n else "–"


def _num(value: float | None, fmt: str) -> str:
    return "–" if value is None else format(value, fmt)


def table(summary: dict[str, dict]) -> str:
    """Markdown table per method, for the README and Devpost."""
    lines = [
        (
            "| method | cases | top-1 | top-3 | false alarms | wrong confirmations "
            "| fix success | P(true cause) | $ per case | mini-jobs per case |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, s in summary.items():
        lines.append(
            f"| {method} | {s['cases']} | {_rate(s['top1'])} | {_rate(s['top3'])} "
            f"| {_rate(s['false_alarms'])} | {s['wrong_confirmations']} "
            f"| {_rate(s['fix_success'])} "
            f"| {_num(s['p_true'], '.2f')} | {_num(s['cost_usd'], '.3f')} "
            f"| {_num(s['mini_jobs'], '.1f')} |"
        )
    return "\n".join(lines)
