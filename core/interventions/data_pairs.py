"""A7: a targeted data fix and an equal-row-count control.

The target drops only probe-flagged rows. The control drops the same number of
unflagged rows, keeping the original recipe. Retained token counts can differ in
either direction: removing short damaged answers can make the target more costly.
Report actual tokens and recipes; row matching does not establish matched compute
or a unique cause for any measured improvement.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DataIntervention:
    hypothesis_id: str
    kind: Literal["target", "control"]
    drop_lines: list[int]  # line numbers of the training file, from 1
    description: str
    ft_job_ids: list[str] = field(default_factory=list)
    metrics: list[float] = field(default_factory=list)  # one value per seed

    def apply(self, rows: Sequence[Any]) -> list[Any]:
        """The training examples this arm trains on, in file order."""
        drop = set(self.drop_lines)
        if max(drop, default=0) > len(rows):
            raise ValueError(f"{self.kind}: line {max(drop)} is past the file's end")
        return [row for line, row in enumerate(rows, start=1) if line not in drop]


def answer_cut_pair(
    hypothesis_id: str, flagged: Sequence[int], examples: int, seed: int = 0
) -> tuple[DataIntervention, DataIntervention]:
    """Drop the flagged examples against as many random unflagged ones.

    flagged are the answer_cut probe's line numbers in a training file of examples
    lines. The control's draw is fixed by seed, the same for every training seed.
    """
    target_lines = sorted(set(flagged))
    if not target_lines:
        raise ValueError("the probe flagged no examples, so there is nothing to drop")
    if target_lines[0] < 1 or target_lines[-1] > examples:
        raise ValueError(f"flagged lines must be within 1..{examples}")
    unflagged = sorted(set(range(1, examples + 1)) - set(target_lines))
    n = len(target_lines)
    if len(unflagged) < n:
        raise ValueError(
            f"{n} flagged of {examples}: too few unflagged examples for a control"
        )
    control_lines = sorted(random.Random(seed).sample(unflagged, n))
    target = DataIntervention(
        hypothesis_id,
        "target",
        target_lines,
        f"drop the {n} examples whose answers look cut",
    )
    control = DataIntervention(
        hypothesis_id,
        "control",
        control_lines,
        f"drop {n} random examples whose answers look whole",
    )
    return target, control


def flagged_lines(probe_raw: dict[str, Any], role: str = "train") -> list[int]:
    """The answer_cut probe's flagged line numbers from its raw result."""
    return list(probe_raw["files"][role]["cut_lines"])
