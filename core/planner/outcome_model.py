"""B16: P(probe outcome | fault class) from the signature table."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

SIGNATURES = Path(__file__).resolve().parents[1] / "probes" / "signatures.yaml"


@lru_cache
def load_signatures(path: Path = SIGNATURES) -> dict[str, dict[str, dict[str, float]]]:
    table = yaml.safe_load(path.read_text(encoding="utf-8"))
    for probe, rows in table.items():
        for fault, row in rows.items():
            if abs(sum(row.values()) - 1) > 1e-6:
                raise ValueError(f"{probe}/{fault}: probabilities add up to {row}")
    return table


def likelihood(probe: str, fault_class: str, outcome: str) -> float:
    rows = load_signatures()[probe]
    return rows.get(fault_class, rows["default"])[outcome]


def outcomes(probe: str) -> list[str]:
    return list(load_signatures()[probe]["default"])
