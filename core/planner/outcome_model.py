"""B16: P(probe outcome | fault class) from the signature table."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import yaml

SIGNATURES = Path(__file__).resolve().parents[1] / "probes" / "signatures.yaml"

_ACTIVE = ContextVar("signature_path", default=SIGNATURES)


@contextmanager
def using_signatures(path: Path):
    """Select a frozen table for this context only; restore it even after failure."""
    path = Path(path).resolve()
    load_signatures(path)
    token = _ACTIVE.set(path)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@lru_cache
def load_signatures(path: Path = SIGNATURES) -> dict[str, dict[str, dict[str, float]]]:
    table = yaml.safe_load(path.read_text(encoding="utf-8"))
    for probe, rows in table.items():
        for fault, row in rows.items():
            if abs(sum(row.values()) - 1) > 1e-6:
                raise ValueError(f"{probe}/{fault}: probabilities add up to {row}")
    return table


def likelihood(probe: str, fault_class: str, outcome: str) -> float:
    rows = load_signatures(_ACTIVE.get())[probe]
    return rows.get(fault_class, rows["default"])[outcome]


def outcomes(probe: str) -> list[str]:
    return list(load_signatures(_ACTIVE.get())[probe]["default"])
