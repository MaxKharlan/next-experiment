"""Causal checks shared between the methods of the benchmark.

Two methods that test the same fix on the same case would train the same pair. The
cache keeps the metrics and job IDs of each pair by the target's and the control's
patches, so the second method reads them instead of training again. Every method
still counts those jobs and their price as its own: the cache saves the team money,
it does not make a method cheaper in the results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.agent.loop import RunPair
from core.connectors.storage import Storage
from core.interventions.pairs import Intervention


class PairNotCached(LookupError):
    pass


def pair_key(target: dict[str, Any], control: dict[str, Any]) -> str:
    return json.dumps([target, control], sort_keys=True)


@dataclass
class PairCache:
    storage: Storage
    key: str  # storage key of the cache file
    run_pair: RunPair | None = None  # None: a pair that is not cached is an error

    def load(self) -> dict[str, Any]:
        if not self.storage.exists(self.key):
            return {}
        return json.loads(self.storage.get_text(self.key))

    def add(self, target: Intervention, control: Intervention) -> None:
        cache = self.load()
        cache[pair_key(target.patch, control.patch)] = {
            arm.kind: {"metrics": arm.metrics, "ft_job_ids": arm.ft_job_ids}
            for arm in (target, control)
        }
        self.storage.put_text(self.key, json.dumps(cache, indent=2))

    def __call__(self, target: Intervention, control: Intervention) -> None:
        found = self.load().get(pair_key(target.patch, control.patch))
        if found is None:
            if self.run_pair is None:
                raise PairNotCached(
                    f"the pair {target.description!r} / {control.description!r} "
                    "is not cached"
                )
            self.run_pair(target, control)
            self.add(target, control)
            return
        for arm in (target, control):
            arm.metrics = list(found[arm.kind]["metrics"])
            arm.ft_job_ids = list(found[arm.kind]["ft_job_ids"])


def seed_cache(cache: PairCache, report_path: Path) -> int:
    """Add the pairs of a saved investigation that the cache does not have."""
    if not report_path.exists():
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    known = cache.load()
    added = 0
    for step in report["steps"]:
        if step["kind"] != "intervention":
            continue
        target = Intervention(**step["target"])
        control = Intervention(**step["control"])
        if pair_key(target.patch, control.patch) not in known:
            cache.add(target, control)
            added += 1
    return added
