"""B10 on Token Factory: score the adapters of every causal check again on a check set,
and pool the noise between seeds.

For each case of a split it reads the pairs that bench/run_bench.py cached
(artifacts/bench/<split>/<case>/pairs.json), finds each job's adapter in local
storage and scores it on --check. Nothing trains. The metrics go under the same keys
to pairs-<check name>.json, which run_bench reads with --check-set. Adapters with the
same weights are scored once: the control of every pair on a case is the failed
recipe on the same files, and Token Factory training is deterministic.

The noise floor is the pooled SD between the seeds of each arm, on the validation
file and on the check set, counting each set of weights once. An arm whose seeds
trained identical weights (runs from before the per-seed shuffling) shows no seed
noise and is left out. Pass the check-set SD to run_bench as --noise-sd.

    uv run --group ml --group eval python -m scripts.rescore_pairs --split dev
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bench.cases import load_cases
from core.connectors.storage import LocalStorage
from core.interventions.stats import NoiseFloor, noise_floor
from core.interventions.tf_runner import weights_digest


def find_adapter(
    job_id: str,
    adapters: dict[str, str],
    storage: LocalStorage,
    tf_factory: Callable[[], Any],
    prefix: str,
) -> Path:
    """The job's adapter: on record, already on disk, or downloaded now."""
    if job_id in adapters:
        return storage.root / adapters[job_id]
    tf = tf_factory()
    last = max(tf.checkpoints(job_id), key=lambda c: c.step_number)
    found = next(storage.root.glob(f"**/{last.id}/adapter_model.safetensors"), None)
    if found is not None:
        return found.parent
    keys = tf.download_checkpoint(last, storage, f"{prefix}/rescore")
    return storage.root / Path(keys[0]).parent


def rescore(
    pairs: dict[str, Any],
    adapter_of: Callable[[str], Path],
    score: Callable[[Path], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pairs with the new metrics, and one row per arm for the noise floor."""
    scored: dict[str, float] = {}  # weights SHA-256 -> metric
    out: dict[str, Any] = {}
    arms: list[dict[str, Any]] = []
    for key, pair in pairs.items():
        out[key] = {}
        for kind, arm in pair.items():
            metrics, weights = [], []
            for job_id in arm["ft_job_ids"]:
                adapter = adapter_of(job_id)
                digest = weights_digest(adapter) or str(adapter)
                if digest not in scored:
                    scored[digest] = score(adapter)
                metrics.append(scored[digest])
                weights.append(digest)
            out[key][kind] = {"metrics": metrics, "ft_job_ids": arm["ft_job_ids"]}
            arms.append(
                {
                    "pair": key,
                    "arm": kind,
                    "valid": arm["metrics"],
                    "check": metrics,
                    "weights": weights,
                }
            )
    return out, arms


def seed_noise(arms: list[dict[str, Any]], field: str) -> NoiseFloor | None:
    """Pooled SD between seeds; identical sets of weights count once."""
    groups: dict[str, list[float]] = {}
    for arm in arms:
        if len(set(arm["weights"])) < len(arm["weights"]):
            continue  # its seeds trained the same weights
        groups["|".join(arm["weights"])] = arm[field]
    return noise_floor(groups) if groups else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "heldout"])
    parser.add_argument("--check", default="artifacts/b10/check.jsonl")
    parser.add_argument("--cases", nargs="+", help="only these case IDs")
    args = parser.parse_args()

    from core.connectors.token_factory import TokenFactory
    from jobs.eval_local.exact_match import evaluate_adapter

    storage = LocalStorage("artifacts")
    clients: list[TokenFactory] = []

    def tf_factory() -> TokenFactory:
        if not clients:
            clients.append(TokenFactory.from_settings())
        return clients[0]

    name = Path(args.check).stem
    all_arms: list[dict[str, Any]] = []
    for case in load_cases(args.split):
        if args.cases and case.id not in args.cases:
            continue
        prefix = f"bench/{args.split}/{case.id}"
        if not storage.exists(f"{prefix}/pairs.json"):
            print(f"{case.id}: no pairs yet")
            continue
        adapters: dict[str, str] = {}
        for state in (storage.root / prefix / "tf").glob("*/jobs.json"):
            adapters |= json.loads(state.read_text()).get("_adapters", {})
        new, arms = rescore(
            json.loads(storage.get_text(f"{prefix}/pairs.json")),
            lambda job_id, a=adapters, p=prefix: find_adapter(
                job_id, a, storage, tf_factory, p
            ),
            lambda adapter: evaluate_adapter(adapter, args.check),
        )
        storage.put_text(f"{prefix}/pairs-{name}.json", json.dumps(new, indent=2))
        for arm in arms:
            target = json.loads(arm["pair"])[0]
            print(
                f"{case.id} {target} {arm['arm']}: valid {arm['valid']} "
                f"-> {name} {arm['check']}"
            )
        all_arms += [{**arm, "case": case.id} for arm in arms]

    summary = {}
    for field in ("valid", "check"):
        floor = seed_noise(all_arms, field)
        summary[field] = asdict(floor) if floor else None
        if floor:
            label = "validation" if field == "valid" else name
            print(
                f"noise floor on {label}: SD {floor.sd:.4f} "
                f"from {floor.runs} runs in {floor.groups} arms"
            )
    storage.put_text(
        f"bench/{args.split}/noise-{name}.json",
        json.dumps({"summary": summary, "arms": all_arms}, indent=2),
    )


if __name__ == "__main__":
    main()
