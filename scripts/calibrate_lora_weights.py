"""B25: weight statistics of local adapters whose recipe is known, and the threshold.

Three sources, each labelled by whether the learning rate was too low (fault 6):

- the B9 sweep trained on this MacBook (artifacts/b9/sweep-500), with known fault
  and strength;
- the failed job of every Token Factory case (artifacts/bench/cases/<case>/), dev
  for the ref- cases and held-out for the ho- cases;
- with --arms, every arm of every causal check the benchmark trained. Matching a
  checkpoint to its job asks Token Factory (read only), as scripts/rescore_pairs.py does.

Identical weights count once: Token Factory training is deterministic, so a control
that repeats a recipe on the same files, or a dropped case that shares a job with a
kept one, trains a copy.

The threshold of the "small" outcome is set on the dev cases only, halfway on a log
scale between the largest fault-6 job and the smallest other job, and then checked
on the held-out cases and the arms. Rows go to artifacts/b25/weights.jsonl.

    uv run --group eval python -m scripts.calibrate_lora_weights --arms
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.interventions.tf_runner import weights_digest
from core.probes.level0.weights import base_norms, update_stats

CASES_DIR = Path("bench/cases")
LOW_LR = 2e-5  # every fault-6 recipe here is at or below it, every healthy one at 1e-4 or more


def case_rows(art: Path) -> Iterator[dict[str, Any]]:
    for d in sorted((art / "bench/cases").glob("*/")):
        adapter = next(d.glob("*/adapter_model.safetensors"), None)
        if adapter is None:
            continue
        case = next(CASES_DIR.glob(f"*/{d.name}.json"), None)
        hp = json.loads(case.read_text())["hyperparameters"] if case else {}
        yield {
            "source": "case",
            "split": "dev" if d.name.startswith("ref-") else "heldout",
            "name": d.name,
            "lr": hp.get("learning_rate"),
            "epochs": hp.get("n_epochs"),
            # cases dropped by make_case keep no JSON; their names say the fault
            "fault6": "fault6" in d.name,
            "adapter": adapter.parent,
        }


def b9_rows(art: Path) -> Iterator[dict[str, Any]]:
    for d in sorted((art / "b9/sweep-500").glob("*/")):
        summary = json.loads((d / "summary.json").read_text())
        hp = summary["hyperparameters"]
        yield {
            "source": "b9",
            "split": "b9",
            "name": d.name,
            "lr": hp["learning_rate"],
            "epochs": hp["n_epochs"],
            "fault6": summary["fault"] == "lr_too_low",
            "adapter": d / "adapter",
        }


def arm_rows(art: Path, tf: Any) -> Iterator[dict[str, Any]]:
    for split in ("dev", "heldout"):
        for case_file in sorted((CASES_DIR / split).glob("*.json")):
            case = json.loads(case_file.read_text())
            prefix = art / "bench" / split / case["id"]
            pairs_file = prefix / "pairs-check.json"
            if not pairs_file.exists():
                continue
            local = {
                p.parent.name: p.parent
                for p in prefix.glob("tf/*/*/adapter_model.safetensors")
            }
            for key, pair in json.loads(pairs_file.read_text()).items():
                patches = json.loads(key)
                for kind, patch in zip(("target", "control"), patches):
                    hp = {**case["hyperparameters"], **patch}
                    for job_id in pair[kind]["ft_job_ids"]:
                        last = max(tf.checkpoints(job_id), key=lambda c: c.step_number)
                        if last.id not in local:
                            continue
                        yield {
                            "source": "arm",
                            "split": split,
                            "name": f"{case['id']} {kind} {json.dumps(patch)}",
                            "lr": hp["learning_rate"],
                            "epochs": hp["n_epochs"],
                            "fault6": hp["learning_rate"] <= LOW_LR,
                            "adapter": local[last.id],
                        }


def threshold(rows: list[dict[str, Any]]) -> float:
    """Halfway on a log scale between the largest fault-6 norm and the smallest other."""
    low = max(r["rel_median"] for r in rows if r["fault6"])
    high = min(r["rel_median"] for r in rows if not r["fault6"])
    if low >= high:
        raise ValueError(f"no threshold separates the rows: {low} >= {high}")
    return math.sqrt(low * high)


def accuracy(rows: list[dict[str, Any]], small_below: float) -> str:
    right = sum((r["rel_median"] < small_below) == r["fault6"] for r in rows)
    return f"{right}/{len(rows)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument(
        "--arms", action="store_true", help="also every causal-check arm"
    )
    args = parser.parse_args()
    art = Path(args.artifacts)

    sources = [b9_rows(art), case_rows(art)]
    if args.arms:
        from core.connectors.token_factory import TokenFactory

        sources.append(arm_rows(art, TokenFactory.from_settings()))
    rows, seen = [], set()
    for source in sources:
        for row in source:
            adapter = row.pop("adapter")
            # controls that repeat a recipe on the same files train the same weights
            digest = weights_digest(adapter) or str(adapter)
            if digest in seen:
                continue
            seen.add(digest)
            config = json.loads((adapter / "adapter_config.json").read_text())
            stats = update_stats(adapter, base_norms(config["base_model_name_or_path"]))
            row.update(asdict(stats))
            rows.append(row)
            print(
                f"{row['source']:4} {row['split']:7} {row['name'][:48]:48} lr {row['lr']!s:8} "
                f"r {row['r']:<3} fault6 {row['fault6']!s:5} rel {row['rel_median']:.4f} "
                f"erank/r {row['erank_ratio']:.2f}",
                flush=True,
            )
    out = art / "b25" / "weights.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    dev = [r for r in rows if r["source"] == "case" and r["split"] == "dev"]
    small_below = threshold(dev)
    print(f"\nthreshold from the {len(dev)} dev cases: {small_below:.4f}")
    for label, group in [
        (
            "held-out cases",
            [r for r in rows if r["source"] == "case" and r["split"] == "heldout"],
        ),
        ("arms", [r for r in rows if r["source"] == "arm"]),
        ("B9 on this MacBook", [r for r in rows if r["source"] == "b9"]),
    ]:
        if group:
            print(f"{label}: {accuracy(group, small_below)} on the right side of it")
    print(f"Rows: {out}")


if __name__ == "__main__":
    main()
