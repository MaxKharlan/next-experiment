"""Analyze the completed B9 local sweep without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bench.cases import load_cases
from bench.loss_classifier import evaluate, fit, load_runs, predict
from bench.loss_transfer import evaluate as evaluate_transfer

RECIPES = (
    "healthy",
    "lr_too_low-mild",
    "lr_too_low-strong",
    "too_many_epochs-mild",
    "too_many_epochs-strong",
    "lora_rank_too_small-mild",
    "lora_rank_too_small-strong",
)


def recipe_name(row: dict) -> str:
    """Return the recipe part of ``<recipe>-s<seed>``."""
    suffix = f"-s{row['seed']}"
    if not row["id"].endswith(suffix):
        raise ValueError(f"Run id does not end with its seed: {row['id']}")
    return row["id"][: -len(suffix)]


def strength_tier(row: dict) -> str:
    recipe = recipe_name(row)
    if recipe == "healthy":
        return "healthy"
    tier = recipe.rsplit("-", 1)[-1]
    if tier not in {"mild", "strong"}:
        raise ValueError(f"Unknown strength tier in {recipe}")
    return tier


def validate_matrix(rows: list[dict], seeds: tuple[int, ...]) -> None:
    expected = {(recipe, seed) for recipe in RECIPES for seed in seeds}
    actual = {(recipe_name(row), row["seed"]) for row in rows}
    if len(actual) != len(rows):
        raise ValueError("Duplicate recipe/seed row")
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Incomplete sweep: missing={missing}, unexpected={unexpected}"
        )


def _classification(predictions: list[dict]) -> dict:
    labels = sorted(
        {row["truth"] for row in predictions}
        | {row["predicted"] for row in predictions}
    )
    confusion = {truth: {predicted: 0 for predicted in labels} for truth in labels}
    for row in predictions:
        confusion[row["truth"]][row["predicted"]] += 1
    recalls = [
        confusion[label][label] / sum(confusion[label].values()) for label in labels
    ]
    return {
        "n": len(predictions),
        "accuracy": sum(r["truth"] == r["predicted"] for r in predictions)
        / len(predictions),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion": confusion,
        "predictions": predictions,
    }


def strength_transfer(rows: list[dict], split_seed: int) -> dict:
    """Test the other strength on disjoint seeds in four symmetric folds."""
    low = {row["seed"] for row in rows if row["seed"] < split_seed}
    high = {row["seed"] for row in rows if row["seed"] >= split_seed}
    if not low or not high:
        raise ValueError("Both seed halves must be nonempty")
    halves = {"low": low, "high": high}
    specs = (
        ("mild", "low", "strong", "high"),
        ("strong", "high", "mild", "low"),
        ("mild", "high", "strong", "low"),
        ("strong", "low", "mild", "high"),
    )
    predictions, folds = [], []
    for train_tier, train_half, test_tier, test_half in specs:
        train = [
            row
            for row in rows
            if row["seed"] in halves[train_half]
            and (row["label"] == "healthy" or strength_tier(row) == train_tier)
        ]
        test = [
            row
            for row in rows
            if row["seed"] in halves[test_half]
            and (row["label"] == "healthy" or strength_tier(row) == test_tier)
        ]
        model = fit(train)
        fold_name = f"{train_tier}-{train_half}_to_{test_tier}-{test_half}"
        folds.append(
            {
                "name": fold_name,
                "train_ids": [row["id"] for row in train],
                "test_ids": [row["id"] for row in test],
                "model": model,
            }
        )
        predictions.extend(
            {
                "fold": fold_name,
                "id": row["id"],
                "seed": row["seed"],
                "truth": row["label"],
                "predicted": predict(model, row["features"]),
            }
            for row in test
        )
    report = _classification(predictions)
    report.update(
        {
            "protocol": (
                "four symmetric folds; train and test use disjoint seed halves and "
                "opposite fault strengths; healthy rows follow the same seed split"
            ),
            "folds": folds,
            "healthy_rows_are_tested_twice": True,
        }
    )
    return report


def _mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_outcomes(
    runs: Path,
    *,
    expected_per_recipe: int = 30,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20_261_019,
) -> dict:
    by_recipe: dict[str, list[dict]] = {recipe: [] for recipe in RECIPES}
    for path in sorted(runs.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        recipe = path.parent.name.rsplit("-s", 1)[0]
        if recipe not in by_recipe:
            raise ValueError(f"Unexpected recipe: {recipe}")
        by_recipe[recipe].append(
            {
                "seed": summary["seed"],
                **summary["final"],
                "wall_seconds": summary["wall_seconds"],
            }
        )
    if any(len(rows) != expected_per_recipe for rows in by_recipe.values()):
        raise ValueError(f"Expected {expected_per_recipe} outcome rows for each recipe")

    healthy = {row["seed"]: row for row in by_recipe["healthy"]}
    rng = np.random.default_rng(bootstrap_seed)
    recipes = {}
    for recipe, rows in by_recipe.items():
        rows.sort(key=lambda row: row["seed"])
        data = {
            key: _mean_sd([float(row[key]) for row in rows])
            for key in (
                "exact_match",
                "valid_loss",
                "valid_mean_token_accuracy",
                "wall_seconds",
            )
        }
        if recipe != "healthy":
            differences = np.asarray(
                [
                    row["exact_match"] - healthy[row["seed"]]["exact_match"]
                    for row in rows
                ],
                dtype=float,
            )
            samples = differences[
                rng.integers(0, len(differences), (bootstrap_samples, len(differences)))
            ].mean(axis=1)
            data["paired_exact_match_vs_healthy"] = {
                "mean_difference": float(differences.mean()),
                "bootstrap_95_interval": [
                    float(value) for value in np.quantile(samples, [0.025, 0.975])
                ],
                "wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "losses": int((differences < 0).sum()),
            }
        recipes[recipe] = data
    return {
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "paired training seed",
        },
        "recipes": recipes,
    }


def relabel(rows: list[dict], *, recipes: bool = False, observed: bool = False):
    result = []
    for row in rows:
        item = dict(row)
        if recipes:
            item["label"] = recipe_name(row)
        elif observed:
            item["label"] = "lr_too_low" if row["label"] == "lr_too_low" else "healthy"
        result.append(item)
    return result


def analyze(runs: Path, seeds: tuple[int, ...]) -> tuple[dict, dict, dict]:
    rows = load_runs(runs)
    validate_matrix(rows, seeds)
    recipe_model = fit(rows)
    observed_rows = relabel(rows, observed=True)
    observed_model = fit(observed_rows)
    cases = load_cases("dev") + load_cases("heldout")
    transfer = evaluate_transfer(cases, observed_model)
    report = {
        "version": 1,
        "scope": (
            "Qwen3-0.6B, 50 BANKING77 training rows, 16 validation rows; "
            "recipe recognition is not causal failure diagnosis"
        ),
        "matrix": {
            "runs": len(rows),
            "seeds": list(seeds),
            "recipes": list(RECIPES),
        },
        "outcomes": aggregate_outcomes(runs, expected_per_recipe=len(seeds)),
        "classifiers": {
            "fault_family_leave_one_seed_out": evaluate(rows),
            "seven_recipe_leave_one_seed_out": evaluate(relabel(rows, recipes=True)),
            "fault_family_cross_strength": strength_transfer(rows, len(seeds) // 2),
            "observed_lr_failure_leave_one_seed_out": evaluate(observed_rows),
        },
        "input_hashes": {
            row["id"]: {
                "summary_sha256": row["summary_sha256"],
                "metrics_sha256": row["metrics_sha256"],
            }
            for row in rows
        },
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "bench/loss_classifier.py",
                "bench/loss_transfer.py",
                "bench/sweep_analysis.py",
            )
        },
    }
    model_bytes = (json.dumps(observed_model, indent=2) + "\n").encode()
    transfer.update(
        {
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "case_sha256": {
                case.id: hashlib.sha256(
                    json.dumps(asdict(case), sort_keys=True).encode()
                ).hexdigest()
                for case in cases
            },
            "source_sha256": report["source_sha256"],
            "scope": (
                "Frozen binary local classifier transferred to 13 previously "
                "inspected Token Factory cases; no refitting"
            ),
        }
    )
    return report, recipe_model, {"model": observed_model, "transfer": transfer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists")
    report, recipe_model, observed = analyze(args.runs, tuple(args.seeds))
    args.out.mkdir(parents=True)
    outputs = {
        "report.json": report,
        "recipe-model.json": recipe_model,
        "observed-failure-model.json": observed["model"],
        "token-factory-transfer.json": observed["transfer"],
    }
    for name, data in outputs.items():
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": report["matrix"]["runs"],
                "fault_family_balanced_accuracy": report["classifiers"][
                    "fault_family_leave_one_seed_out"
                ]["balanced_accuracy"],
                "observed_failure_balanced_accuracy": report["classifiers"][
                    "observed_lr_failure_leave_one_seed_out"
                ]["balanced_accuracy"],
                "token_factory": observed["transfer"]["splits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
