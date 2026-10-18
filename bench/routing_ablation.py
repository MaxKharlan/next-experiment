"""Compare mixed routing with Super-only using cached training and live narratives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from bench.ablations import fingerprint, frozen_path, replay
from bench.cases import load_cases
from core.agent.router import Router
from core.connectors.token_factory import TokenFactory
from core.probes.weight_cache import weights_enabled
from core.telemetry.calls import CallLog, LLMCall


def run(args, cases, *, tf=None):
    if not args.allow_api_calls:
        raise ValueError(
            "Pass --allow-api-calls to authorize paid narrative generation"
        )
    if not (math.isfinite(args.noise_sd) and args.noise_sd > 0):
        raise ValueError("noise_sd must be finite and positive")
    if not (math.isfinite(args.min_effect) and 0 < args.min_effect <= 1):
        raise ValueError("min_effect must be finite and in (0, 1]")
    if not cases:
        raise ValueError("No cases selected")
    if args.out.exists():
        raise FileExistsError(args.out)

    # Check every input and the offline trajectory before creating a provider.
    inputs = []
    for case in cases:
        path = frozen_path(args.priors, case, "with_tavily")
        content = path.read_bytes()
        frozen = json.loads(content)
        if frozen["fingerprint"] != fingerprint(case):
            raise ValueError(f"Stale hypotheses: {path}")
        replay(case, frozen, "eig", 0, args)
        inputs.append((case, frozen, hashlib.sha256(content).hexdigest()))

    provider = tf if tf is not None else TokenFactory.from_settings()
    # The shared hypothesis call must itself be a Super call.
    for case, frozen, _ in inputs:
        calls = frozen["llm_calls"]
        if not calls or any(
            call["purpose"] != "hypotheses" or call["model"] != provider.model("super")
            for call in calls
        ):
            raise ValueError(f"Expected recorded Super hypothesis calls for {case.id}")

    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "version": 1,
        "split": args.split,
        "variants": ["mixed", "super_only"],
        "models": {role: provider.model(role) for role in ("nano", "super", "ultra")},
        "source_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "core/agent/router.py",
                "core/agent/loop.py",
                "bench/ablations.py",
                "bench/routing_ablation.py",
                "core/agent/steps.py",
                "core/agent/prompts/verdict.v1.md",
                "core/agent/prompts/interpret_probe.v1.md",
            )
        },
        "pair_cache_hashes": {
            case.id: hashlib.sha256(path.read_bytes()).hexdigest()
            if path.exists()
            else None
            for case, _, _ in inputs
            for path in [
                args.artifacts / f"bench/{case.split}/{case.id}/{args.pairs_name}.json"
            ]
        },
        "frozen_inputs": {case.id: digest for case, _, digest in inputs},
        "weight_probe_enabled": weights_enabled(),
        "noise_sd": args.noise_sd,
        "min_effect": args.min_effect,
        "pairs_name": args.pairs_name,
        "budget_usd": 1.0,
        "max_interventions": 2,
        "seeds_per_arm": 2,
        "new_training_jobs": 0,
        "hypotheses": "shared frozen Super calls with Tavily evidence",
        "narratives": "live probe explanations and verdicts in both variants",
        "cost_scope": "shared hypothesis cost plus live calls plus estimated cached training",
        "limitations": [
            "Narrative quality is not scored automatically.",
            "Model text does not determine causal status or patch.",
            "Different LLM costs can change budget-limited trajectories.",
            "One generation per variant; generation variance is not estimated.",
            "Held-out reruns are exploratory.",
        ],
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    try:
        for case, frozen, _ in inputs:
            for variant in ("mixed", "super_only"):
                log = CallLog(
                    f"routing-{case.split}-{case.id}-{variant}",
                    [LLMCall(**call) for call in frozen["llm_calls"]],
                )
                original_count = len(log.calls)
                router = Router(provider, log, super_only=variant == "super_only")
                prefix = args.out / f"{case.id}--{variant}"
                try:
                    diagnosis = replay(case, frozen, "eig", 0, args, router=router)
                finally:
                    # Preserve billed attempts even when a provider request fails.
                    prefix.with_suffix(".calls.json").write_text(
                        json.dumps(
                            {
                                "shared_hypothesis_calls": [
                                    asdict(c) for c in log.calls[:original_count]
                                ],
                                "new_calls": [
                                    asdict(c) for c in log.calls[original_count:]
                                ],
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                diagnosis.method = variant
                prefix.with_suffix(".json").write_text(
                    json.dumps(asdict(diagnosis), indent=2) + "\n"
                )
                rows.append(
                    {
                        "case": case.id,
                        "variant": variant,
                        "status": diagnosis.status,
                        "cause": diagnosis.cause,
                        "ranking": diagnosis.ranking,
                        "llm_cost_usd": diagnosis.llm_cost_usd,
                        "llm_tokens": diagnosis.llm_tokens,
                        "new_llm_cost_usd": sum(
                            c.cost_usd for c in log.calls[original_count:]
                        ),
                        "new_llm_calls": len(log.calls) - original_count,
                        "estimated_training_cost_usd": diagnosis.training_cost_usd,
                        "cached_training_jobs": diagnosis.mini_jobs,
                    }
                )
    except Exception as err:
        (args.out / "INCOMPLETE.txt").write_text(f"{type(err).__name__}: {err}\n")
        raise
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Saved routing comparison to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--priors", type=Path, default=Path("artifacts/ablations/priors-v1")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs-name", default="pairs-check")
    parser.add_argument("--noise-sd", type=float, default=0.019)
    parser.add_argument("--min-effect", type=float, default=0.06)
    args = parser.parse_args()
    if not args.allow_api_calls:
        parser.error("--allow-api-calls is required; this experiment calls Nemotron")
    run(args, load_cases(args.split))


if __name__ == "__main__":
    main()
