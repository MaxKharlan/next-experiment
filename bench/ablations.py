"""Prepare frozen hypotheses, then replay ablations without network calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bench.cases import load_cases
from bench.diagnosis import from_investigation
from bench.fixes import FixCache, fixed_hyperparameters, reference_metric
from bench.methods import pair_cost
from bench.pair_cache import PairCache, PairNotCached
from bench.score import score, summarize, table
from core.agent.context import InvestigationContext
from core.agent.loop import investigate
from core.agent.schemas import Hypothesis, Verdict
from core.connectors.storage import LocalStorage
from core.planner.causal_belief import MODEL
from core.probes.weight_cache import investigation_probes, weights_enabled
from core.telemetry.calls import CallLog, LLMCall


def fingerprint(case):
    """Bind frozen hypotheses to the case and hypothesis prompt sources."""
    paths = [
        Path("core/agent/steps.py"),
        Path("core/agent/catalog.py"),
        Path("core/agent/known_issues.py"),
        Path("core/agent/prompts/hypotheses.v3.md"),
    ]
    payload = {
        "case": asdict(case),
        "sources": {str(p): p.read_text() for p in paths},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def frozen_path(root, case, condition):
    return root / case.split / case.id / f"{condition}.json"


def prepare(args, cases):
    """This is the only phase that can call Nemotron or Tavily."""
    from core.agent.known_issues import find_known_issues
    from core.agent.router import Router
    from core.agent.steps import propose_hypotheses
    from core.config import get_settings
    from core.connectors.tavily import Tavily
    from core.connectors.token_factory import TokenFactory

    if not get_settings().tavily_api_key:
        raise SystemExit("TAVILY_API_KEY is required for the with-Tavily condition")

    storage = LocalStorage(args.artifacts)
    tf = TokenFactory.from_settings()
    tavily = Tavily.from_settings(storage)

    for case in cases:
        for condition in ("with_tavily", "without_tavily"):
            path = frozen_path(args.priors, case, condition)
            digest = fingerprint(case)
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["fingerprint"] != digest:
                    raise SystemExit(f"Stale hypotheses: {path}")
                print(f"Reusing {path}")
                continue

            inv_id = f"ablation-{case.split}-{case.id}-{condition}"
            payload = dict(case.case)
            payload.pop("known_issues", None)
            found = None

            if condition == "with_tavily":
                job = payload.get(
                    "job",
                    {
                        "model": case.model,
                        "hyperparameters": case.hyperparameters,
                    },
                )
                found = find_known_issues(
                    tavily, inv_id, job, payload.get("symptom", "")
                )
                if not found["sources"]:
                    raise RuntimeError(
                        f"No Tavily sources for {case.id}; comparison is undefined"
                    )
                payload["known_issues"] = [
                    {k: source[k] for k in ("title", "url", "snippet")}
                    for source in found["sources"]
                ]

            router = Router(tf, CallLog(inv_id))
            hypotheses = propose_hypotheses(router, payload, mode="A")
            saved = {
                "fingerprint": digest,
                "condition": condition,
                "hypotheses": [h.model_dump() for h in hypotheses],
                "llm_calls": [asdict(c) for c in router.log.calls],
                "known_issues": found,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2) + "\n")
            print(f"Saved {path}")


class ReplayRouter:
    """Preserve hypothesis costs and forbid new LLM requests."""

    def __init__(self, inv_id, calls):
        self.log = CallLog(inv_id, [LLMCall(**call) for call in calls])

    def structured(self, step, messages, schema, *, prompt_version):
        if step != "verdict" or schema is not Verdict:
            raise AssertionError(f"Unexpected LLM request during replay: {step}")
        return Verdict(
            hypothesis_id="",
            status="inconclusive",
            evidence=[],
            patch="",
            summary="Narrative generation is disabled for all ablation variants.",
        )


def replay(case, frozen, policy, seed, args):
    ctx = InvestigationContext(
        investigation_id=f"ablation-{case.id}-{policy}-{seed}",
        mode="A",
        job_id=case.job_id,
        job_config={"model": case.model, **case.job_config()},
        budget_left_usd=1.0,
    )
    storage = LocalStorage(args.artifacts)
    pairs = PairCache(
        storage,
        f"bench/{case.split}/{case.id}/{args.pairs_name}.json",
        run_pair=None,
    )

    def cached_pair(target, control):
        pairs(target, control)
        if any(len(arm.metrics) != 2 for arm in (target, control)):
            raise ValueError("This protocol requires exactly two seeds per arm")

    hypotheses = [Hypothesis.model_validate(h) for h in frozen["hypotheses"]]
    result = investigate(
        ctx,
        {},
        ReplayRouter(ctx.investigation_id, frozen["llm_calls"]),
        cached_pair,
        initial_hypotheses=hypotheses,
        probes=investigation_probes(ctx),
        explain=False,
        policy="eig" if policy == "no_causal" else policy,
        causal_checks=policy != "no_causal",
        random_seed=seed,
        max_interventions=2,
        intervention_cost_usd=pair_cost(case, seeds=2),
        min_effect=args.min_effect,
        noise_sd=args.noise_sd,
    )
    return from_investigation(result)


VARIANT_NAMES = ("full", "no_tavily", "no_planner", "no_causal", "random")


def selected_variants(random_seeds, selected=None):
    names = set(VARIANT_NAMES if selected is None else selected)
    unknown = names - set(VARIANT_NAMES)
    if unknown:
        raise ValueError(f"Unknown ablation variants: {sorted(unknown)}")
    if not names:
        raise ValueError("Select at least one ablation variant")
    variants = [
        ("full", "with_tavily", "eig", 0),
        ("no_tavily", "without_tavily", "eig", 0),
        ("no_planner", "with_tavily", "fixed", 0),
        ("no_causal", "with_tavily", "no_causal", 0),
    ]
    return [row for row in variants if row[0] in names] + (
        [(f"random_{seed}", "with_tavily", "random", seed) for seed in random_seeds]
        if "random" in names
        else []
    )


def run(args, cases):
    if args.out is None:
        raise SystemExit("--out is required for replay")
    args.out.mkdir(parents=True, exist_ok=False)

    variants = selected_variants(args.random_seeds, getattr(args, "variants", None))
    metadata = {
        "protocol_version": 2,
        "no_causal_rule": "EIG until the first intervention; retain ranking without confirmation",
        "belief_model": MODEL,
        "weight_probe_enabled": weights_enabled(),
        "split": args.split,
        "cases": [case.id for case in cases],
        "variants": variants,
        "noise_sd": args.noise_sd,
        "min_effect": args.min_effect,
        "pairs_name": args.pairs_name,
        "budget_usd": 1.0,
        "max_interventions": 2,
        "seeds_per_arm": 2,
        "narratives": "disabled for every variant",
        "cost_scope": "hypothesis LLM calls plus estimated pair training",
        "tavily_cost": "credits recorded in priors; excluded from USD totals",
        "new_training_jobs": 0,
        "new_api_calls_during_replay": 0,
    }
    (args.out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rows = []
    storage = LocalStorage(args.artifacts)
    try:
        for case in cases:
            inputs = {}
            for condition in sorted({row[1] for row in variants}):
                path = frozen_path(args.priors, case, condition)
                frozen = json.loads(path.read_text())
                if frozen["fingerprint"] != fingerprint(case):
                    raise ValueError(f"Stale hypotheses: {path}")
                inputs[condition] = frozen

            fixes = FixCache(storage, f"bench/{case.split}/{case.id}/fixes.json")
            for name, condition, policy, seed in variants:
                d = replay(case, inputs[condition], policy, seed, args)
                d.method = name
                hp = (
                    fixed_hyperparameters(case, d.cause)
                    if d.status == "confirmed" and not case.healthy
                    else None
                )
                row = score(
                    case,
                    d,
                    has_fix=hp is not None,
                    fixed_metric=fixes(hp) if hp is not None else None,
                    reference=reference_metric(case.task, case.model),
                )
                rows.append(row)
                report = args.out / f"{case.id}--{name}.json"
                report.write_text(json.dumps(asdict(d), indent=2) + "\n")
                print(f"{case.id} {name}: {d.status}, cause={d.cause}")
    except (PairNotCached, FileNotFoundError, KeyError, ValueError) as err:
        (args.out / "INCOMPLETE.txt").write_text(str(err) + "\n")
        raise SystemExit(
            f"Incomplete experiment: {err}\n"
            "No comparison table was produced. No training was started."
        ) from err

    (args.out / "rows.jsonl").write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in rows)
    )
    summary = table(summarize(rows))
    (args.out / "summary.md").write_text(summary + "\n")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_NAMES,
        help="Replay selected variants; random expands to --random-seeds",
    )
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--priors", type=Path, default=Path("artifacts/ablations/priors-v1")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pairs-name", default="pairs-check")
    parser.add_argument("--noise-sd", type=float, default=0.019)
    parser.add_argument("--min-effect", type=float, default=0.06)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    if args.prepare and args.variants is not None:
        parser.error(
            "--variants selects replay only; preparation creates both retrieval conditions"
        )
    if len(args.random_seeds) != len(set(args.random_seeds)):
        parser.error("Random seeds must be unique")
    cases = load_cases(args.split)
    if not cases:
        parser.error("No cases found")
    if args.prepare:
        prepare(args, cases)
    else:
        run(args, cases)


if __name__ == "__main__":
    main()
