"""Mode B repairs for the four supported local trainer fault switches."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.interventions.pairs import Intervention, job_hyperparameters

CODE_FAULTS = {
    "loss_on_prompt": "prompt_loss",
    "pad_eos_masking": "eos_as_pad",
    "double_label_shift": "double_shift",
    "grad_accum_normalization": "microbatch_mean",
}


def make_code_pair(hypothesis, ctx):
    """Repair only the named switch; replay the failed switch in the control.

    This factory is opt-in for the local runner. Code switches must never be
    submitted as Token Factory hyperparameters.
    """
    if ctx.mode != "B":
        raise ValueError("Code interventions require mode B")
    expected = CODE_FAULTS.get(hypothesis.fault_class)
    if expected is None:
        raise ValueError("Unsupported code fault")
    if "code_fault" not in ctx.job_config:
        raise ValueError("Declare code_fault explicitly; use null for healthy code")
    observed = ctx.job_config["code_fault"]
    if observed not in {None, *CODE_FAULTS.values()}:
        raise ValueError("Unsupported trainer code_fault")
    if observed is not None and observed != expected:
        raise ValueError("The hypothesis does not match the configured fault switch")
    hp = job_hyperparameters(ctx)
    if expected == "microbatch_mean" and hp.get("gradient_accumulation_steps", 1) < 2:
        raise ValueError("Accumulation intervention requires at least two microbatches")
    return (
        Intervention(
            hypothesis.id,
            "target",
            {"code_fault": None},
            f"Use the healthy {hypothesis.fault_class} implementation",
        ),
        Intervention(
            hypothesis.id,
            "control",
            {"code_fault": observed},
            "Repeat the observed implementation with matched seeds and steps",
        ),
    )


@dataclass
class LocalCodePairRunner:
    """Run isolated arms through a local training-and-evaluation callback.

    execute(hyperparameters, code_fault, seed, output_dir) returns a finite
    higher-is-better held-out metric. The callback owns dataset loading and must
    evaluate on data excluded from optimizer updates. No cached result is reused.
    """

    base_hp: dict
    seeds: list[int]
    out: Path
    execute: Callable

    def __call__(self, target, control):
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Require at least two distinct seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if target.kind != "target" or control.kind != "control":
            raise ValueError("Expected target and control arms")
        if target.hypothesis_id != control.hypothesis_id:
            raise ValueError("Arms must test the same hypothesis")
        for arm in (target, control):
            if set(arm.patch) != {"code_fault"}:
                raise ValueError("Code repairs must change only code_fault")
            if arm.patch["code_fault"] not in {None, *CODE_FAULTS.values()}:
                raise ValueError("Unsupported trainer code_fault")
            if arm.metrics or arm.ft_job_ids:
                raise ValueError("Arms must not contain results from previous runs")
        # Fail before starting either arm; partial results are never silently reused.
        self.out.mkdir(parents=True, exist_ok=False)
        pending = []
        for arm in (target, control):
            scores, ids = [], []
            for seed in self.seeds:
                directory = self.out / f"{arm.kind}-s{seed}"
                directory.mkdir()
                metric = self.execute(
                    dict(self.base_hp), arm.patch["code_fault"], seed, directory
                )
                if isinstance(metric, bool) or not math.isfinite(metric):
                    raise ValueError("Evaluation must return a finite numeric metric")
                scores.append(float(metric))
                ids.append(f"local:{directory}")
            pending.append((arm, scores, ids))
        # Do not publish a half-completed causal comparison.
        for arm, scores, ids in pending:
            arm.metrics.extend(scores)
            arm.ft_job_ids.extend(ids)


def investigate_code(ctx, case, router, runner, **kwargs):
    """Opt in to local code interventions through the existing agent loop."""
    from core.agent.loop import investigate

    if ctx.mode != "B":
        raise ValueError("Code interventions require mode B")
    if "code_fault" not in ctx.job_config:
        raise ValueError("Declare code_fault explicitly; use null for healthy code")
    if ctx.job_config["code_fault"] not in {None, *CODE_FAULTS.values()}:
        raise ValueError("Unsupported trainer code_fault")
    kwargs.setdefault("max_interventions", 1)
    return investigate(
        ctx,
        case,
        router,
        runner,
        pair_factory=make_code_pair,
        testable_classes={
            name
            for name, switch in CODE_FAULTS.items()
            if ctx.job_config.get("code_fault") in {None, switch}
        },
        **kwargs,
    )


@dataclass
class LocalLoRAExecutor:
    """Adapter for the existing local trainer and a separate evaluation JSONL.

    Callers supply existing files and model weights. This executor may load model
    weights through Transformers; the offline synthetic experiment does not use it.
    """

    train_path: Path
    valid_path: Path
    check_path: Path
    model_id: str
    device: str = "cpu"

    def __post_init__(self):
        import hashlib
        import json

        self._digests = {}
        datasets = []
        for path in (self.train_path, self.valid_path, self.check_path):
            data = path.read_bytes()
            self._digests[path] = hashlib.sha256(data).hexdigest()
            rows = {
                json.dumps(json.loads(line), sort_keys=True)
                for line in data.splitlines()
                if line.strip()
            }
            if not rows:
                raise ValueError("Datasets must be nonempty")
            datasets.append(rows)
        if any(datasets[i] & datasets[j] for i in range(3) for j in range(i)):
            raise ValueError("Training, validation and check examples must be disjoint")

    def __call__(self, hp, code_fault, seed, directory):
        import hashlib
        import json

        from jobs.eval_local.exact_match import evaluate_adapter
        from jobs.kaggle.train_lora import train

        for path, digest in self._digests.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Dataset changed after pair preparation")
        train(
            self.train_path,
            self.valid_path,
            directory,
            model_id=self.model_id,
            hp=hp,
            seed=seed,
            device=self.device,
            code_fault=code_fault,
            generate=False,
        )
        metric = evaluate_adapter(
            directory / "adapter",
            self.check_path,
            base_model=self.model_id,
            device=self.device,
        )
        (directory / "check-score.json").write_text(
            json.dumps(
                {
                    "exact_match": metric,
                    "seed": seed,
                    "code_fault": code_fault,
                    "datasets": {
                        str(path): digest for path, digest in self._digests.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        return metric
