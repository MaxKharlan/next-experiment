"""Connect answer-cut evidence to the investigation loop and TF execution."""

from __future__ import annotations

import difflib
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.agent.loop import PaidStepApprovalRequired, investigate
from core.interventions.data_pairs import answer_cut_pair, flagged_lines
from core.interventions.pairs import BUILDERS, Intervention, make_pair
from core.interventions.tf_runner import TokenFactoryPairRunner
from core.probes.level0.answers import AnswerCut


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class AnswerDrop(Intervention):
    """A recipe-compatible intervention bound to one training file."""

    drop_lines: list[int] = field(default_factory=list)
    source_sha256: str = ""

    def rows(self, path):
        path = Path(path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.source_sha256:
            raise ValueError("Training file changed after the answer-cut probe")
        rows = raw.decode("utf-8").splitlines()
        if not rows or any(not row.strip() for row in rows):
            raise ValueError("Answer interventions require JSONL without blank lines")
        drop = set(self.drop_lines)
        if (
            not drop
            or len(drop) != len(self.drop_lines)
            or any(type(line) is not int for line in drop)
            or min(drop) < 1
            or max(drop) > len(rows)
            or len(drop) >= len(rows)
        ):
            raise ValueError("Invalid answer-cut line selection")
        return [row for line, row in enumerate(rows, 1) if line not in drop]

    def diff(self, base):
        patch = {
            "version": 1,
            "operation": "drop_training_lines",
            "source_sha256": self.source_sha256,
            "line_numbers": self.drop_lines,
            "numbering": "one-based",
        }
        return "\n".join(
            difflib.unified_diff(
                ["{}"],
                json.dumps(patch, indent=2).splitlines(),
                "data_patch.json",
                "data_patch.json",
                lineterm="",
            )
        )


def bind_pair(hypothesis_id, raw, source_sha256):
    """Use observed probe lines, never an injection manifest or original labels."""
    info = raw["files"]["train"]
    pair = answer_cut_pair(
        hypothesis_id,
        flagged_lines(raw),
        examples=info["examples"],
        seed=0,
    )
    return tuple(
        AnswerDrop(
            hypothesis_id=arm.hypothesis_id,
            kind=arm.kind,
            patch={},
            description=arm.description,
            drop_lines=arm.drop_lines,
            source_sha256=source_sha256,
        )
        for arm in pair
    )


def investigate_with_answers(ctx, case, router, run_pair, **kwargs):
    """Run the free data check before hypotheses, then use the existing loop."""
    resume_state = kwargs.get("resume_state")
    if resume_state is not None:
        saved = next(
            (
                step
                for step in resume_state["steps"]
                if step.get("probe") == "answer_cut"
            ),
            None,
        )
        if saved is None:
            return investigate(ctx, case, router, run_pair, **kwargs)
        raw = saved["raw"]
        train = Path(ctx.files["train"])
        source_sha256 = raw["source_sha256"]["train"]

        def build(hypothesis, context):
            if hypothesis.fault_class != "answer_truncation":
                return make_pair(hypothesis, context)
            pair = bind_pair(hypothesis.id, raw, source_sha256)
            for arm in pair:
                arm.rows(train)
            return pair

        classes = set(BUILDERS) | {"answer_truncation"}
        recipe_proposal = kwargs.pop("propose_retrain", None)

        def proposal(target, control):
            if isinstance(target, AnswerDrop):
                return None
            return recipe_proposal(target, control) if recipe_proposal else None

        return investigate(
            ctx,
            case,
            router,
            run_pair,
            pair_factory=build,
            testable_classes=classes,
            propose_retrain=proposal,
            **kwargs,
        )
    probe = AnswerCut()
    if not getattr(run_pair, "supports_answer_drop", False) or not probe.available(ctx):
        return investigate(ctx, case, router, run_pair, **kwargs)

    paths = {
        role: Path(ctx.files[role])
        for role in ("train", "validation")
        if ctx.files.get(role)
    }
    before = {role: digest(path) for role, path in paths.items()}
    train_rows = paths["train"].read_text(encoding="utf-8").splitlines()
    if any(not row.strip() for row in train_rows):
        raise ValueError("Answer interventions require JSONL without blank lines")

    run = probe.run(ctx)
    if before != {role: digest(path) for role, path in paths.items()}:
        raise ValueError("Dataset changed while running the answer-cut probe")

    raw = json.loads(run.summary)
    raw["source_sha256"] = before
    info = raw["files"]["train"]
    count = len(flagged_lines(raw))
    can_test = run.outcome == "cut" and 0 < count <= info["examples"] // 2

    step = {
        "kind": "probe",
        "probe": "answer_cut",
        "outcome": run.outcome,
        "raw": raw,
        "phase": "before_hypotheses",
        "cost_usd": 0.0,
        "causal_check_available": can_test,
    }
    if not can_test:
        step["reason"] = (
            "No actionable cut signal, or too few unflagged rows for a control"
        )

    callback = kwargs.get("on_step")
    if callback:
        callback(step)

    def build(hypothesis, context):
        if hypothesis.fault_class != "answer_truncation":
            return make_pair(hypothesis, context)
        if not can_test:
            raise ValueError("No answer-cut intervention is available")
        pair = bind_pair(hypothesis.id, raw, before["train"])
        for arm in pair:
            arm.rows(paths["train"])
        return pair

    recipe_proposal = kwargs.pop("propose_retrain", None)

    def proposal(target, control):
        if isinstance(target, AnswerDrop):
            return None
        return recipe_proposal(target, control) if recipe_proposal else None

    # Evidence is already supplied to the hypothesis model. Do not apply the
    # same answer-cut signal again as an independent Bayesian observation.
    if "probes" in kwargs:
        kwargs["probes"] = tuple(
            probe for probe in kwargs["probes"] if probe.name != "answer_cut"
        )

    classes = set(BUILDERS)
    if can_test:
        classes.add("answer_truncation")
    try:
        result = investigate(
            ctx,
            {**case, "answer_cut_evidence": raw},
            router,
            run_pair,
            pair_factory=build,
            testable_classes=classes,
            initial_evidence=("probe:answer_cut",),
            propose_retrain=proposal,
            **kwargs,
        )
    except PaidStepApprovalRequired as pause:
        pause.proposal["continuation"]["steps"].insert(0, step)
        raise
    result.steps.insert(0, step)
    return result


@dataclass
class DataAwareTFPairRunner(TokenFactoryPairRunner):
    """Keep recipe pairs unchanged; materialize bound data interventions."""

    protocol: dict | None = None

    def __call__(self, target, control):
        data_pair = isinstance(target, AnswerDrop) or isinstance(control, AnswerDrop)
        if data_pair:
            if not all(isinstance(arm, AnswerDrop) for arm in (target, control)):
                raise TypeError("Both arms must be answer interventions")
            if self.protocol is None:
                raise ValueError("Data interventions require a frozen protocol")
            if (
                target.kind != "target"
                or control.kind != "control"
                or target.hypothesis_id != control.hypothesis_id
                or target.patch
                or control.patch
                or len(target.drop_lines) != len(control.drop_lines)
                or set(target.drop_lines) & set(control.drop_lines)
            ):
                raise ValueError("Invalid answer intervention pair")

            for entry in self.protocol["datasets"].values():
                if digest(entry["path"]) != entry["sha256"]:
                    raise ValueError("Dataset changed after protocol creation")
            for arm in (target, control):
                arm.rows(self.train_path)
                if arm.metrics or arm.ft_job_ids:
                    raise ValueError("Use fresh intervention objects for a rerun")

            binding = {
                "version": 1,
                "protocol": self.protocol,
                "model": self.model,
                "base_hp": self.base_hp,
                "seeds": self.seeds,
                "train_sha256": digest(self.train_path),
                "validation_sha256": digest(self.valid_path),
                "target": asdict(target),
                "control": asdict(control),
            }
            key = f"{self.prefix}/data-pair.json"
            if self.storage.exists(key):
                if json.loads(self.storage.get_text(key)) != binding:
                    raise ValueError("Saved data pair belongs to another protocol")
            elif self.storage.exists(f"{self.prefix}/jobs.json"):
                raise ValueError("Existing jobs have no data-pair provenance")
            else:
                self.storage.put_text(key, json.dumps(binding, indent=2))
        super().__call__(target, control)

    def _train_file(self, arm, seed):
        if not isinstance(arm, AnswerDrop):
            return super()._train_file(arm, seed)
        rows = arm.rows(self.train_path)
        random.Random(seed).shuffle(rows)
        key = f"{self.prefix}/{arm.kind}-s{seed}-train.jsonl"
        self.storage.put_text(key, "\n".join(rows) + "\n")
        return self.storage.root / key
