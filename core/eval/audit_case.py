"""Free dataset investigation using the shared case contract and actual probes."""

import hashlib
import json

from apps.api.live_protocol import inspect_datasets
from core.agent.context import InvestigationContext
from core.eval.data import read_examples
from core.eval.structured import validate_answer, validate_evaluator
from core.probes.level0.answers import AnswerCut


def audit_case(case):
    spec = validate_evaluator(case.task, case.evaluator)
    paths = {
        role: getattr(case.datasets, role).ref
        for role in ("train", "validation", "check", "test")
    }
    manifests = inspect_datasets(paths)
    summaries = {}
    owners = {}
    for role, entry in manifests.items():
        declared = getattr(case.datasets, role)
        if declared.sha256 is not None and declared.sha256 != entry["sha256"]:
            raise ValueError(f"{role}: declared hash differs")
        if declared.examples is not None and declared.examples != entry["examples"]:
            raise ValueError(f"{role}: declared count differs")
        rows = read_examples(entry["path"])
        invalid = []
        for row in rows:
            key = hashlib.sha256(
                json.dumps(row.prompt, sort_keys=True).encode()
            ).hexdigest()
            if key in owners:
                raise ValueError(f"Duplicate prompt in {owners[key]} and {role}")
            owners[key] = role
            if case.task != "banking77":
                try:
                    validate_answer(case.task, row.expected)
                except ValueError:
                    invalid.append(row.line)
        summaries[role] = {"examples": len(rows), "invalid_reference_lines": invalid}
        if role in {"check", "test"} and invalid:
            raise ValueError(f"{role}: invalid scoring references")
    context = InvestigationContext(
        "dataset-audit", case.mode, case.job.id, case.job.model_dump(), files=paths
    )
    probe = AnswerCut().run(context)
    return {
        "investigation_id": "dataset-audit",
        "status": "inconclusive",
        "hypotheses": [],
        "llm_cost_usd": 0.0,
        "llm_calls": {},
        "evidence_scope": "dataset inspection; no model training or causal experiment",
        "steps": [
            {
                "kind": "dataset_audit",
                "raw": summaries,
                "metric": spec.metric,
                "datasets": manifests,
                "cost_usd": 0.0,
            },
            {
                "kind": "probe",
                "probe": "answer_cut",
                "outcome": probe.outcome,
                "raw": json.loads(probe.summary),
                "cost_usd": 0.0,
                "gain_bits": 0.0,
                "probs": {},
                "card": "Observed label integrity; this does not prove a model failure.",
            },
        ],
        "protocol": {
            "task": case.task,
            "evaluator": spec.model_dump(),
            "datasets": manifests,
        },
    }
