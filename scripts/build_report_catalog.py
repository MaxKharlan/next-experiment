"""Export measured historical traces and fresh local audits as replay reports.

Does not invent a diagnosis or run paid models. Existing demo files are preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from bench.faults.data import build as inject_answers
from bench.faults.labels import (
    include_prompt_labels,
    mask_padding_labels,
    shift_labels_once,
)
from bench.tasks.structured import load_case, sha, write_json
from core.agent.label_checks import check_training_labels
from core.eval.audit_case import audit_case


def save(out, name, title, note, case, report, provenance):
    case = copy.deepcopy(case)
    for spec in case.get("datasets", {}).values():
        path = Path(spec["ref"])
        if path.is_absolute() and path.is_relative_to(Path.cwd()):
            spec["ref"] = str(path.relative_to(Path.cwd()))
    write_json(
        out / f"{name}.json",
        {
            "title": title,
            "note": note,
            "case": case,
            "report": report,
            "provenance": provenance,
        },
    )


def accumulation_audit():
    import torch
    from torch import nn

    from jobs.kaggle.train_lora import backward_microbatches, collate

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([0.1, 0.2, -0.4, 0.3]))

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 4))

    rows = [
        {"input_ids": [1, 2], "labels": [-100, 2]},
        {"input_ids": [1, 3, 3, 3], "labels": [-100, 3, 3, 3]},
    ]
    gradients = {}
    for name, batches, fault in (
        ("full_batch", [collate(rows, 0, "cpu")], False),
        ("token_weighted", [collate([r], 0, "cpu") for r in rows], False),
        ("microbatch_mean", [collate([r], 0, "cpu") for r in rows], True),
    ):
        model = Model()
        backward_microbatches(model, batches, microbatch_mean=fault)
        gradients[name] = model.logits.grad.tolist()
    full = torch.tensor(gradients["full_batch"])
    good = float((full - torch.tensor(gradients["token_weighted"])).norm())
    bad = float((full - torch.tensor(gradients["microbatch_mean"])).norm())
    if good > 1e-6 or bad < 0.1:
        raise ValueError("Accumulation audit failed its equivalence check")
    return {
        "gradients": gradients,
        "healthy_difference_l2": good,
        "faulty_difference_l2": bad,
        "targets_per_microbatch": [1, 3],
        "scope": "CPU gradient mechanism; no downstream quality measurement",
        "causal_confirmation": False,
    }


def build(out, artifacts, tasks):
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=False)
    exported = []
    source_dir = Path("docs/results/b467-20261019/detection")
    for source in sorted(source_dir.glob("case-*.json")):
        data = json.loads(source.read_text())
        if data["error"] is not None:
            continue
        observation = copy.deepcopy(data["observation"])
        observation["job"]["model"] = observation["model"]
        report = data["trace"]
        fault = data["fault"]
        label = {
            "lr_too_low": "Fault 6 recipe",
            "too_many_epochs": "Fault 7 recipe",
            "lora_rank_too_small": "Fault 8 recipe",
            None: "Healthy recipe",
        }[fault]
        hp = data["hyperparameters"]
        title = f"{label}: lr {hp['learning_rate']:g}, rank {hp['lora_r']}, epochs {hp['n_epochs']}, seed {data['seed']} — inconclusive"
        note = (
            f"{data['study']}. Validation exact match {observation['validation_exact_match']:.3f}; "
            f"matched reference {observation['reference_validation_exact_match']:.3f}. "
            "Saved pre-intervention trace; no causal confirmation. A changed recipe is not proof of a harmful fault."
        )
        report = {
            **report,
            "evidence_scope": "pre-intervention ranking; no causal experiment",
        }
        name = "measured-" + data["case_id"]
        save(
            out,
            name,
            title,
            note,
            observation,
            report,
            {"source": str(source), "sha256": sha(source)},
        )
        exported.append(name)
    reference = [{"input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2]}]
    transforms = {
        9: include_prompt_labels,
        10: lambda r: mask_padding_labels(r, 2),
        11: shift_labels_once,
    }
    for number in (9, 10, 11, 12):
        if number in transforms:
            payload = {
                "reference": reference,
                "observed": transforms[number](reference),
                "eos_token_id": 2,
            }
            raw = check_training_labels(payload)
            write_json(artifacts / f"fault{number}-input.json", payload)
        else:
            raw = accumulation_audit()
        write_json(artifacts / f"fault{number}-audit.json", raw)
        kind = "training_label_checks" if number != 12 else "code_audit"
        report = {
            "investigation_id": f"local-audit-{number}",
            "status": "inconclusive",
            "hypotheses": [],
            "llm_cost_usd": 0.0,
            "llm_calls": {},
            "audit_only": True,
            "evidence_scope": "local mechanism inspection; no trained model diagnosis",
            "steps": [{"kind": kind, "raw": raw, "cost_usd": 0.0}],
        }
        label = {
            9: "Prompt loss",
            10: "EOS label masking",
            11: "Double label shift",
            12: "Gradient accumulation",
        }[number]
        name = f"audit-code-fault{number}"
        save(
            out,
            name,
            f"Fault {number}: {label} — local audit",
            "Reproducible CPU inspection. Findings do not establish a downstream quality failure.",
            {
                "mode": "B",
                "symptom": label,
                "job": {"model": "local mechanism fixture", "hyperparameters": {}},
            },
            report,
            {
                "source": str(artifacts / f"fault{number}-audit.json"),
                "sha256": sha(artifacts / f"fault{number}-audit.json"),
            },
        )
        exported.append(name)
    for task in ("json_extraction", "function_calling"):
        case = load_case(tasks / task / "audit-case.json")
        for damaged in (False, True):
            active = case.model_copy(deep=True)
            if damaged:
                target = artifacts / f"{task}-cut"
                manifest, _ = inject_answers(tasks / task, target, 0.3, seed=7)
                (target / "schema.json").write_bytes(
                    (tasks / task / "schema.json").read_bytes()
                )
                for role, spec in manifest["datasets"].items():
                    setattr(
                        active.datasets,
                        role,
                        type(active.datasets.train)(
                            **{**spec, "ref": str(target / spec["ref"])}
                        ),
                    )
            report = audit_case(active)
            report["audit_only"] = True
            name = f"{task}-" + ("fault1-audit" if damaged else "healthy-audit")
            # Keep downloadable reports portable; the hashes identify the bytes.
            for entry in report["protocol"]["datasets"].values():
                entry["path"] = str(Path(entry["path"]).relative_to(Path.cwd()))
            write_json(artifacts / f"{name}.json", report)
            save(
                out,
                name,
                f"{task}: "
                + (
                    "Fault 1 answer cuts — local audit"
                    if damaged
                    else "Healthy dataset — local audit"
                ),
                "Dataset inspection only. No model trained; no causal confirmation.",
                active.model_dump(),
                report,
                {
                    "source": str(artifacts / f"{name}.json"),
                    "sha256": sha(artifacts / f"{name}.json"),
                },
            )
            exported.append(name)
    write_json(
        artifacts / "catalog.json",
        {
            "reports": exported,
            "new_service_cost_usd": 0.0,
            "uncovered_faults": [2, 3, 4, 5],
            "reason": "No measured implementation report for faults 2–5 exists in audited main.",
        },
    )
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--tasks", type=Path, default=Path("datasets/structured-mixed-v1")
    )
    args = parser.parse_args()
    print("\n".join(build(args.out, args.artifacts, args.tasks)))


if __name__ == "__main__":
    main()
