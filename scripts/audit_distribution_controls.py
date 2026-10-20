"""Predeclared balanced and two-intent distribution controls on development vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.eval.runtime import file_digest
from core.probes.level0.distribution import label_divergence, two_sample_auc
from scripts.fresh_answer_calibration import save


def run(source, out):
    if out.exists():
        raise FileExistsError("Distribution experiment already exists")
    refs = json.loads((source / "reference.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    truth = json.loads((source / "constructed-truth.json").read_text())
    labels = json.loads((source / "protocol.json").read_text())["labels"]
    control = [i for i, row in enumerate(refs) if i % 50 < 25]
    shifted = [i for i, row in enumerate(refs) if row["label"] in labels[:2]]
    target = {label: 1 / len(labels) for label in labels}
    out.mkdir(parents=True)
    protocol = {
        "id": "distribution-controls-20261020-v1",
        "data_protocol_sha256": file_digest(source / "protocol.json"),
        "control_reference_indices": control,
        "shifted_reference_indices": shifted,
        "target_ids": [r["id"] for r in audit],
        "target_label_distribution": target,
        "auc_flag_above": 0.75,
        "js_flag_above": 0.1,
        "cv_seed": 5101,
        "source_sha256": {
            name: file_digest(name)
            for name in (
                "scripts/audit_distribution_controls.py",
                "core/probes/level0/distribution.py",
            )
        },
        "scope": "Two authored development distribution controls; no fitted signature or downstream fine-tuning effect.",
    }
    save(out / "protocol.json", protocol)
    results = {}
    for name in ("qwen-local", "nvidia-local", "qwen-provider"):
        path = source / "evidence" / f"{name}-embeddings.npy"
        vectors = np.load(path)
        results[name] = {}
        for case, indices in (("balanced", control), ("two_intents", shifted)):
            auc = two_sample_auc(
                vectors[indices],
                vectors[400:],
                left_ids=[refs[i]["id"] for i in indices],
                right_ids=[r["id"] for r in audit],
            )
            div = label_divergence([refs[i]["label"] for i in indices], target)
            results[name][case] = {
                "embedding_sha256": file_digest(path),
                "auc": auc,
                "labels": div,
                "flag": auc["auc"] > 0.75 or div["js_divergence_bits"] > 0.1,
            }
    result = {
        "protocol_sha256": file_digest(out / "protocol.json"),
        "results": results,
        "target_observed_labels": label_divergence(
            [r["source_label"] for r in truth], target
        ),
    }
    save(out / "result.json", result)
    print(
        json.dumps(
            {
                name: {
                    case: {
                        "auc": r["auc"]["auc"],
                        "js": r["labels"]["js_divergence_bits"],
                        "flag": r["flag"],
                    }
                    for case, r in cases.items()
                }
                for name, cases in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.out)
