"""Apply the frozen observed-label rule; source truth is excluded from its input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from bench.faults.semantic_data import drop_candidates_with_control
from core.eval.runtime import file_digest
from scripts.prepare_data_fault_quality import save, write_rows


def prepare(root, embeddings):
    if (root / "label-repair.json").exists():
        raise FileExistsError("Completed observed repair is immutable")
    plan = json.loads((root / "preparation.json").read_text())
    for relative, expected in plan["files_sha256"].items():
        if file_digest(root / relative) != expected:
            raise ValueError(f"Frozen preparation changed: {relative}")
    previous = Path("docs/results/semantic-audit-20261020")
    refs = json.loads((previous / "reference.json").read_text())
    train = [
        json.loads(line)
        for line in (root / "training/labels-50.jsonl").read_text().splitlines()
    ]
    old_vectors = previous / "evidence/qwen-provider-embeddings.npy"
    new_vectors = embeddings / "embeddings.npy"
    old = np.load(old_vectors, allow_pickle=False)[: len(refs)]
    new = np.load(new_vectors, allow_pickle=False)
    execution = json.loads((embeddings / "execution.json").read_text())
    inputs = json.loads((root / "embedding-input/reference.json").read_text())
    if execution["ids"] != [row["id"] for row in inputs] or len(new) != len(train):
        raise ValueError("Embedding order differs from training")
    if any(
        a["text"] != b["messages"][-2]["content"]
        for a, b in zip(inputs, train, strict=True)
    ):
        raise ValueError("Training query order differs")
    x = np.concatenate([old, new])
    if not np.isfinite(x).all():
        raise ValueError("Invalid vectors")
    labels = sorted({row["label"] for row in refs})
    observed = [row["label"] for row in refs] + [
        row["messages"][-1]["content"] for row in train
    ]
    y = np.array([labels.index(label) for label in observed])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=6110)
    probabilities = cross_val_predict(
        LogisticRegression(C=1, max_iter=1000),
        x,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    flags = find_label_issues(y, probabilities, filter_by="prune_by_noise_rate")
    selected = np.flatnonzero(flags[len(refs) :]).tolist()
    result = {
        "preparation_sha256": file_digest(root / "preparation.json"),
        "source_sha256": file_digest(__file__),
        "reference_sha256": file_digest(previous / "reference.json"),
        "reference_vectors_sha256": file_digest(old_vectors),
        "train_vectors_sha256": file_digest(new_vectors),
        "embedding_execution_sha256": file_digest(embeddings / "execution.json"),
        "rule": plan["label_repair"],
        "candidate_indices": selected,
        "examples_before": len(train),
    }
    if 0 < len(selected) < len(train):
        target, control = drop_candidates_with_control(train, selected, seed=6111)
        write_rows(root / "training/labels-repair.jsonl", target)
        write_rows(root / "training/labels-control.jsonl", control)
        result.update(
            status="prepared",
            examples_after=len(target),
            target_sha256=file_digest(root / "training/labels-repair.jsonl"),
            control_sha256=file_digest(root / "training/labels-control.jsonl"),
        )
    else:
        result["status"] = "repair_unavailable"
    save(root / "label-repair.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.embeddings)
    print(
        json.dumps(
            {"status": result["status"], "flagged": len(result["candidate_indices"])}
        )
    )
