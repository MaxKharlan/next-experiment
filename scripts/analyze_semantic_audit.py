"""Compare frozen embedding decisions and cleanlab against constructed references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from core.eval.runtime import file_digest
from core.probes.level0.semantic import confusion, neighbor_audit
from scripts.fresh_answer_calibration import save
from scripts.semantic_embeddings import read_data


def metrics(rows, truth):
    return {
        key: confusion([t["injection"] == fault for t in truth], [r[key] for r in rows])
        for key, fault in (
            ("format_error", "invalid_format"),
            ("contradiction", "wrong_label"),
        )
    }


def analyze(root, artifacts, out):
    if out.exists():
        raise FileExistsError("Audit results already exist")
    plan, rows = read_data(root)
    refs, audit = rows[:400], rows[400:]
    truth = json.loads((root / "constructed-truth.json").read_text())
    labels = plan["labels"]
    out.mkdir(parents=True)
    methods = {}
    sources = {}
    for key in ("qwen-local", "nvidia-local", "qwen-provider"):
        directory = artifacts / f"semantic-{key}"
        metadata = json.loads((directory / "execution.json").read_text())
        if (
            metadata["audit_protocol_sha256"] != file_digest(root / "protocol.json")
            or file_digest(directory / "embeddings.npy")
            != metadata["embeddings_sha256"]
        ):
            raise ValueError("Embedding provenance mismatch")
        if metadata["ids"] != [row["id"] for row in rows]:
            raise ValueError("Embedding row order mismatch")
        vectors = np.load(directory / "embeddings.npy")
        neighbors = neighbor_audit(refs, audit, vectors[:400], vectors[400:], labels)
        save(out / f"{key}-neighbors.json", neighbors)
        methods[key] = {"rows": neighbors, "metrics": metrics(neighbors, truth)}
        sources[key] = {
            "execution_sha256": file_digest(directory / "execution.json"),
            "model": metadata["model"],
            "dimensions": vectors.shape[1],
        }
        # Every training label is excluded from the fold that produces its own probability.
        valid = [i for i, r in enumerate(audit) if r["observed_label"] in labels]
        ids = list(range(400)) + [400 + i for i in valid]
        observed = [r["label"] for r in refs] + [
            audit[i]["observed_label"] for i in valid
        ]
        y = np.array([labels.index(label) for label in observed])
        classifier = LogisticRegression(
            C=plan["cleanlab"]["logistic_C"], max_iter=plan["cleanlab"]["max_iter"]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3103)
        probabilities = cross_val_predict(
            classifier, vectors[ids], y, cv=cv, method="predict_proba", n_jobs=1
        )
        issues = find_label_issues(
            labels=y,
            pred_probs=probabilities,
            filter_by="prune_by_noise_rate",
            n_jobs=1,
        )
        issue_map = {index: bool(issues[400 + j]) for j, index in enumerate(valid)}
        clf_rows = [
            {
                "id": row["id"],
                "format_error": row["observed_label"] not in labels,
                "contradiction": issue_map.get(i, False),
            }
            for i, row in enumerate(audit)
        ]
        save(
            out / f"{key}-cleanlab.json",
            {
                "rows": clf_rows,
                "out_of_fold_rows": len(ids),
                "probabilities": probabilities.tolist(),
                "row_ids": [rows[i]["id"] for i in ids],
            },
        )
        methods[f"{key}-cleanlab"] = {
            "rows": clf_rows,
            "metrics": metrics(clf_rows, truth),
        }
    agreement = {}
    for a, b in (
        ("qwen-local", "nvidia-local"),
        ("qwen-local", "qwen-provider"),
        ("nvidia-local", "qwen-provider"),
    ):
        left, right = methods[a]["rows"], methods[b]["rows"]
        agreement[f"{a}__{b}"] = {
            key: sum(x[key] == y[key] for x, y in zip(left, right, strict=True)) / 200
            for key in ("format_error", "contradiction")
        }
        agreement[f"{a}__{b}"]["disagreement_ids"] = [
            x["id"]
            for x, y in zip(left, right, strict=True)
            if x["contradiction"] != y["contradiction"]
        ]
    result = {
        "scope": "Automatic constructed audit reference; 200 selected BANKING77 training examples, no human adjudication or deployment calibration.",
        "protocol_sha256": file_digest(root / "protocol.json"),
        "amendment_sha256": file_digest(root / "acceptance-amendment.json"),
        "source_sha256": file_digest(__file__),
        "sources": sources,
        "metrics": {key: value["metrics"] for key, value in methods.items()},
        "outcome_agreement": agreement,
    }
    save(out / "report.json", result)
    print(json.dumps(result["metrics"], indent=2))


def add_reviews(root, artifacts, base, out):
    if out.exists():
        raise FileExistsError("Combined audit exists")
    truth = json.loads((root / "constructed-truth.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    report = json.loads((base / "report.json").read_text())
    all_rows = {}
    report["reviews"] = {}
    for role in ("nano", "super"):
        predictions = []
        failures = []
        cost = 0
        tokens_in = 0
        tokens_out = 0
        for example in audit:
            data = json.loads(
                (artifacts / f"semantic-{role}" / f"{example['id']}.json").read_text()
            )
            cost += data.get("estimated_cost_usd_excluding_tax", 0)
            usage = data.get("response", {}).get("usage") or {}
            tokens_in += usage.get("prompt_tokens", 0)
            tokens_out += usage.get("completion_tokens", 0)
            review = data.get("review")
            if review is None:
                failures.append(example["id"])
                # Missing predictions receive no detection credit; counts stay in denominator.
                review = {
                    "format_error": False,
                    "contradiction": False,
                    "uncertain": True,
                }
            predictions.append({"id": example["id"], **review})
        all_rows[role] = predictions
        report["reviews"][role] = {
            "metrics": metrics(predictions, truth),
            "failed_ids": failures,
            "valid_predictions": 200 - len(failures),
            "estimated_cost_usd_excluding_tax": cost,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "failure_policy": "Failed predictions receive no detection credit; included in all 200 cases and separately counted.",
        }
    cleanlab = json.loads((base / "qwen-local-cleanlab.json").read_text())["rows"]
    combined = [
        {
            "id": n["id"],
            **{key: n[key] or c[key] for key in ("format_error", "contradiction")},
        }
        for n, c in zip(all_rows["nano"], cleanlab, strict=True)
    ]
    report["combined_nano_or_qwen_cleanlab"] = {
        "rule": "Prospectively fixed union; no tuning after outcomes",
        "metrics": metrics(combined, truth),
    }
    report["nano_super_disagreements"] = [
        {"id": n["id"], "nano": n, "super": s}
        for n, s in zip(all_rows["nano"], all_rows["super"], strict=True)
        if any(
            n.get(k) != s.get(k)
            for k in ("format_error", "contradiction", "uncertain", "correct_label")
        )
    ]
    report["human_review_required"] = False
    out.mkdir(parents=True)
    save(out / "report.json", report)
    save(out / "predictions.json", {**all_rows, "combined": combined})
    print(json.dumps(report["reviews"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("embeddings", "combine"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    if args.action == "embeddings":
        analyze(args.data, args.artifacts, args.out)
    else:
        add_reviews(args.data, args.artifacts, args.base, args.out)
