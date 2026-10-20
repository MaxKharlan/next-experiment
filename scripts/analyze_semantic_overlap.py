"""Compare three embedding methods on the same authored overlap stress controls."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from core.eval.runtime import file_digest
from core.probes.level0.semantic import confusion
from core.probes.level0.semantic_overlap import compare_prompts
from scripts.fresh_answer_calibration import save


def analyze(root, artifacts):
    output = root / "report.json"
    if output.exists():
        raise FileExistsError("Overlap report already exists")
    plan = json.loads((root / "protocol.json").read_text())
    pairs = json.loads((root / "controls.json").read_text())["pairs"]
    results = {}
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    for model in ("qwen-local", "nvidia-local", "qwen-provider"):
        src = artifacts / f"overlap-{model}"
        meta = json.loads((src / "execution.json").read_text())
        if meta["audit_protocol_sha256"] != file_digest(root / "protocol.json") or meta[
            "embeddings_sha256"
        ] != file_digest(src / "embeddings.npy"):
            raise ValueError("Embedding provenance mismatch")
        vectors = np.load(src / "embeddings.npy")
        expected = [
            p["id"] + "-" + side for p in pairs for side in ("train", "validation")
        ]
        if meta["ids"] != expected:
            raise ValueError("Embedding order mismatch")
        rows = [
            {
                "id": p["id"],
                "category": p["category"],
                "same_request": p["same_request"],
                **compare_prompts(
                    p["train_prompt"],
                    p["validation_prompt"],
                    vectors[2 * i],
                    vectors[2 * i + 1],
                    threshold=plan["cosine_min"],
                ),
            }
            for i, p in enumerate(pairs)
        ]
        results[model] = {
            "rows": rows,
            "metrics": confusion(
                [r["same_request"] for r in rows], [r["candidate"] for r in rows]
            ),
            "paraphrases": confusion(
                [r["same_request"] for r in rows if r["category"] == "paraphrase"],
                [r["candidate"] for r in rows if r["category"] == "paraphrase"],
            ),
        }
        for name in ("execution.json", "embeddings.npy"):
            shutil.copy2(src / name, evidence / f"{model}-{name}")
    save(
        output,
        {
            "protocol_sha256": file_digest(root / "protocol.json"),
            "source_sha256": {
                name: file_digest(name)
                for name in (
                    "scripts/analyze_semantic_overlap.py",
                    "core/probes/level0/semantic_overlap.py",
                )
            },
            "results": results,
            "scope": plan["scope"],
        },
    )
    save(
        root / "manifest.json",
        {
            str(p.relative_to(root)): file_digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    )
    print(
        json.dumps(
            {
                k: {"all": v["metrics"], "paraphrases": v["paraphrases"]}
                for k, v in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    a = p.parse_args()
    analyze(a.root, a.artifacts)
