"""Persist local or provider embeddings for the frozen semantic audit, without retries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core.db.spending import SpendingLedger
from core.eval.runtime import file_digest
from scripts.fresh_answer_calibration import provider, save


def read_data(root):
    plan = json.loads((root / "protocol.json").read_text())
    for name, digest in plan["data_sha256"].items():
        if file_digest(root / name) != digest:
            raise ValueError("Audit data drift")
    rows = json.loads((root / "reference.json").read_text()) + json.loads(
        (root / "audit.json").read_text()
    )
    return plan, rows


def normalized(vectors, count):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != count or not np.isfinite(array).all():
        raise ValueError("Invalid embedding shape or values")
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if (lengths < 1e-8).any():
        raise ValueError("Zero embedding")
    return array / lengths


def run(root, output, *, metadata=None, env=None):
    if output.exists():
        raise FileExistsError("Embeddings output already exists; no hidden retry")
    plan, rows = read_data(root)
    output.mkdir(parents=True)
    texts = [row["text"] for row in rows]
    execution = {
        "audit_protocol_sha256": file_digest(root / "protocol.json"),
        "source_sha256": file_digest(__file__),
        "ids": [row["id"] for row in rows],
        "started_at_unix": time.time(),
    }
    save(output / "execution.json", execution)
    if metadata:
        from importlib.metadata import version

        import torch
        from sentence_transformers import SentenceTransformer

        meta = json.loads(metadata.read_text())
        snapshot = Path(meta["snapshot"])
        execution.update(
            model=meta["model"],
            revision=meta["revision"],
            runtime={
                p: version(p)
                for p in ("torch", "transformers", "sentence-transformers", "numpy")
            },
            snapshot_sha256={
                str(p.relative_to(snapshot)): file_digest(p)
                for p in sorted(snapshot.rglob("*"))
                if p.is_file()
            },
        )
        save(output / "execution.json", execution)
        model = SentenceTransformer(
            str(snapshot),
            device="mps",
            local_files_only=True,
            model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        )
        prompt = model.prompts.get("document", "")
        lengths = [len(model.tokenizer.encode(prompt + t)) for t in texts]
        if max(lengths) > plan["embedding"]["max_length"]:
            raise ValueError("Input would truncate")
        model.max_seq_length = plan["embedding"]["max_length"]
        chunks = []
        for start in range(0, len(texts), plan["embedding"]["batch_size"]):
            end = start + plan["embedding"]["batch_size"]
            chunk = model.encode(
                texts[start:end],
                batch_size=plan["embedding"]["batch_size"],
                prompt=prompt,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            chunks.append(chunk)
            np.save(output / f"batch-{start:04d}.npy", chunk)
            if start % 80 == 0:
                print(meta["model"], start, flush=True)
        vectors = np.concatenate(chunks)
        execution.update(cost_usd=0, max_input_tokens=max(lengths), prompt=prompt)
    else:
        tf = provider(env)
        ledger = SpendingLedger(
            "artifacts/remaining-queue-20261020/spending.sqlite3", 25
        )
        model = plan["embedding"]["provider"]
        execution.update(
            model=model,
            price_usd_per_million=0.01,
            budget_multiplier=1.2,
            price_source="https://nebius.com/services/token-factory/models/qwen-models-inference",
        )
        save(output / "execution.json", execution)
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            operation = f"semantic-audit-20261020-v1-embedding-{start}"
            reserve = (sum(len(t.encode()) for t in batch) + 4096) * 0.01 / 1e6 * 1.2
            ledger.reserve(operation, "A30 Qwen3-Embedding-8B", reserve)
            save(
                output / f"pending-{start:04d}.json",
                {"operation": operation, "reserved_usd": reserve},
            )
            result = tf.client.embeddings.create(
                model=model, input=batch, encoding_format="float"
            )
            receipt = {
                "request_id": getattr(result, "_request_id", None),
                "usage": result.usage.model_dump(),
                "model": result.model,
            }
            save(output / f"receipt-{start:04d}.json", receipt)
            ledger.record_provider_receipt(
                operation,
                request_id=receipt["request_id"],
                response_id=None,
                outcome="response",
                http_status=200,
            )
            if sorted(r.index for r in result.data) != list(range(len(batch))):
                raise ValueError("Embedding indices differ from request")
            chunk = normalized(
                [r.embedding for r in sorted(result.data, key=lambda r: r.index)],
                len(batch),
            )
            np.save(output / f"batch-{start:04d}.npy", chunk)
            vectors.extend(chunk)
            ledger.settle(operation, result.usage.prompt_tokens * 0.01 / 1e6 * 1.2)
            print(model, start, flush=True)
    array = normalized(vectors, len(rows))
    np.save(output / "embeddings.npy", array)
    execution.update(
        completed_at_unix=time.time(),
        shape=list(array.shape),
        embeddings_sha256=file_digest(output / "embeddings.npy"),
    )
    save(output / "execution.json", execution)
    return execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    result = run(args.data, args.out, metadata=args.metadata, env=args.env_file)
    print(json.dumps({"model": result["model"], "shape": result["shape"]}))
