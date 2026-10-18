"""Evaluate completed epoch-grid adapters locally; never submits training jobs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from core.connectors.storage import LocalStorage
from core.connectors.token_factory import TokenFactory
from core.interventions.tf_runner import download_last_checkpoint
from jobs.eval_local.exact_match import evaluate_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    tf = TokenFactory.from_settings()
    storage = LocalStorage("artifacts")
    digest = hashlib.sha256(args.valid.read_bytes()).hexdigest()
    examples = len(args.valid.read_text().splitlines())
    while True:
        for path in sorted(args.directory.glob("s*/*.json")):
            if path.name == "jobs.json" or path.name.endswith("-evaluation.json"):
                continue
            row = json.loads(path.read_text())
            if row.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(f"Cannot evaluate unsuccessful job: {row['job_id']}")
            if row.get("status") != "succeeded":
                continue
            output = path.with_name(path.stem + "-evaluation.json")
            if output.exists():
                saved = json.loads(output.read_text())
                if (
                    saved["job_id"] != row["job_id"]
                    or saved.get("validation_sha256") != digest
                ):
                    raise ValueError(
                        "Existing evaluation belongs to a different job or dataset"
                    )
                continue
            adapter = download_last_checkpoint(
                tf, storage, row["job_id"], "epoch-sensitivity/adapters"
            )
            score = evaluate_adapter(adapter, args.valid, device=args.device)
            result = {
                "job_id": row["job_id"],
                "exact_match": score,
                "evaluation": "validation",
                "examples": examples,
                "validation_sha256": digest,
                "device": args.device,
            }
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(output)
            print(row["run"], score, flush=True)
        if (
            not args.watch
            or len(list(args.directory.glob("s*/*-evaluation.json"))) == 9
        ):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
