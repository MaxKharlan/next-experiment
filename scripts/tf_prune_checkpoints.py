"""Delete the files of Token Factory checkpoints that nothing here uses.

Token Factory counts every checkpoint file against the account's limit of 5000
fine-tune files: a checkpoint is 6 files and a job saves one per epoch, so a
12-epoch job takes 72. The benchmark, the references and the retrain only ever
download a job's last checkpoint (tf_runner.download_last_checkpoint), so this keeps
the last one and deletes the files of the others.

It looks only at jobs this repository created (job IDs found in the JSON files under
artifacts/), only at jobs that have finished, and never deletes uploaded training or
validation files. With --downloaded it also deletes the last checkpoint of a job
whose adapter is already in local storage. Pruned jobs are remembered in
artifacts/tf/pruned.json, so a repeat run only asks about new jobs.

    uv run python -m scripts.tf_prune_checkpoints              # what it would delete
    uv run python -m scripts.tf_prune_checkpoints --delete
    uv run python -m scripts.tf_prune_checkpoints --delete --watch 10   # next to a run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

JOB_ID = re.compile(r"ftjob-[0-9a-f]{32}")
FINISHED = {"succeeded", "failed", "cancelled"}
STATE = "tf/pruned.json"
MAX_BYTES = 2_000_000


def local_job_ids(root: Path) -> set[str]:
    """Every job ID written to a JSON or JSONL file under root."""
    ids: set[str] = set()
    for path in [*root.rglob("*.json"), *root.rglob("*.jsonl")]:
        try:
            # skips this script's own state and the tokenizer files of checkpoints
            if path.name == Path(STATE).name or path.stat().st_size > MAX_BYTES:
                continue
            ids.update(JOB_ID.findall(path.read_text(errors="ignore")))
        except OSError:
            continue
    return ids


def downloaded_checkpoints(root: Path) -> set[str]:
    """IDs of checkpoints whose adapter is in local storage."""
    return {p.parent.name for p in root.rglob("adapter_model.safetensors")}


def files_to_delete(
    tf: Any, job_id: str, *, downloaded: set[str] | None = None
) -> list[str] | None:
    """File IDs to delete for a job, or None while the job is still running."""
    if tf.job(job_id).status not in FINISHED:
        return None
    checkpoints = list(tf.checkpoints(job_id))
    if not checkpoints:
        return []
    last = max(checkpoints, key=lambda c: c.step_number)
    keep = set() if downloaded is not None and last.id in downloaded else {last.id}
    return [f for c in checkpoints if c.id not in keep for f in c.result_files]


def prune(
    tf: Any,
    job_ids: Iterable[str],
    done: dict[str, int],
    delete: Callable[[str], None] | None,
    *,
    downloaded: set[str] | None = None,
    on_job: Callable[[str, int], None] | None = None,
) -> tuple[int, int]:
    """Prune every finished job not in done; returns (jobs pruned, files deleted).

    Without delete nothing changes and done is left as it was. on_job gets each
    pruned job and its file count as soon as the job is done.
    """
    jobs = files = 0
    for job_id in sorted(set(job_ids) - set(done)):
        found = files_to_delete(tf, job_id, downloaded=downloaded)
        if found is None:
            continue
        if delete is not None:
            for file_id in found:
                delete(file_id)
            done[job_id] = len(found)
        jobs += 1
        files += len(found)
        if on_job is not None:
            on_job(job_id, len(found))
    return jobs, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete", action="store_true", help="delete; otherwise only count"
    )
    parser.add_argument(
        "--downloaded",
        action="store_true",
        help="also delete last checkpoints already in local storage",
    )
    parser.add_argument(
        "--watch", type=float, metavar="MIN", help="repeat every MIN minutes"
    )
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    import openai

    from core.connectors.token_factory import TokenFactory

    tf = TokenFactory.from_settings()
    root = Path(args.artifacts)

    def delete(file_id: str) -> None:
        try:
            tf.client.files.delete(file_id)
        except openai.NotFoundError:
            pass  # already gone

    while True:
        run_once(tf, root, delete if args.delete else None, downloaded=args.downloaded)
        if not args.watch:
            break
        time.sleep(args.watch * 60)


def run_once(
    tf: Any, root: Path, delete: Callable[[str], None] | None, *, downloaded: bool
) -> None:
    """One pass over the jobs known under root, printing a line per pruned job."""
    state = root / STATE
    verb = "deleted" if delete is not None else "would delete"
    done = json.loads(state.read_text()) if state.exists() else {}
    before = len(done)
    ids = local_job_ids(root)
    left = len(ids - set(done))
    count = 0

    def on_job(job_id: str, n: int) -> None:
        nonlocal count
        count += 1
        if delete is not None:  # saved after every job, so a stop loses nothing
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps(done, indent=2))
        print(f"  {count}/{left} {job_id}: {verb} {n} files", flush=True)

    jobs, files = prune(
        tf,
        ids,
        done,
        delete,
        downloaded=downloaded_checkpoints(root) if downloaded else None,
        on_job=on_job,
    )
    print(
        f"{time.strftime('%H:%M')} {len(ids)} jobs known, {before} pruned before; "
        f"{verb} {files} files of {jobs} finished jobs",
        flush=True,
    )


if __name__ == "__main__":
    main()
