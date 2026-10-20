"""Score the frozen clean reference grid after the existing MPS studies finish."""

import argparse
import fcntl
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

from core.eval.data import read_examples
from core.eval.generate import answer, free_cache, load_model
from core.eval.metrics import get_metric
from core.eval.runtime import file_digest
from core.eval.structured import TASK_EVALUATORS
from core.interventions.tf_runner import weights_digest
from scripts.execute_reference_grid import verify_reference_runtime
from scripts.fresh_answer_calibration import save


def verify_rows(record, examples, binding, metric):
    if any(record.get(key) != value for key, value in binding.items()):
        raise ValueError("Prediction binding changed")
    rows = record["rows"]
    if len(rows) > len(examples):
        raise ValueError("Too many predictions")
    for row, example in zip(rows, examples):
        if (
            (row["line"], row["expected"], row["source"])
            != (example.line, example.expected, example.source)
            or row["score"] != metric(row["answer"], example.expected)
            or row["hit_limit"]
            and row["stopped_on_eos"]
        ):
            raise ValueError("Prediction identity or score changed")
    if record.get("complete") and len(rows) != len(examples):
        raise ValueError("Incomplete predictions marked complete")
    return rows


def choose(scores, jobs):
    if not scores or set(scores) != set(jobs):
        raise ValueError("Every grid point must be scored before selection")
    return min(scores, key=lambda name: (-scores[name], jobs[name]["grid_index"]))


def evaluate(root, plan, digest, model_id, task, role, name, entry):
    import torch

    spec = plan["tasks"][task][role]
    if file_digest(spec["path"]) != spec["sha256"]:
        raise ValueError("Evaluation dataset changed")
    examples = read_examples(spec["path"])
    runtime = plan["runtimes"][model_id]
    metric = get_metric(TASK_EVALUATORS[task][1])
    path = root / "scores" / f"{name}-{role}.json"
    binding = {
        "protocol_sha256": digest,
        "data_sha256": spec["sha256"],
        "runtime": runtime,
        "job_id": entry.get("job_id"),
        "weights_sha256": entry.get("weights_sha256"),
    }
    record = (
        json.loads(path.read_text())
        if path.exists()
        else {**binding, "rows": [], "complete": False}
    )
    rows = verify_rows(record, examples, binding, metric)
    if record["complete"]:
        return sum(row["score"] for row in rows) / len(rows)
    adapter = entry.get("adapter")
    if adapter and weights_digest(Path(adapter)) != entry["weights_sha256"]:
        raise ValueError("Adapter weights changed")
    verify_reference_runtime(runtime, adapter)
    model, tokenizer, _ = load_model(adapter, base_model=runtime["snapshot"])
    model.to("mps").eval()
    with torch.inference_mode():
        for example in examples[len(rows) :]:
            rows.append(
                asdict(
                    answer(
                        model,
                        tokenizer,
                        example,
                        "mps",
                        metric,
                        max_new_tokens=plan["generation"]["max_new_tokens"][task],
                    )
                )
            )
            save(path, record)
            free_cache("mps")
            if len(rows) % 20 == 0:
                print(name, role, len(rows), "/", len(examples), flush=True)
    record.update(
        complete=True,
        exact_match=sum(row["score"] for row in rows) / len(rows),
        stopped_on_eos=sum(row["stopped_on_eos"] for row in rows) / len(rows),
        hit_limit=sum(row["hit_limit"] for row in rows) / len(rows),
    )
    save(path, record)
    del model, tokenizer
    gc.collect()
    free_cache("mps")
    return record["exact_match"]


def run(root):
    digest = file_digest(root / "protocol.json")
    if digest != (root / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol changed")
    plan = json.loads((root / "protocol.json").read_text())
    for name, expected in plan["source_sha256"].items():
        if file_digest(name) != expected:
            raise ValueError("Source changed")
    state = json.loads((root / "state.json").read_text())
    if set(state) != set(plan["jobs"]) or any(
        not entry.get("settled") or entry["status"] != "succeeded"
        for entry in state.values()
    ):
        raise ValueError("All reference training jobs must finish before scoring")
    selected = {}
    for model in plan["runtimes"]:
        for task in plan["tasks"]:
            key = f"{model.split('/')[-1]}-{task}"
            base = {
                role: evaluate(root, plan, digest, model, task, role, key + "-base", {})
                for role in ("validation", "check")
            }
            jobs = {
                name: spec
                for name, spec in plan["jobs"].items()
                if spec["model"] == model and spec["task"] == task
            }
            scores = {
                name: evaluate(
                    root, plan, digest, model, task, "validation", name, state[name]
                )
                for name in jobs
            }
            best = choose(scores, jobs)
            choice_path = root / "selection" / f"{key}.json"
            choice = {
                "protocol_sha256": digest,
                "base": base,
                "validation": scores,
                "selected": best,
                "selection_rule": plan["selection"],
            }
            if choice_path.exists() and json.loads(choice_path.read_text()) != choice:
                raise ValueError("Previous selection changed")
            save(choice_path, choice)
            check = evaluate(
                root, plan, digest, model, task, "check", best, state[best]
            )
            selected[key] = {**choice, "selected_check_exact_match": check}
            save(
                root / "results.json",
                {
                    "status": "partial",
                    "pairs": selected,
                    "scope": plan["scope"],
                },
            )
    save(
        root / "results.json",
        {
            "status": "complete",
            "pairs": selected,
            "scope": plan["scope"],
            "llama": plan["llama"],
        },
    )


def wait_dependencies(root):
    rest = Path("docs/results/rest-code-investigation-20261020/execution-status.json")
    deadline = time.monotonic() + 12 * 3600
    while True:
        status = json.loads(rest.read_text()) if rest.exists() else {}
        state_path = root / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        plan = json.loads((root / "protocol.json").read_text())
        if (
            status.get("status") == "trace_complete"
            and set(state) == set(plan["jobs"])
            and all(entry.get("settled") for entry in state.values())
        ):
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Training or REST MPS dependency unfinished")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait:
        wait_dependencies(args.root)
    with (args.root.parent / "reference-scoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The REST runner starts only after all twelve ledger scores complete.
        status = json.loads(
            Path(
                "docs/results/rest-code-investigation-20261020/execution-status.json"
            ).read_text()
        )
        if status.get("status") != "trace_complete":
            raise ValueError("REST study must finish before acquiring MPS")
        run(args.root)
