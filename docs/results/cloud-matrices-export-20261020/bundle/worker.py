"""Execute each frozen study once, with separate output and time limits."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).parent
plan = json.loads((root / "experiment.json").read_text())
for name, digest in plan["bundle_files_sha256"].items():
    with (root / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
output = Path(os.environ["NEBIUS_OUTPUT_DIR"])
results = {"status": "running", "studies": {}, "protocol_sha256": hashlib.sha256((root / "experiment.json").read_bytes()).hexdigest()}
started = time.monotonic()
for study in plan["studies"]:
    destination = output / study["name"]
    destination.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NEBIUS_OUTPUT_DIR": str(destination)}
    try:
        run = subprocess.run([sys.executable, str(root / study["worker"])], env=env, timeout=study["timeout_seconds"])
        entry = {"returncode": run.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        entry = {"returncode": None, "timed_out": True}
    entry["elapsed_total_seconds"] = time.monotonic() - started
    results["studies"][study["name"]] = entry
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
results["status"] = "complete" if all(r["returncode"] == 0 for r in results["studies"].values()) else "partial_or_failed"
(output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
sys.exit(0 if results["status"] == "complete" else 1)
