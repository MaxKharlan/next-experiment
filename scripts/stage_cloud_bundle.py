"""Stage an oversized frozen bundle in the job's dedicated bucket prefix."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from core.config import Settings
from core.connectors.storage import S3Storage
from core.eval.runtime import file_digest


def stage(root, job_name, env):
    bundle = root / "bundle"
    plan = json.loads((bundle / "experiment.json").read_text())
    expected = set(plan["bundle_files_sha256"]) | {"experiment.json"}
    files = {
        p.relative_to(bundle).as_posix(): p for p in bundle.rglob("*") if p.is_file()
    }
    if set(files) != expected or any(p.is_symlink() for p in files.values()):
        raise ValueError("Frozen bundle has extra, missing or linked files")
    for name, digest in plan["bundle_files_sha256"].items():
        if file_digest(files[name]) != digest:
            raise ValueError(f"Bundle changed: {name}")
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.writestr(name, path.read_bytes())
    raw = content.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    storage = S3Storage.from_settings(Settings(_env_file=env))
    key = f"runs/{job_name}/input.zip"
    if storage.exists(key):
        raise FileExistsError("Staged input already exists; verify it before resuming")
    storage.put_bytes(key, raw)
    if hashlib.sha256(storage.get_bytes(key)).hexdigest() != digest:
        raise ValueError("Uploaded bundle verification failed")
    launcher = root / "launcher"
    launcher.mkdir(exist_ok=False)
    source = '''"""Verify and unpack this job's frozen source/data payload."""
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

payload = Path(os.environ["NEBIUS_OUTPUT_DIR"]) / "input.zip"
assert hashlib.sha256(payload.read_bytes()).hexdigest() == DIGEST
root = Path("/tmp/next-experiment-bundle")
root.mkdir(exist_ok=False)
with zipfile.ZipFile(payload) as archive:
    for item in archive.infolist():
        name = Path(item.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    archive.extractall(root)
subprocess.run([sys.executable, str(root / "main.py")], check=True)
'''
    (launcher / "main.py").write_text(source.replace("DIGEST", repr(digest)))
    record = {
        "bucket": storage.bucket,
        "key": key,
        "sha256": digest,
        "bytes": len(raw),
        "files": len(files),
        "job_name": job_name,
        "launcher_sha256": file_digest(launcher / "main.py"),
    }
    (root / "staged-input.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.root, args.job_name, args.env_file)))
