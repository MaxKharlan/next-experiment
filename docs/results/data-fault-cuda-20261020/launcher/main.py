"""Verify and unpack this job's frozen source/data payload."""
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

payload = Path(os.environ["NEBIUS_OUTPUT_DIR"]) / "input.zip"
assert hashlib.sha256(payload.read_bytes()).hexdigest() == '2108ec7260ca23817a18178e7713b60cdfd80984f29fdcd2104e442ac489297c'
root = Path("/tmp/next-experiment-bundle")
root.mkdir(exist_ok=False)
with zipfile.ZipFile(payload) as archive:
    for item in archive.infolist():
        name = Path(item.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    archive.extractall(root)
subprocess.run([sys.executable, str(root / "main.py")], check=True)
