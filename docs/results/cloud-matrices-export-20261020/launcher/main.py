"""Verify and unpack this job's frozen source/data payload."""
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

payload = Path(os.environ["NEBIUS_OUTPUT_DIR"]) / "input.zip"
assert hashlib.sha256(payload.read_bytes()).hexdigest() == 'c68bdb0c1d18fcdfa627339ec8ec540399874d04dcd5cb7e0997012219cc304d'
root = Path("/tmp/next-experiment-bundle")
root.mkdir(exist_ok=False)
with zipfile.ZipFile(payload) as archive:
    for item in archive.infolist():
        name = Path(item.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    archive.extractall(root)
subprocess.run([sys.executable, str(root / "main.py")], check=True)
