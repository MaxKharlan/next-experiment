"""Install the pinned runtime outside the container image's Python packages."""

import os
import subprocess
import sys
import venv
from pathlib import Path

root = Path(__file__).parent
environment = Path("/tmp/next-experiment-runtime")
venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment)
python = str(environment / "bin/python")
subprocess.run(
    [sys.executable, "-m", "pip", "--python", python, "install", "--no-cache-dir", "-r", str(root / "dependencies.txt")],
    check=True,
    timeout=1200,
)
os.execv(python, [python, str(root / "worker.py")])
