"""Install the pinned runtime outside the container image's Python packages."""

import os
import subprocess
import venv
from pathlib import Path

root = Path(__file__).parent
environment = Path("/tmp/next-experiment-runtime")
venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
python = str(environment / "bin/python")
subprocess.run(
    [python, "-m", "pip", "install", "--no-cache-dir", "-r", str(root / "dependencies.txt")],
    check=True,
    timeout=1200,
)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
os.execv(python, [python, str(root / "worker.py")])
