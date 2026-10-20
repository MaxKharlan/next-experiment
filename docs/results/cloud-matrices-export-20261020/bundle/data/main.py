"""Install the frozen project runtime independently of the image's pip pins."""

import os
import subprocess
import venv
from pathlib import Path

root = Path(__file__).parent
environment = Path("/tmp/next-experiment-runtime")
venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
python = str(environment / "bin/python")
install_env = os.environ.copy()
for variable in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT", "PIP_REQUIREMENT"):
    install_env.pop(variable, None)
install_env["PIP_CONFIG_FILE"] = os.devnull
subprocess.run(
    [
        python,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-r",
        str(root / "dependencies.txt"),
    ],
    env=install_env,
    check=True,
    timeout=1200,
)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
os.execv(python, [python, str(root / "worker.py")])
