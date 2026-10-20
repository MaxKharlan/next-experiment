import os
import subprocess
import sys
import venv
from pathlib import Path

root = Path(__file__).parent
environment = Path("/tmp/next-experiment-runtime")
venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment)
python = str(environment / "bin/python")
env = dict(os.environ)
env.pop("PYTHONPATH", None)
env.pop("PYTHONHOME", None)
env["PYTHONNOUSERSITE"] = "1"
env["KAGGLE_BOOTSTRAP_PYTHON"] = sys.executable
env["NVIDIA_TF32_OVERRIDE"] = "0"
env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
output = Path(env["NEBIUS_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
with (output / "dependency-install.log").open("w") as log:
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "--python", python, "install",
         "--no-cache-dir", "--report", str(output / "dependency-install.json"),
         "-r", str(root / "dependencies.txt")],
        env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200,
    )
# Still collect all import/version/CUDA failures if installation failed.
preflight = subprocess.run([python, str(root / "preflight.py")], env=env, timeout=300)
if installed.returncode or preflight.returncode:
    raise SystemExit("Dependency preflight failed; see evaluation/preflight.json and dependency-install.log")
os.execve(python, [python, str(root / "worker.py")], env)
