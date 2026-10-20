import os
import runpy
import subprocess
import venv


def test_isolated_installer_ignores_image_constraints_but_preserves_credentials(
    monkeypatch,
):
    monkeypatch.setenv("PIP_CONSTRAINT", "/image/constraints.txt")
    monkeypatch.setenv("PIP_BUILD_CONSTRAINT", "/image/build.txt")
    monkeypatch.setenv("PIP_REQUIREMENT", "/image/requirements.txt")
    monkeypatch.setenv("PIP_CONFIG_FILE", "/image/pip.conf")
    monkeypatch.setenv("NEBIUS_OUTPUT_DIR", "/outputs")
    monkeypatch.setattr(venv.EnvBuilder, "create", lambda *a: None)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    execs = []
    monkeypatch.setattr(os, "execv", lambda *a: execs.append(a))
    runpy.run_path("jobs/cloud/bootstrap.py")
    arguments, options = calls[0]
    assert arguments[0][-1].endswith("dependencies.txt")
    assert options["check"] and options["timeout"] == 1200
    env = options["env"]
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["NEBIUS_OUTPUT_DIR"] == "/outputs"
    assert (
        not {"PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT", "PIP_REQUIREMENT"} & env.keys()
    )
    assert os.environ["PIP_CONSTRAINT"] == "/image/constraints.txt"
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
    assert execs[0][1][-1].endswith("worker.py")
