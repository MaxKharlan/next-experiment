"""Validation without Docker; opt-in integration tests use a real daemon."""

import os
import shutil
import subprocess

import pytest

from jobs.sandbox.runner import SandboxUnavailable, run_python


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": 301},
        {"max_output_bytes": 0},
        {"image": "--privileged"},
        {"source": "x" * 65537},
        {"source": "\x00"},
    ],
)
def test_invalid_request(kwargs):
    with pytest.raises(ValueError):
        run_python(**({"source": "print(1)"} | kwargs))


def test_no_host_fallback(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SandboxUnavailable, match="CLI"):
        run_python("raise AssertionError('must not execute on host')")


live = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_SANDBOX_TESTS") != "1" or not shutil.which("docker"),
    reason="requires RUN_DOCKER_SANDBOX_TESTS=1 and a local Python image",
)


@live
def test_isolation(monkeypatch):
    monkeypatch.setenv("TOKEN_FACTORY_API_KEY", "host-secret-sentinel")
    result = run_python("""
import os, pathlib, socket
assert "TOKEN_FACTORY_API_KEY" not in os.environ
assert os.getuid() == 65534
assert not pathlib.Path("/var/run/docker.sock").exists()
try:
    pathlib.Path("/host-write-test").write_text("forbidden")
except OSError:
    pass
else:
    raise AssertionError("root filesystem is writable")
assert "NoNewPrivs:\\t1" in pathlib.Path("/proc/self/status").read_text()
assert "CapEff:\\t0000000000000000" in pathlib.Path("/proc/self/status").read_text()
try:
    socket.create_connection(("1.1.1.1", 443), timeout=1)
except OSError:
    pass
else:
    raise AssertionError("network is available")
pathlib.Path("/tmp/result").write_text("ok")
print("isolated")
""")
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "isolated\n"


@live
def test_timeout():
    result = run_python("while True: pass", timeout=1)
    assert result.timed_out
    assert result.exit_code != 0


@live
def test_output_limit():
    result = run_python(
        "while True: print('x' * 8192, flush=True)", max_output_bytes=1024
    )
    assert result.output_limited
    assert len(result.stdout.encode()) == 1024


@live
def test_failure_and_stderr():
    result = run_python("raise ValueError('check failed')")
    assert result.exit_code == 1
    assert "check failed" in result.stderr


@live
def test_missing_image():
    with pytest.raises(SandboxUnavailable):
        run_python("print(1)", image="next-experiment-missing-image:never")


@live
def test_containers_removed_after_timeout():
    def containers():
        return set(
            subprocess.run(
                ["docker", "ps", "-aq", "--filter", "name=next-experiment-check-"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.split()
        )

    before = containers()
    assert run_python("while True: pass", timeout=1).timed_out
    assert containers() == before
