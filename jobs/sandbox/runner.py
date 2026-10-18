"""Opt-in Docker execution for small, untrusted Python checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_IMAGE = "python:3.12-slim"
MAX_SOURCE_BYTES = 64 * 1024


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    output_limited: bool


class SandboxUnavailable(RuntimeError):
    """Docker could not start or remove the check container."""


def run_python(
    source: str,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: float = 10,
    max_output_bytes: int = 64 * 1024,
) -> SandboxResult:
    """Run code in a disposable container; never fall back to host execution.

    The image must be trusted and already present locally. Output is bounded
    per stream. Docker daemon access remains a privileged host capability.
    """
    if not 0 < timeout <= 300:
        raise ValueError("timeout must be in (0, 300]")
    if not 0 < max_output_bytes <= 1024 * 1024:
        raise ValueError("max_output_bytes must be in [1, 1048576]")
    if len(source.encode()) > MAX_SOURCE_BYTES or "\x00" in source:
        raise ValueError("source must be at most 64 KiB and contain no NUL")
    if not image or image.startswith("-"):
        raise ValueError("a trusted local image is required")
    name = f"next-experiment-check-{uuid.uuid4().hex}"
    command = [
        "docker",
        "run",
        "--name",
        name,
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=128m",
        "--memory-swap=128m",
        "--cpus=1",
        "--ulimit=nofile=64:64",
        "--ulimit=core=0",
        "--log-driver=none",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        "--workdir=/tmp",
        "--entrypoint=python",
        image,
        "-I",
        "-B",
        "-c",
        source,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise SandboxUnavailable("Docker CLI is unavailable") from exc

    buffers = [bytearray(), bytearray()]
    limited = threading.Event()

    def drain(stream, buffer):
        while chunk := stream.read(4096):
            remaining = max_output_bytes - len(buffer)
            buffer.extend(chunk[:remaining])
            if len(chunk) > remaining:
                limited.set()
                process.kill()
        stream.close()

    readers = [
        threading.Thread(target=drain, args=(stream, buffer), daemon=True)
        for stream, buffer in zip((process.stdout, process.stderr), buffers)
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
    finally:
        # Killing the CLI does not reliably kill the container.
        try:
            cleanup = subprocess.run(
                ["docker", "rm", "--force", name],
                capture_output=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"Could not remove container {name}") from exc
        finally:
            for reader in readers:
                reader.join(timeout=5)
        if cleanup.returncode and b"No such container" not in cleanup.stderr:
            raise SandboxUnavailable(f"Could not verify removal of container {name}")
    if process.returncode == 125:
        raise SandboxUnavailable(buffers[1].decode(errors="replace"))
    return SandboxResult(
        exit_code=process.returncode,
        stdout=buffers[0].decode(errors="replace"),
        stderr=buffers[1].decode(errors="replace"),
        timed_out=timed_out,
        output_limited=limited.is_set(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    result = run_python(args.source.read_text(), image=args.image, timeout=args.timeout)
    print(json.dumps(asdict(result)))
    success = (
        result.exit_code == 0 and not result.timed_out and not result.output_limited
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
