# Local Python check runner (B4)

`jobs.sandbox.runner.run_python` executes a small Python check in a disposable
Docker container. The investigation loop can explicitly opt into [generated checks](generated-checks.md).
Library probes remain the default.
Token Factory Sandbox beta access has not been requested by this implementation.

Start Docker and provision a trusted image explicitly:

```bash
docker pull python:3.12-slim
uv run python -m jobs.sandbox.runner /absolute/path/check.py --timeout 10
```

The command prints JSON containing the exit code, stdout, stderr, timeout flag,
and output-limit flag. It exits unsuccessfully when the check fails. Source must
fit in 64 KiB. Each output stream is capped at 64 KiB by default. A timeout or
excess output kills the Docker client and forcibly removes its named container.
Cleanup failure raises `SandboxUnavailable`, including the container name for
manual inspection. Missing Docker, daemon or image never causes host execution.
Images are never pulled implicitly. Use `--image python@sha256:...` with a trusted
local digest to reproduce an image exactly.

The container receives no host mounts, environment variables or Docker socket.
It has no external network, runs as UID/GID 65534, drops all Linux capabilities,
and enables no-new-privileges. The root filesystem is read-only; `/tmp` is a
16 MiB tmpfs. Limits are 128 MiB memory (no additional swap), one CPU, 32 processes,
64 file descriptors, and no core dumps. Docker log storage is disabled. Python
runs in isolated mode with bytecode writes disabled. The image's own environment
and installed packages remain accessible; use images that contain no secrets.
Do not put credentials into the source code, which is visible in container
metadata and the local process arguments while running.

Docker shares the container host's kernel and requires a trusted daemon. These
settings are defense in depth, not proof against container escapes. Do not expose
this primitive as a public service for arbitrary code. There is no package
installation or file upload. Generated-check results are persisted in investigation reports.

Run the opt-in integration checks with an available daemon and image:

```bash
RUN_DOCKER_SANDBOX_TESTS=1 uv run pytest -q tests/unit/test_sandbox.py
docker ps -a --filter name=next-experiment-check-
```

The integration checks cover secret environment exclusion, UID, capabilities,
no-new-privileges, filesystem writes, external network denial, timeout, excessive
output, check errors, and missing images. They do not establish resistance to
kernel exploits or validate every resource limit under load.

## Local verification

On October 18, 2026, the opt-in tests passed with Docker 29.4.0 and local image
`python@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17`.
No check containers remained after the run. The full suite was also run with
`RUN_DOCKER_SANDBOX_TESTS=1 uv run --with matplotlib==3.10.7 pytest -q`.
