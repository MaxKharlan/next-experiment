"""Bounded Serverless Jobs through Nebius CLI and explicit artifact storage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    name: str
    project: str
    entry: Path
    image: str
    platform: str
    preset: str
    bundle_sha256: dict[str, str]
    output_bucket: str | None = None
    output_bucket_id: str | None = None
    timeout_hours: int = 1
    disk_gib: int = 120

    def validate(self):
        if not re.fullmatch(r"[a-z][a-z0-9-]{2,62}", self.name):
            raise ValueError("Require a unique lowercase job name")
        if not self.project.startswith("project-"):
            raise ValueError("Explicit project ID required")
        if type(self.timeout_hours) is not int or not 1 <= self.timeout_hours <= 4:
            raise ValueError("Jobs are bounded to one through four hours")
        if type(self.disk_gib) is not int or not 20 <= self.disk_gib <= 250:
            raise ValueError("Explicit disk size from 20 through 250 GiB required")
        for value in (self.image, self.platform, self.preset):
            if not value or any(c.isspace() for c in value):
                raise ValueError("Explicit image, platform and preset required")
        if self.output_bucket is not None and not re.fullmatch(
            r"[a-z0-9][a-z0-9.-]+", self.output_bucket
        ):
            raise ValueError("Use an explicit bucket name or ID")
        if bool(self.output_bucket) != bool(self.output_bucket_id) or (
            self.output_bucket_id
            and not re.fullmatch(r"storagebucket-[a-z0-9]+", self.output_bucket_id)
        ):
            raise ValueError(
                "Artifact storage requires both bucket name and resource ID"
            )
        root = self.entry.parent
        actual = {}
        for path in root.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError("Symlinks are forbidden in job bundles")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if any(part.startswith(".") for part in Path(relative).parts):
                    raise ValueError("Hidden files are forbidden in job bundles")
                actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.entry.name not in actual or actual != self.bundle_sha256:
            raise ValueError("Job bundle differs from the explicit file manifest")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class ServerlessJobs:
    """A failed or ambiguous submission retains its hold and is never retried."""

    def __init__(self, ledger, records: Path, *, cli="nebius", run=subprocess.run):
        self.ledger = ledger
        self.records = Path(records)
        self.cli = cli
        self.run = run

    def _command(self, args, *, timeout=120):
        result = self.run(
            [self.cli, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            # Persist CLI errors separately; do not misread a failed call as success.
            raise RuntimeError(
                f"Nebius CLI exited {result.returncode}: {result.stderr[-2000:]}"
            )
        return result.stdout

    @staticmethod
    def _arguments(spec):
        args = [
            "ai",
            "job",
            "run",
            str(spec.entry),
            "--name",
            spec.name,
            "--parent-id",
            spec.project,
            "--platform",
            spec.platform,
            "--preset",
            spec.preset,
            "--image",
            spec.image,
            "--disk-size",
            f"{spec.disk_gib}Gi",
            "--timeout",
            f"{spec.timeout_hours}h",
        ]
        if spec.output_bucket:
            args += ["--output", spec.output_bucket_id]
        else:
            args += ["--no-output"]
        return args

    def validate(self, spec):
        spec.validate()
        return self._command([*self._arguments(spec), "--dry-run", "--format", "json"])

    def submit(self, spec, *, reservation_usd):
        spec.validate()
        if (
            isinstance(reservation_usd, bool)
            or not math.isfinite(reservation_usd)
            or reservation_usd <= 0
        ):
            raise ValueError("A positive full-duration reservation is required")
        self.records.mkdir(parents=True, exist_ok=True)
        record = self.records / f"{spec.name}.json"
        with (self.records / f"{spec.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            request = asdict(spec)
            request["entry"] = str(spec.entry)
            if record.exists():
                previous = json.loads(record.read_text())
                if previous["request"] != request:
                    raise ValueError("Conflicting job request")
                if previous.get("job_id"):
                    return previous["job_id"]
                raise ValueError(
                    "Ambiguous submission; reconcile the existing job first"
                )
            self.ledger.reserve(spec.name, "serverless_job", reservation_usd)
            state = {
                "request": request,
                "reservation_usd": reservation_usd,
                "status": "submitting",
            }
            _save(record, state)
            try:
                operation = json.loads(
                    self._command(
                        [*self._arguments(spec), "--detach", "--format", "json"]
                    )
                )
                job_id = operation["resource_id"]
                if not isinstance(job_id, str) or not job_id.startswith("aijob-"):
                    raise ValueError("Submission response did not identify a job")
            except Exception as error:
                state.update(status="ambiguous", error=str(error))
                _save(record, state)
                raise
            state.update(status="submitted", job_id=job_id, operation=operation)
            _save(record, state)
            return job_id

    def status(self, job_id):
        if not re.fullmatch(r"aijob-[a-z0-9]+", job_id):
            raise ValueError("Invalid job ID")
        return json.loads(
            self._command(["ai", "job", "get", job_id, "--format", "json"])
        )

    def cancel(self, job_id):
        self.status(job_id)  # Validate the identity and obtain current state first.
        return self._command(["ai", "job", "cancel", job_id, "--format", "json"])

    def fetch_json(self, storage, spec, job_id, filename, *, expected_sha256=None):
        status = self.status(job_id)
        metadata = status.get("metadata", {})
        if (
            metadata.get("name") != spec.name
            or metadata.get("parent_id") != spec.project
        ):
            raise ValueError("Job does not match the requested artifact namespace")
        if status.get("status", {}).get("state") != "COMPLETED":
            raise ValueError("Artifacts require a completed job")
        if (
            not spec.output_bucket
            or getattr(storage, "bucket", None) != spec.output_bucket
        ):
            raise ValueError("Artifact storage must match the job's output bucket")
        expected_prefix = f"runs/{spec.name}/"
        if not any(
            volume.get("source") == spec.output_bucket_id
            and volume.get("source_path") == expected_prefix
            and volume.get("container_path") == "/outputs"
            for volume in status.get("spec", {}).get("volumes", [])
        ):
            raise ValueError("Job output mount differs from the artifact namespace")
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ValueError("Expected a relative JSON artifact path")
        raw = storage.get_bytes(f"{expected_prefix}{path.as_posix()}")
        if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Artifact digest differs")
        return json.loads(raw)
