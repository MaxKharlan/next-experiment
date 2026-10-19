"""Bind fresh evaluations to local weights, tokenizer, template and dependencies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

REQUIRED_FILES = {
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
}
PACKAGES = ("torch", "transformers", "peft", "safetensors", "tokenizers", "jinja2")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EvaluationRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1, le=1)
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot: str
    files_sha256: dict[str, str]
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packages: dict[str, str]

    @field_validator("files_sha256")
    @classmethod
    def safe_files(cls, value):
        if set(value) != REQUIRED_FILES:
            raise ValueError("Require all supported single-file snapshot hashes")
        for name, digest in value.items():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid hash for {name}")
        return value

    @field_validator("packages")
    @classmethod
    def complete_packages(cls, value):
        if set(value) != set(PACKAGES) or not all(value.values()):
            raise ValueError("Require all evaluation dependency versions")
        return value


def freeze_runtime(snapshot, model):
    snapshot = Path(snapshot).absolute()
    config = json.loads((snapshot / "tokenizer_config.json").read_text())
    template = config["chat_template"]
    if not isinstance(template, str):
        raise TypeError("Require a single explicit chat template")
    runtime = EvaluationRuntime(
        model=model,
        revision=snapshot.name,
        snapshot=str(snapshot),
        files_sha256={
            name: file_digest(snapshot / name) for name in sorted(REQUIRED_FILES)
        },
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in PACKAGES},
    )
    verify_runtime(runtime)
    return runtime.model_dump()


def verify_runtime(runtime, *, model=None, template_dir=None):
    spec = EvaluationRuntime.model_validate(runtime)
    if model is not None and model != spec.model:
        raise ValueError("Evaluation runtime model differs from protocol")
    root = Path(spec.snapshot)
    if not root.is_absolute() or root.name != spec.revision or not root.is_dir():
        raise ValueError(
            "Evaluation snapshot must be an existing absolute revision path"
        )
    for name, expected in spec.files_sha256.items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Evaluation snapshot changed: {name}")
    for name, expected in spec.packages.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Evaluation dependency changed: {name}")
    template = json.loads((root / "tokenizer_config.json").read_text())["chat_template"]
    if template_dir is not None:
        override = Path(template_dir) / "chat_template.jinja"
        if override.exists():
            template = override.read_text()
    if (
        not isinstance(template, str)
        or hashlib.sha256(template.encode()).hexdigest() != spec.template_sha256
    ):
        raise ValueError("Evaluation chat template differs from frozen template")
    return spec
