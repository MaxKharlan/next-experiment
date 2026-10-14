"""Artifact storage: local folder for development, Nebius Object Storage for deploys."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from core.config import Settings, get_settings


def _check_key(key: str) -> str:
    path = PurePosixPath(key)
    if not key or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid storage key {key!r}")
    return key


class Storage(ABC):
    """Keys look like 'investigations/inv-1/probes/p1.json' in both backends."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    def put_text(self, key: str, text: str) -> str:
        return self.put_bytes(key, text.encode("utf-8"))

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")


class LocalStorage(Storage):
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / _check_key(key)

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        keys = (
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file()
        )
        return sorted(k for k in keys if k.startswith(prefix))


class S3Storage(Storage):
    def __init__(self, bucket: str, client: BaseClient):
        self.bucket = bucket
        self.client = client

    @classmethod
    def from_settings(cls, s: Settings) -> S3Storage:
        if not (s.s3_bucket and s.s3_access_key_id and s.s3_secret_access_key):
            raise ValueError(
                "S3_BUCKET, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set"
            )
        client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            region_name=s.s3_region,
            aws_access_key_id=s.s3_access_key_id,
            aws_secret_access_key=s.s3_secret_access_key,
        )
        return cls(s.s3_bucket, client)

    def put_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=_check_key(key), Body=data)
        return key

    def get_bytes(self, key: str) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=_check_key(key))
        except self.client.exceptions.NoSuchKey:
            raise KeyError(key) from None
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=_check_key(key))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise
        return True

    def list_keys(self, prefix: str = "") -> list[str]:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        )
        return sorted(obj["Key"] for page in pages for obj in page.get("Contents", []))


def get_storage(s: Settings | None = None) -> Storage:
    s = s or get_settings()
    if s.storage_backend == "s3":
        return S3Storage.from_settings(s)
    return LocalStorage(s.artifacts_dir)
