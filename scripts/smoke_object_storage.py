"""Check write and read access to the Nebius Object Storage bucket from .env."""

from core.config import get_settings
from core.connectors.storage import S3Storage

s = get_settings()
storage = S3Storage.from_settings(s)

key = storage.put_text("smoke/hello.txt", "ok")
print(f"{s.s3_bucket}/{key} -> {storage.get_text(key)!r}")
print(f"keys under smoke/: {storage.list_keys('smoke/')}")
