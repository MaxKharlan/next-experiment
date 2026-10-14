import boto3
import pytest
from moto import mock_aws

from core.config import Settings
from core.connectors.storage import LocalStorage, S3Storage, get_storage


@pytest.fixture
def local(tmp_path):
    return LocalStorage(tmp_path)


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield S3Storage("test-bucket", client)


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_round_trip(backend, request):
    storage = request.getfixturevalue(backend)
    assert (
        storage.put_text("inv-1/probes/p1.json", '{"ok": true}')
        == "inv-1/probes/p1.json"
    )
    storage.put_bytes("inv-1/report.md", b"# report")
    storage.put_bytes("inv-2/report.md", b"# other")

    assert storage.get_text("inv-1/probes/p1.json") == '{"ok": true}'
    assert storage.exists("inv-1/report.md")
    assert not storage.exists("inv-1/missing.json")
    assert storage.list_keys("inv-1/") == ["inv-1/probes/p1.json", "inv-1/report.md"]
    with pytest.raises(KeyError):
        storage.get_bytes("inv-1/missing.json")


@pytest.mark.parametrize("key", ["", "/etc/passwd", "../outside.txt", "a/../../b"])
def test_bad_keys_rejected(local, key):
    with pytest.raises(ValueError):
        local.put_bytes(key, b"x")


def test_get_storage_picks_backend(tmp_path):
    s = Settings(_env_file=None, storage_backend="local", artifacts_dir=str(tmp_path))
    assert isinstance(get_storage(s), LocalStorage)

    s = Settings(_env_file=None, storage_backend="s3", s3_bucket="b")
    with pytest.raises(ValueError):
        get_storage(s)
