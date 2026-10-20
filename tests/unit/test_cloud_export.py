import shutil

import pytest

from jobs.cloud.export import export_tree


def test_object_mount_export_does_not_preserve_posix_attributes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "adapter").mkdir(parents=True)
    (source / "adapter/weights").write_bytes(b"weight bytes")
    (source / "result.json").write_text("{}")

    def unsupported(*args, **kwargs):
        raise PermissionError("Object mount does not implement chmod/utime")

    monkeypatch.setattr(shutil, "copystat", unsupported)
    with pytest.raises(shutil.Error):
        shutil.copytree(source, tmp_path / "old-export")
    export_tree(source, tmp_path / "export")
    assert (tmp_path / "export/adapter/weights").read_bytes() == b"weight bytes"
    assert (tmp_path / "export/result.json").read_text() == "{}"


def test_object_export_rejects_linked_files_before_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "external").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlinks"):
        export_tree(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()
