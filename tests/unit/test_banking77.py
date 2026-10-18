import csv
import hashlib
import io
import json

import pytest

from bench.tasks.banking77 import build, query_key, read_csv, split_rows
from core.case import DatasetSplits


def source(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["text", "category"])
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def pools():
    train = [(f"query {label} {i}", label) for label in ("a", "b") for i in range(8)]
    test = [(f"held out {label}", label) for label in ("a", "b")]
    return train, test


def test_build_is_reproducible_and_manifest_matches_files(tmp_path, pools):
    train, test = map(source, pools)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build(train, test, first, validation_per_label=2, check_per_label=2)
    assert (
        build(train, test, second, validation_per_label=2, check_per_label=2)
        == manifest
    )
    DatasetSplits.model_validate(manifest["datasets"])
    seen = set()
    for name, spec in manifest["datasets"].items():
        raw = (first / spec["ref"]).read_bytes()
        assert raw == (second / spec["ref"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
        records = [json.loads(line) for line in raw.splitlines()]
        assert len(records) == spec["examples"]
        queries = {query_key(r["messages"][1]["content"]) for r in records}
        assert not seen & queries
        seen.update(queries)
        assert {r["messages"][2]["content"] for r in records} == {"a", "b"}
        assert all(
            [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
            for r in records
        )
        if name == "test":
            assert queries == {"held out a", "held out b"}
    assert json.loads((first / "manifest.json").read_text()) == manifest


def test_duplicates_do_not_leak_from_test_into_training(pools):
    train, test = pools
    train += [(" HELD   OUT A ", "a"), ("QUERY a 0", "a")]
    splits, dropped = split_rows(train, test, 0, 2, 2)
    assert dropped == {"train": 2, "test": 0}
    assert len(splits["train"]) == 8


def test_conflicting_labels_fail(pools):
    train, test = pools
    with pytest.raises(ValueError, match="conflicting labels"):
        split_rows(train + [("held out a", "b")], test, 0, 2, 2)


def test_seed_changes_partition_but_not_held_out_membership(pools):
    first, _ = split_rows(*pools, 0, 2, 2)
    second, _ = split_rows(*pools, 1, 2, 2)
    assert set(first["train"]) != set(second["train"])
    assert set(first["test"]) == set(second["test"])


@pytest.mark.parametrize("validation,check", [(0, 1), (1, -1), (4, 4)])
def test_invalid_split_sizes_fail(pools, validation, check):
    with pytest.raises(ValueError):
        split_rows(*pools, 0, validation, check)


@pytest.mark.parametrize(
    "data", [b"text,label\na,b\n", b"text,category\n", b"text,category\na,\n"]
)
def test_malformed_source_fails(data):
    with pytest.raises(ValueError):
        read_csv(data)


def test_existing_output_is_not_overwritten(tmp_path, pools):
    marker = tmp_path / "train.jsonl"
    marker.write_text("existing data")
    with pytest.raises(FileExistsError):
        build(*map(source, pools), tmp_path, validation_per_label=2, check_per_label=2)
    assert marker.read_text() == "existing data"
