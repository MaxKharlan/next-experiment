from types import SimpleNamespace as NS

from core.connectors.storage import LocalStorage
from scripts.rescore_pairs import find_adapter, rescore, seed_noise


def _adapter(root, name, weights):
    path = root / name
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def test_pairs_are_scored_again_and_shared_weights_once(tmp_path):
    adapters = {
        "t0": _adapter(tmp_path, "t0", b"lr-s0"),
        "t1": _adapter(tmp_path, "t1", b"lr-s1"),
        "r0": _adapter(tmp_path, "r0", b"rank-s0"),
        "r1": _adapter(tmp_path, "r1", b"rank-s1"),
        # the control of both pairs: the failed recipe on the same files
        "c0": _adapter(tmp_path, "c0", b"failed-s0"),
        "c1": _adapter(tmp_path, "c1", b"failed-s1"),
        "d0": _adapter(tmp_path, "d0", b"failed-s0"),
        "d1": _adapter(tmp_path, "d1", b"failed-s1"),
    }
    check = {"t0": 0.45, "t1": 0.41, "r0": 0.20, "r1": 0.26, "c0": 0.12, "c1": 0.10}
    pairs = {
        "lr": {
            "target": {"metrics": [0.5, 0.5], "ft_job_ids": ["t0", "t1"]},
            "control": {"metrics": [0.16, 0.12], "ft_job_ids": ["c0", "c1"]},
        },
        "rank": {
            "target": {"metrics": [0.24, 0.30], "ft_job_ids": ["r0", "r1"]},
            "control": {"metrics": [0.16, 0.12], "ft_job_ids": ["d0", "d1"]},
        },
    }
    scored = []

    def score(adapter):
        scored.append(adapter.name)
        return check[adapter.name]

    new, arms = rescore(pairs, adapters.get, score)
    assert new["lr"]["target"] == {"metrics": [0.45, 0.41], "ft_job_ids": ["t0", "t1"]}
    assert new["rank"]["control"]["metrics"] == [0.12, 0.10]
    assert sorted(scored) == ["c0", "c1", "r0", "r1", "t0", "t1"]  # d0, d1 reused

    # the shared control counts once: three arms with two seeds each
    floor = seed_noise(arms, "check")
    assert (floor.groups, floor.runs) == (3, 6)


def test_arms_with_identical_seeds_show_no_noise():
    arms = [
        {"weights": ["a", "a"], "check": [0.5, 0.5], "valid": [0.5, 0.5]},
        {"weights": ["b", "c"], "check": [0.2, 0.3], "valid": [0.2, 0.2]},
    ]
    floor = seed_noise(arms, "check")
    assert floor.groups == 1 and abs(floor.sd - 0.0707) < 1e-3
    assert seed_noise(arms[:1], "check") is None


def test_adapter_is_on_record_on_disk_or_downloaded(tmp_path):
    storage = LocalStorage(tmp_path)
    on_disk = _adapter(tmp_path, "investigations/x/ckpt-old", b"w")

    class FakeTF:
        def checkpoints(self, job_id):
            return [NS(id=f"ckpt-{job_id}", step_number=75)]

        def download_checkpoint(self, checkpoint, storage, prefix):
            key = f"{prefix}/{checkpoint.id}/adapter_model.safetensors"
            return [storage.put_bytes(key, b"w")]

    record = {"job-1": "bench/dev/c/tf/p/ckpt-job-1"}
    get = lambda job: find_adapter(job, record, storage, FakeTF, "bench/dev/c")
    assert get("job-1") == tmp_path / "bench/dev/c/tf/p/ckpt-job-1"
    assert get("old") == on_disk
    assert get("new") == tmp_path / "bench/dev/c/rescore/ckpt-new"
