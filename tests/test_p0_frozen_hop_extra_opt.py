from pathlib import Path

import pytest

from scripts.diagnostics.p0_frozen_hop_extra_opt import (
    cache_ckpt_path,
    planned_jobs,
    resolve_all,
    resolve_checkpoint,
    tau_step,
)


def test_tau_step_is_t_over_k():
    assert tau_step(10, 4) == 2.5


def test_planned_jobs_are_four_cells_times_three_hops():
    jobs = planned_jobs()
    assert len(jobs) == 12
    assert {job["environment"] for job in jobs} == {
        "hopper-medium-v2",
        "hopper-expert-v2",
    }
    assert {job["hop"] for job in jobs} == {2, 3, 4}


def test_resolve_refuses_emergency_leftover(tmp_path: Path):
    cell = {
        "environment": "hopper-medium-v2",
        "seed": 0,
        "expected_sha256": "abc",
        "inventory_path": str(tmp_path / "missing.pkl"),
    }
    run_dir = cache_ckpt_path(tmp_path, "hopper-medium-v2", 0).parent
    run_dir.mkdir(parents=True)
    (run_dir / "params_3904.pkl").write_bytes(b"not-a-1M")
    with pytest.raises(FileNotFoundError, match="params_3904"):
        resolve_checkpoint(cell, cache_root=tmp_path)


def test_resolve_refuses_hash_mismatch(tmp_path: Path):
    cell = {
        "environment": "hopper-medium-v2",
        "seed": 0,
        "expected_sha256": "0" * 64,
        "inventory_path": str(tmp_path / "missing.pkl"),
    }
    path = cache_ckpt_path(tmp_path, "hopper-medium-v2", 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong-weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_checkpoint(cell, cache_root=tmp_path)


def test_resolve_accepts_matching_1m(tmp_path: Path):
    payload = b"first90-hopper-1M"
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    cell = {
        "environment": "hopper-medium-v2",
        "seed": 0,
        "expected_sha256": digest,
        "inventory_path": str(tmp_path / "missing.pkl"),
    }
    path = cache_ckpt_path(tmp_path, "hopper-medium-v2", 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    assert resolve_checkpoint(cell, cache_root=tmp_path) == path


def test_resolve_all_reports_missing(tmp_path: Path):
    jobs = planned_jobs(tasks=["hopper-medium-v2"], seeds=[0], hops=[2])
    found, missing = resolve_all(jobs, cache_root=tmp_path)
    assert found == {}
    assert len(missing) == 1
    assert missing[0]["environment"] == "hopper-medium-v2"
    assert missing[0]["seed"] == 0
