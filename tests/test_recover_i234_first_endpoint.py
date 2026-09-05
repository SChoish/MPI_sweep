from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.diagnostics.recover_i234_first_endpoint import (
    inventory_historical_i4,
    recover_grid,
)


def _write_run(root: Path, env: str, tau: str, tag: str, seed: int, k: int, first: float, deploy: float) -> None:
    run = root / f"{env}_tau{tau}_{tag}_seed{seed}"
    run.mkdir(parents=True)
    (run / "params_1000000.pkl").write_bytes(b"stub")
    with (run / "eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "d4rl_score", f"d4rl_pi{k}"],
        )
        writer.writeheader()
        writer.writerow(
            {"step": "1000000", "d4rl_score": str(first), f"d4rl_pi{k}": str(deploy)}
        )


def test_recover_grid_keeps_requested_seeds_only(tmp_path: Path) -> None:
    env = "hopper-medium-v2"
    _write_run(tmp_path, env, "4", "mpi2", 0, 2, 10.0, 12.0)
    _write_run(tmp_path, env, "4", "mpi2", 1, 2, 11.0, 13.0)
    _write_run(tmp_path, env, "4", "mpi2", 2, 2, 99.0, 100.0)
    rows = recover_grid(
        "historical_score_grid_s01",
        2,
        "implicit",
        tmp_path,
        "mpi2",
        seeds=(0, 1),
        environments=(env,),
        taus=(4.0,),
    )
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {"0", "1"}
    assert all(row["family"] == "historical_score_grid_s01" for row in rows)
    assert rows[0]["delta_endpoint_minus_first"] == "2.0"


def test_recover_grid_requires_complete_grid(tmp_path: Path) -> None:
    env = "hopper-medium-v2"
    _write_run(tmp_path, env, "4", "mpi2", 0, 2, 10.0, 12.0)
    with pytest.raises(ValueError, match="missing"):
        recover_grid(
            "historical_score_grid_s01",
            2,
            "implicit",
            tmp_path,
            "mpi2",
            seeds=(0, 1),
            environments=(env,),
            taus=(4.0,),
        )


def test_historical_i4_inventory_does_not_invent_empty_first(tmp_path: Path) -> None:
    env = "hopper-medium-v2"
    run = tmp_path / f"{env}_tau4_mpi4_seed0"
    run.mkdir()
    (run / "params_1000000.pkl").write_bytes(b"stub")
    with (run / "eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "d4rl_score", "d4rl_pi4"])
        writer.writeheader()
        writer.writerow({"step": "1000000", "d4rl_score": "", "d4rl_pi4": "80.0"})
    cov = inventory_historical_i4(
        tmp_path,
        environments=(env,),
        taus=(4.0,),
        seeds=(0,),
    )
    assert cov["n_expected"] == 1
    assert cov["n_endpoint"] == 1
    assert cov["n_first_actor"] == 0
    assert cov["n_missing_first_actor"] == 1
    assert cov["first_actor_present"] == []
