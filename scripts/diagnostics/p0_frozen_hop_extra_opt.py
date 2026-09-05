"""Frozen-critic extra hop-actor steps on first90 Hopper MART 1M snapshots.

Does not retrain the critic or μ_{k-1}. Does not substitute mpi1/exp3 or the
incomplete local hopper dirs (e.g. emergency params_3904.pkl).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

FIRST90_INVENTORY = Path(
    "sweep_results/diagnostics/p0_hop_actor_audit/INVENTORY.json"
)
DEFAULT_CACHE = Path(
    "sweep_results/diagnostics/p0_frozen_hop_extra_opt/source_ckpts"
)
FROZEN_SOURCE_REVISION = "29fea94bee9d9276ac463caa2b897e3474fe1b2b"
T_VALUE = 10
K = 4
HOPS = (2, 3, 4)
SEEDS = (0, 1)
TASKS = ("hopper-medium-v2", "hopper-expert-v2")
LR = 3e-4
BATCH_SIZE = 256
DATASET_N = 4096
DATASET_SAMPLE_SEED = 20260905
EVAL_SEED_BASE = 10_000
FORBIDDEN_PATH_MARKERS = (
    "results/mpi1",
    "results/mpi2",
    "results/exp3",
    "mpi1_s23",
    "mpi2_s23",
    "exp3_s23",
)

# first90 hop-actor audit hashes (sha256 of params_1000000.pkl).
FIRST90_CELLS: tuple[dict[str, Any], ...] = (
    {
        "environment": "hopper-medium-v2",
        "T": T_VALUE,
        "seed": 0,
        "expected_sha256": (
            "c18b5784bbb5ff656afb5d80ca61d1b5a828853cc6108b710ad009500f1d0f3d"
        ),
        "inventory_path": (
            "/home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94/runs/"
            "bar_p4/hopper-medium-v2_tau10_mpi4_seed0/params_1000000.pkl"
        ),
    },
    {
        "environment": "hopper-medium-v2",
        "T": T_VALUE,
        "seed": 1,
        "expected_sha256": (
            "6aa372aefdb40f3dac09e1f026d781d3bab7e63f4735743c358c1bda69a1acc3"
        ),
        "inventory_path": (
            "/home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94/runs/"
            "bar_p4/hopper-medium-v2_tau10_mpi4_seed1/params_1000000.pkl"
        ),
    },
    {
        "environment": "hopper-expert-v2",
        "T": T_VALUE,
        "seed": 0,
        "expected_sha256": (
            "51f81441d59524dce5138520603a366434d87fa32a2e61090c2a9f51d6dcee11"
        ),
        "inventory_path": (
            "/home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94_cont2_gpu_d923/"
            "runs/bar_p4/hopper-expert-v2_tau10_mpi4_seed0/params_1000000.pkl"
        ),
    },
    {
        "environment": "hopper-expert-v2",
        "T": T_VALUE,
        "seed": 1,
        "expected_sha256": (
            "7fb0cd4a9b7259187baae09587141fe05649f34ca4769f34127c62dbe9245319"
        ),
        "inventory_path": (
            "/home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94_cont2_gpu_d923/"
            "runs/bar_p4/hopper-expert-v2_tau10_mpi4_seed1/params_1000000.pkl"
        ),
    },
)

INCOMPLETE_LOCAL_REFUSAL = (
    "Refusing incomplete local hopper run dirs. Experiment 1 uses the first90 "
    "MART 1M snapshots (params_1000000.pkl) hashed in p0_hop_actor_audit. "
    "Do not substitute params_3904.pkl, mpi1/exp3, or a different T/seed."
)


def tau_step(total_t: float = T_VALUE, hops: int = K) -> float:
    return float(total_t) / float(hops)


def cache_ckpt_path(cache_root: Path, environment: str, seed: int) -> Path:
    return cache_root / f"{environment}_tau{T_VALUE}_mpi4_seed{seed}" / "params_1000000.pkl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden(path: Path) -> bool:
    text = str(path)
    return any(marker in text for marker in FORBIDDEN_PATH_MARKERS)


def planned_jobs(
    tasks: Sequence[str] = TASKS,
    seeds: Sequence[int] = SEEDS,
    hops: Sequence[int] = HOPS,
) -> list[dict[str, Any]]:
    task_set = set(tasks)
    seed_set = set(int(s) for s in seeds)
    hop_set = set(int(h) for h in hops)
    jobs = []
    for cell in FIRST90_CELLS:
        if cell["environment"] not in task_set or int(cell["seed"]) not in seed_set:
            continue
        for hop in HOPS:
            if hop not in hop_set:
                continue
            jobs.append({**cell, "hop": hop, "ref_index": hop - 1, "actor_index": hop})
    return jobs


def copy_instructions(
    missing: Sequence[Mapping[str, Any]],
    cache_root: Path | None = None,
) -> str:
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE
    lines = [
        "First90 Hopper MART 1M checkpoints are not readable on this host.",
        "Copy these files onto ext_csh, keeping the filenames:",
        "",
    ]
    for cell in missing:
        dest = cache_ckpt_path(root, cell["environment"], int(cell["seed"]))
        lines.append(f"  src: {cell['inventory_path']}")
        lines.append(f"  dst: {dest}")
        lines.append(f"  sha256: {cell['expected_sha256']}")
        lines.append("")
    lines.append("Then rerun run_p0_frozen_hop_extra_opt.py. Do not substitute other sweeps.")
    return "\n".join(lines) + "\n"


def resolve_checkpoint(
    cell: Mapping[str, Any],
    *,
    cache_root: Path,
    extra_roots: Sequence[Path] = (),
) -> Path:
    """Return the 1M pickle after hash check. Never accepts a non-1M leftover."""
    expected = str(cell["expected_sha256"])
    candidates = [
        cache_ckpt_path(cache_root, cell["environment"], int(cell["seed"])),
        Path(cell["inventory_path"]),
    ]
    for root in extra_roots:
        candidates.append(
            cache_ckpt_path(Path(root), cell["environment"], int(cell["seed"]))
        )
        candidates.append(
            Path(root)
            / f"{cell['environment']}_tau{T_VALUE}_mpi4_seed{cell['seed']}"
            / "params_1000000.pkl"
        )

    seen: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.append(path)
        if not path.is_file():
            continue
        if path.name != "params_1000000.pkl":
            raise ValueError(INCOMPLETE_LOCAL_REFUSAL + f" got {path.name}")
        if _forbidden(path):
            raise ValueError(
                f"forbidden substitution path {path}. {INCOMPLETE_LOCAL_REFUSAL}"
            )
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"checkpoint hash mismatch for {cell['environment']} seed "
                f"{cell['seed']}: got {digest}, expected {expected} ({path})"
            )
        return path

    nearby = cache_ckpt_path(cache_root, cell["environment"], int(cell["seed"])).parent
    if nearby.is_dir():
        leftovers = sorted(p.name for p in nearby.glob("params_*.pkl"))
        if leftovers and "params_1000000.pkl" not in leftovers:
            raise FileNotFoundError(
                INCOMPLETE_LOCAL_REFUSAL + f" found {leftovers} under {nearby}"
            )
    raise FileNotFoundError(
        f"missing first90 1M checkpoint for {cell['environment']} seed "
        f"{cell['seed']} (expected sha256 {expected})"
    )


def resolve_all(
    jobs: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path,
    extra_roots: Sequence[Path] = (),
) -> tuple[dict[tuple[str, int], Path], list[dict[str, Any]]]:
    found: dict[tuple[str, int], Path] = {}
    missing: list[dict[str, Any]] = []
    unique_cells = {(job["environment"], int(job["seed"])): job for job in jobs}
    for key, cell in unique_cells.items():
        try:
            found[key] = resolve_checkpoint(
                cell, cache_root=cache_root, extra_roots=extra_roots
            )
        except FileNotFoundError:
            missing.append(dict(cell))
    return found, missing
