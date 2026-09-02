#!/usr/bin/env python3
"""Development-only P2 v4 ReLU activation-region residence pilot.

This script deliberately does not estimate a learned-critic convergence slope.
It writes a design lock before opening the learned checkpoint, validates the
numerics on synthetic cases, and then measures exact residence of one exposed
ReLU critic's affine action regions on 64 new dataset anchors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

PROTOCOL = "p2_relu_residence_v1"
ENVIRONMENT = "halfcheetah-medium-v2"
SEED = 0
CHECKPOINT_STEP = 1_000_000
PILOT_TAU = 1.0
N_PILOT = 64
PILOT_SAMPLE_SEED = 20260902
FINAL_SAMPLE_SEED_BASE = 20260903
TIMES = (0.025, 0.05, 0.1, 0.2)
SUBSTEPS = (1, 2, 4, 8, 16)
EPS = 1e-6
BOX = 1.0
ACTION_ELIGIBILITY = 0.95
ROUNDING_SAFETY = 32.0
TIME_ATOL = 1e-12
TIME_RTOL = 1e-9
FD_DELTAS = tuple(2.0**-power for power in range(10, 31))
FINAL_ENVIRONMENTS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
PINNED_INPUTS = {
    "checkpoint": {
        "semantic_id": "halfcheetah-medium-v2_tau1_seed0_step1000000_qlearning_checkpoint",
        "basename": "params_1000000.pkl",
        "sha256": "cafaabea01bfb25af09e6b21f64d7953cc7aa76c6d4e486cd9075db239445ab1",
    },
    "config": {
        "semantic_id": "halfcheetah-medium-v2_tau1_seed0_training_config",
        "basename": "config.json",
        "sha256": "de316e5ab27bc2c7b09ef5ec12a93bcdfe9ccbffedf9bbeaf1a6ec46c8c923c9",
    },
    "dataset": {
        "semantic_id": "d4rl_halfcheetah_medium_v2_hdf5",
        "basename": "halfcheetah_medium-v2.hdf5",
        "sha256": "41324e8487a9556d8cfa4538827f8937f4d63dae3030825ebc9f3a726ed08ca0",
    },
}
PINNED_EXCLUSIONS = (
    {
        "semantic_id": "archived_fixed_operator_halfcheetah_medium_v2_seed0_indices",
        "basename": "halfcheetah-medium-v2_seed0.npy",
        "sha256": "ddc575b77e76797d7b021e8f122cb1345747c277b66f8a64e66ac334b1d397d0",
        "n_indices": 512,
    },
    {
        "semantic_id": "archived_fixed_operator_halfcheetah_medium_v2_seed1_indices",
        "basename": "halfcheetah-medium-v2_seed1.npy",
        "sha256": "8e80d56920b1d4c1360ac408863024677a0027457361c4dacd6b3485f16414ae",
        "n_indices": 512,
    },
)
PINNED_EXCLUSION_UNION = 1022
SOURCE_SNAPSHOT_DIR = "SOURCE_SNAPSHOT"
DEFAULT_CHECKPOINT = Path(
    "/home/ext_csv/mpi_sweep_lab/results_qnorm/"
    "halfcheetah-medium-v2_tau1_seed0/params_1000000.pkl"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.json")
DEFAULT_DATASET = Path("/raid/ext_csv/datasets/d4rl/halfcheetah_medium-v2.hdf5")
DEFAULT_OLD_INDICES = (
    Path(
        "sweep_results/diagnostics/fixed_operator_order/"
        "state_indices/halfcheetah-medium-v2_seed0.npy"
    ),
    Path(
        "sweep_results/diagnostics/fixed_operator_order/"
        "state_indices/halfcheetah-medium-v2_seed1.npy"
    ),
)
ROOT = Path(__file__).resolve().parents[2]

# --- Final scientific audit constants -------------------------------------
FINAL_PROTOCOL = "p2_relu_residence_final_v2"
FINAL_RAW_STATUS = "analysis_complete_pending_verification"
FINAL_N_STATES = 512
FINAL_SEEDS = (0, 1)
FINAL_TAU = 1.0
FINAL_EXCLUDED_FAMILY = "halfcheetah"
KNOWN_DEV_BUNDLE_IDS = (
    "VnOjA4",
    "InTWgf",
    "OLDBdx",
    "xlMNfc",
    "sqfIVj",
    "meAdJI",
)
DEFAULT_RESULTS_ROOT = Path("/home/ext_csv/mpi_sweep_lab/results_qnorm")
DEFAULT_DATASET_DIR = Path("/raid/ext_csv/datasets/d4rl")
DEFAULT_ARCHIVED_INDEX_DIR = (
    ROOT / "sweep_results/diagnostics/fixed_operator_order/state_indices"
)
DEFAULT_DEV_BUNDLE_GLOB = "/home/ext_csv/mpi_sweep_lab/p2_relu_residence_pilot.*"
FINAL_REQUIRED_FIXED_CONFIG = {
    "eval_freq": 50_000,
    "eval_episodes": 10,
    "max_timesteps": 1_000_000,
    "batch_size": 256,
    "discount": 0.99,
    "polyak": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "lr": 0.0003,
    "normalize": True,
    "n_jitted_updates": 8,
}

FIRST_NONE = 0
FIRST_RELU = 1
FIRST_BOX = 2
FIRST_TIE = 3
FIRST_LABELS = {
    FIRST_NONE: "none",
    FIRST_RELU: "relu",
    FIRST_BOX: "box",
    FIRST_TIE: "tie",
}
CATEGORIES = (
    "resident",
    "relu_cross",
    "box_cross",
    "tie_cross",
    "boundary_ambiguous",
    "boundary_touch",
    "nonfinite",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("design", "harness", "pilot", "all", "final"), default="all"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--exclude-indices",
        type=Path,
        nargs="+",
        default=[ROOT / path for path in DEFAULT_OLD_INDICES],
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--archived-index-dir", type=Path, default=DEFAULT_ARCHIVED_INDEX_DIR
    )
    parser.add_argument("--dev-bundle-glob", type=str, default=DEFAULT_DEV_BUNDLE_GLOB)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(array.view(np.uint8).tobytes())


def write_json_create(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_source_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(__file__).with_name("verify_p2_relu_residence.py").resolve(),
    )


def final_source_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(__file__).with_name("verify_p2_relu_residence.py").resolve(),
        Path(__file__).with_name("verify_p2_relu_residence_final.py").resolve(),
    )


def source_hashes(sources: tuple[Path, ...] | None = None) -> dict[str, str]:
    files = default_source_files() if sources is None else tuple(
        Path(source).resolve() for source in sources
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required source files missing: {missing}")
    return {path.name: sha256_file(path) for path in files}


def design_document() -> dict[str, Any]:
    design: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "design_locked_before_learned_pilot",
        "written_at": now_iso(),
        "estimand": {
            "learned_primary": "relu_activation_region_residence",
            "direction": "v = action_dim * grad_a Q1(s,a0) / C_ref",
            "C_ref": "mean_i(abs(Q1(s_i,a0_i))) + 1e-6",
            "full_residence": "T is strictly before first ReLU/action-box exit",
            "first_step_residence": "T/K is strictly before first exit",
            "unique_horizon_estimand": "empirical survival curve P(t_exit > t)",
            "cell_dependence": (
                "repeated (T,K) cells with the same horizon t=T or t=T/K are "
                "deterministic views, never independent evidence"
            ),
            "boundary_rows": "retained scope diagnostics, never numerical errors",
            "implicit_scope": "same-region local backward-Euler branch only",
            "learned_slope": None,
            "learned_support_gate": None,
            "global_proximal_claim": False,
        },
        "grid": {"T": list(TIMES), "K": list(SUBSTEPS)},
        "numeric_contract": {
            "precision": "float64",
            "action_box": [-BOX, BOX],
            "action_eligibility": ACTION_ELIGIBILITY,
            "layer1_rounding_radius": (
                "32 * gamma_n * max(1, sum(abs(products)) + abs(bias)); "
                "gamma_n=n*eps64/(1-n*eps64)"
            ),
            "layer2_rounding_radius": (
                "local layer-2 rounding radius + layer1_radius @ abs(W2)"
            ),
            "time_tolerance": "max(1e-12, 1e-9 * max(1, abs(t)))",
            "categories": list(CATEGORIES),
            "denominator_rule": "all sampled anchors retained in exactly one category",
        },
        "harness": {
            "smooth_quadratic": (
                "explicit and exact backward Euler versus exact matrix-exponential flow; "
                "K={4,8,16} slopes in [0.9,1.1]"
            ),
            "relu_cases": [
                "resident",
                "layer1_exit",
                "layer2_exit",
                "box_exit",
                "simultaneous_exit",
                "anchor_boundary",
            ],
        },
        "pilot_contract": {
            "development_only": True,
            "scientific_admissible": False,
            "environment": ENVIRONMENT,
            "seed": SEED,
            "tau": PILOT_TAU,
            "checkpoint_step": CHECKPOINT_STEP,
            "n_states": N_PILOT,
            "sample_seed": PILOT_SAMPLE_SEED,
            "new_indices": True,
            "exclude_archived_indices": True,
            "coverage_gate": None,
        },
        "pilot_input_contract": {
            "bound_by_input_lock_before_semantic_read": True,
            "inputs": PINNED_INPUTS,
            "exclusions": list(PINNED_EXCLUSIONS),
            "exclusion_union_n_indices": PINNED_EXCLUSION_UNION,
            "reject_missing_extra_or_empty_exclusion_inventory": True,
        },
        "final_contract_not_executed": {
            "environments": list(FINAL_ENVIRONMENTS),
            "excluded_family": "halfcheetah",
            "seeds": [0, 1],
            "n_runs": 12,
            "n_states_per_run": 512,
            "sample_seed_base": FINAL_SAMPLE_SEED_BASE,
            "indices_created_only_after_final_protocol_freeze": True,
            "exclude_all_archived_and_pilot_indices": True,
            "exclude_every_development_attempt_index_set": True,
            "development_attempts_include_failed_and_superseded_bundles": True,
            "primary_summary": "unique-horizon empirical survival curve P(t_exit > t)",
            "repeated_cell_policy": "collapse equal horizons; do not count as independent evidence",
            "coverage_gate": None,
        },
        "expected_pilot_counts": {
            "state_geometry_rows": N_PILOT,
            "residence_cells": len(TIMES) * len(SUBSTEPS),
        },
        "source_sha256": source_hashes(),
    }
    design["design_sha256"] = sha256_bytes(canonical_bytes(design))
    return design


def write_design_lock(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"design phase requires an empty directory: {out_dir}")
    design = design_document()
    write_json_create(out_dir / "DESIGN_LOCK.json", design)
    return design


def write_source_snapshots(
    out_dir: Path, sources: tuple[Path, ...] | None = None
) -> dict[str, dict[str, str]]:
    snapshot_dir = out_dir / SOURCE_SNAPSHOT_DIR
    snapshot_dir.mkdir(exist_ok=False)
    records: dict[str, dict[str, str]] = {}
    for source in (
        default_source_files() if sources is None
        else tuple(Path(item).resolve() for item in sources)
    ):
        target = snapshot_dir / source.name
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        digest = sha256_file(target)
        if digest != sha256_file(source):
            raise AssertionError(f"source snapshot mismatch: {source.name}")
        records[source.name] = {
            "path": f"{SOURCE_SNAPSHOT_DIR}/{source.name}",
            "sha256": digest,
        }
    return records


def git_provenance() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    origin_main = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_revision": head,
        "git_dirty": bool(porcelain),
        "git_tracked_dirty": bool(tracked_porcelain),
        "origin_main_revision": origin_main,
        "head_matches_origin_main": head == origin_main,
        "git_status_porcelain_sha256": sha256_bytes(porcelain.encode("utf-8")),
        "git_tracked_status_porcelain_sha256": sha256_bytes(
            tracked_porcelain.encode("utf-8")
        ),
        "source_snapshot_is_durable_provenance": True,
    }


def require_clean_origin_main(provenance: dict[str, Any]) -> None:
    if provenance.get("git_dirty") is not False:
        raise RuntimeError("final audit requires a fully clean worktree")
    if provenance.get("git_tracked_dirty") is not False:
        raise RuntimeError("final audit requires a clean tracked worktree")
    if provenance.get("head_matches_origin_main") is not True:
        raise RuntimeError("final audit requires HEAD to equal local origin/main")
    if provenance.get("git_revision") != provenance.get("origin_main_revision"):
        raise RuntimeError("final audit revision/origin-main provenance mismatch")


def write_input_lock(
    out_dir: Path,
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    exclusion_paths: list[Path],
) -> dict[str, Any]:
    """Hash and lock exact pilot inputs before any pickle/HDF5/NumPy parsing."""
    actual_inputs: dict[str, dict[str, str]] = {}
    for key, path in (
        ("checkpoint", checkpoint),
        ("config", config_path),
        ("dataset", dataset),
    ):
        resolved = path.resolve()
        spec = PINNED_INPUTS[key]
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved.name != spec["basename"]:
            raise ValueError(f"{key} basename {resolved.name!r} is not pinned")
        digest = sha256_file(resolved)
        if digest != spec["sha256"]:
            raise ValueError(f"{key} content hash is not the pinned exposed input")
        actual_inputs[key] = dict(spec)

    expected_exclusions = {row["basename"]: row for row in PINNED_EXCLUSIONS}
    if len(exclusion_paths) != len(expected_exclusions):
        raise ValueError("exclusion inventory must contain exactly seed0 and seed1")
    actual_exclusions: dict[str, dict[str, Any]] = {}
    for path in exclusion_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved.name in actual_exclusions or resolved.name not in expected_exclusions:
            raise ValueError(f"unexpected or duplicate exclusion: {resolved.name}")
        spec = expected_exclusions[resolved.name]
        digest = sha256_file(resolved)
        if digest != spec["sha256"]:
            raise ValueError(f"exclusion content hash mismatch: {resolved.name}")
        actual_exclusions[resolved.name] = dict(spec)
    if set(actual_exclusions) != set(expected_exclusions):
        raise ValueError("canonical exclusion inventory is incomplete")

    document: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "inputs_locked_before_semantic_read",
        "development_only": True,
        "scientific_admissible": False,
        "inputs": actual_inputs,
        "exclusions": [actual_exclusions[name] for name in sorted(actual_exclusions)],
        "exclusion_union_n_indices": PINNED_EXCLUSION_UNION,
        "written_at": now_iso(),
    }
    document["input_lock_sha256"] = sha256_bytes(canonical_bytes(document))
    write_json_create(out_dir / "INPUT_LOCK.json", document)
    return document


def gamma_n(n_terms: int) -> float:
    eps = np.finfo(np.float64).eps
    product = float(n_terms) * eps
    return product / (1.0 - product)


def affine_tol(abs_sum: np.ndarray, n_terms: int) -> np.ndarray:
    return ROUNDING_SAFETY * gamma_n(n_terms) * np.maximum(1.0, abs_sum)


def time_tol(value: float) -> float:
    return max(TIME_ATOL, TIME_RTOL * max(1.0, abs(float(value))))


def q1_forward(
    params: dict[str, np.ndarray], states: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.concatenate((states, actions), axis=-1, dtype=np.float64)
    z1 = inputs @ params["w1"] + params["b1"]
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ params["w2"] + params["b2"]
    h2 = np.maximum(z2, 0.0)
    q = (h2 @ params["w3"] + params["b3"]).reshape(-1)
    return q, z1, z2


def analyze_geometry(
    params: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    c_ref: float,
) -> dict[str, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    q, z1, z2 = q1_forward(params, states, actions)
    inputs = np.concatenate((states, actions), axis=-1)
    m1 = z1 > 0.0
    m2 = z2 > 0.0
    w3 = params["w3"].reshape(-1)
    back2 = m2 * w3[None, :]
    back1 = (back2 @ params["w2"].T) * m1
    action_dim = actions.shape[-1]
    state_dim = states.shape[-1]
    w1_action = params["w1"][state_dim:, :]
    grad = back1 @ w1_action.T
    velocity = (float(action_dim) * grad) / float(c_ref)
    dz1 = velocity @ w1_action
    dz2 = (dz1 * m1) @ params["w2"]

    z1_abs_sum = np.abs(inputs) @ np.abs(params["w1"]) + np.abs(params["b1"])
    h1 = np.maximum(z1, 0.0)
    z2_abs_sum = np.abs(h1) @ np.abs(params["w2"]) + np.abs(params["b2"])
    tol1 = affine_tol(z1_abs_sum, inputs.shape[-1] + 1)
    tol2_local = affine_tol(z2_abs_sum, h1.shape[-1] + 1)
    # ReLU is 1-Lipschitz, so layer-1 forward error propagates through |W2|.
    propagated = tol1 @ np.abs(params["w2"])
    tol2 = tol2_local + (1.0 + ROUNDING_SAFETY * gamma_n(h1.shape[-1] + 1)) * propagated

    n_states = len(actions)
    relu_time = np.full(n_states, np.inf, dtype=np.float64)
    box_time = np.full(n_states, np.inf, dtype=np.float64)
    first_time = np.full(n_states, np.inf, dtype=np.float64)
    first_code = np.full(n_states, FIRST_NONE, dtype=np.int8)
    first_layer = np.full(n_states, -1, dtype=np.int16)
    first_unit = np.full(n_states, -1, dtype=np.int16)
    boundary_ambiguous = np.zeros(n_states, dtype=bool)
    nonfinite = np.zeros(n_states, dtype=bool)
    min_scaled_margin = np.full(n_states, np.inf, dtype=np.float64)

    for row in range(n_states):
        finite = all(
            np.all(np.isfinite(value[row]))
            for value in (actions, q[:, None], z1, z2, grad, velocity, dz1, dz2)
        ) and math.isfinite(c_ref)
        if not finite:
            nonfinite[row] = True
            continue
        scaled = np.concatenate(
            (np.abs(z1[row]) / tol1[row], np.abs(z2[row]) / tol2[row])
        )
        min_scaled_margin[row] = float(np.min(scaled))
        near1 = np.abs(z1[row]) <= tol1[row]
        near2 = np.abs(z2[row]) <= tol2[row]
        if bool(np.any(near1) or np.any(near2)):
            boundary_ambiguous[row] = True

        candidates: list[tuple[float, int, int]] = []
        for layer, values, derivs, tolerances in (
            (1, z1[row], dz1[row], tol1[row]),
            (2, z2[row], dz2[row], tol2[row]),
        ):
            for unit, (value, deriv, tolerance) in enumerate(
                zip(values, derivs, tolerances, strict=True)
            ):
                if abs(float(value)) <= float(tolerance) or deriv == 0.0:
                    continue
                crosses = (value > 0.0 and deriv < 0.0) or (
                    value < 0.0 and deriv > 0.0
                )
                if crosses:
                    hit = -float(value) / float(deriv)
                    if hit > 0.0 and math.isfinite(hit):
                        candidates.append((hit, layer, unit))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            relu_time[row] = candidates[0][0]

        box_candidates: list[float] = []
        for action, direction in zip(actions[row], velocity[row], strict=True):
            if direction > 0.0:
                hit = (BOX - float(action)) / float(direction)
            elif direction < 0.0:
                hit = (-BOX - float(action)) / float(direction)
            else:
                continue
            if hit > 0.0 and math.isfinite(hit):
                box_candidates.append(hit)
        if box_candidates:
            box_time[row] = min(box_candidates)

        first = min(relu_time[row], box_time[row])
        first_time[row] = first
        if not math.isfinite(first):
            continue
        tolerance = time_tol(first)
        relu_hits = [item for item in candidates if abs(item[0] - first) <= tolerance]
        box_hits = sum(abs(item - first) <= tolerance for item in box_candidates)
        if len(relu_hits) + box_hits > 1:
            first_code[row] = FIRST_TIE
        elif relu_hits:
            first_code[row] = FIRST_RELU
        else:
            first_code[row] = FIRST_BOX
        if relu_hits:
            first_layer[row] = relu_hits[0][1]
            first_unit[row] = relu_hits[0][2]

    return {
        "q_anchor": q,
        "z1": z1,
        "z2": z2,
        "pattern1": m1,
        "pattern2": m2,
        "grad": grad,
        "velocity": velocity,
        "relu_hit_time": relu_time,
        "box_hit_time": box_time,
        "first_hit_time": first_time,
        "first_hit_code": first_code,
        "first_layer": first_layer,
        "first_unit": first_unit,
        "boundary_ambiguous": boundary_ambiguous,
        "nonfinite": nonfinite,
        "anchor_min_scaled_margin": min_scaled_margin,
    }


def classify_horizon(geometry: dict[str, np.ndarray], row: int, horizon: float) -> str:
    if bool(geometry["nonfinite"][row]):
        return "nonfinite"
    if bool(geometry["boundary_ambiguous"][row]):
        return "boundary_ambiguous"
    hit = float(geometry["first_hit_time"][row])
    if not math.isfinite(hit) or horizon < hit - time_tol(hit):
        return "resident"
    if abs(horizon - hit) <= time_tol(hit):
        return "boundary_touch"
    code = int(geometry["first_hit_code"][row])
    return {
        FIRST_RELU: "relu_cross",
        FIRST_BOX: "box_cross",
        FIRST_TIE: "tie_cross",
    }.get(code, "nonfinite")


def explicit_quadratic(
    a0: np.ndarray, diag_a: np.ndarray, b: np.ndarray, total: float, k: int
) -> np.ndarray:
    action = np.asarray(a0, dtype=np.float64).copy()
    step = total / k
    for _ in range(k):
        action = action + step * 2.0 * (diag_a * action + b)
    return action


def implicit_quadratic(
    a0: np.ndarray, diag_a: np.ndarray, b: np.ndarray, total: float, k: int
) -> np.ndarray:
    action = np.asarray(a0, dtype=np.float64).copy()
    step = total / k
    for _ in range(k):
        action = (action + 2.0 * step * b) / (1.0 - 2.0 * step * diag_a)
    return action


def exact_quadratic_flow(
    a0: np.ndarray, diag_a: np.ndarray, b: np.ndarray, total: float
) -> np.ndarray:
    rate = 2.0 * diag_a
    exponent = np.exp(rate * total)
    return exponent * a0 + ((exponent - 1.0) / rate) * (2.0 * b)


def fit_slope(errors: list[float], ks: tuple[int, ...], total: float) -> float:
    return float(
        np.polyfit(np.log([total / k for k in ks]), np.log(errors), 1)[0]
    )


def synthetic_params(case: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if case == "layer1_exit":
        w1 = np.array([[1.0, 0.0]])
        b1 = np.array([0.1, 1.0])
        w2 = np.eye(2)
        b2 = np.array([1.0, 1.0])
        w3 = np.array([[-1.0], [0.0]])
        action = np.array([[0.0]])
    elif case == "layer2_exit":
        w1 = np.array([[1.0, 0.0]])
        b1 = np.array([2.0, 1.0])
        w2 = np.eye(2)
        b2 = np.array([-1.9, 1.0])
        w3 = np.array([[-1.0], [0.0]])
        action = np.array([[0.0]])
    elif case == "box_exit":
        w1 = np.array([[1.0, 0.0]])
        b1 = np.array([2.0, 1.0])
        w2 = np.eye(2)
        b2 = np.array([1.0, 1.0])
        w3 = np.array([[1.0], [0.0]])
        action = np.array([[0.9]])
    elif case == "simultaneous_exit":
        w1 = np.array([[1.0, 1.0]])
        b1 = np.array([0.1, 0.1])
        w2 = np.eye(2)
        b2 = np.array([1.0, 1.0])
        w3 = np.array([[-0.5], [-0.5]])
        action = np.array([[0.0]])
    elif case == "anchor_boundary":
        w1 = np.array([[1.0, 0.0]])
        b1 = np.array([0.0, 1.0])
        w2 = np.eye(2)
        b2 = np.array([1.0, 1.0])
        w3 = np.array([[-1.0], [0.0]])
        action = np.array([[0.0]])
    else:
        raise KeyError(case)
    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": np.array([0.0]),
    }, action


def run_harness(out_dir: Path, gate_filename: str = "DESIGN_LOCK.json") -> dict[str, Any]:
    if not (out_dir / gate_filename).is_file():
        raise FileNotFoundError(f"{gate_filename} must exist before harness")
    a0 = np.array([0.2, -0.3], dtype=np.float64)
    diag_a = np.array([-0.25, -0.5], dtype=np.float64)
    b = np.array([0.1, 0.05], dtype=np.float64)
    total = 0.2
    exact = exact_quadratic_flow(a0, diag_a, b, total)
    ks = (4, 8, 16)
    smooth: dict[str, Any] = {"exact": exact.tolist(), "rows": []}
    slopes: dict[str, float] = {}
    for name, solver in (
        ("explicit", explicit_quadratic),
        ("implicit", implicit_quadratic),
    ):
        errors = []
        for k in ks:
            endpoint = solver(a0, diag_a, b, total, k)
            error = float(np.sqrt(np.mean(np.square(endpoint - exact))))
            errors.append(error)
            smooth["rows"].append({"scheme": name, "K": k, "error": error})
        slopes[name] = fit_slope(errors, ks, total)
    smooth["slopes_K4_8_16"] = slopes
    smooth["pass"] = all(0.9 <= value <= 1.1 for value in slopes.values())

    expected = {
        "layer1_exit": (0.1, FIRST_RELU, 1),
        "layer2_exit": (0.1, FIRST_RELU, 2),
        "box_exit": (0.1, FIRST_BOX, -1),
        "simultaneous_exit": (0.1, FIRST_TIE, 1),
        "anchor_boundary": (math.inf, FIRST_NONE, -1),
    }
    cases: dict[str, Any] = {}
    for case, (expected_time, expected_code, expected_layer) in expected.items():
        params, action = synthetic_params(case)
        geometry = analyze_geometry(params, np.zeros((1, 0)), action, 1.0)
        actual_time = float(geometry["first_hit_time"][0])
        boundary = bool(geometry["boundary_ambiguous"][0])
        case_pass = (
            (math.isinf(expected_time) and math.isinf(actual_time))
            or abs(actual_time - expected_time) <= 1e-12
        ) and int(geometry["first_hit_code"][0]) == expected_code
        if expected_layer >= 0:
            case_pass &= int(geometry["first_layer"][0]) == expected_layer
        if case == "anchor_boundary":
            case_pass &= boundary
        cases[case] = {
            "first_hit_time": None if math.isinf(actual_time) else actual_time,
            "first_hit_type": FIRST_LABELS[int(geometry["first_hit_code"][0])],
            "first_layer": int(geometry["first_layer"][0]),
            "boundary_ambiguous": boundary,
            "pass": bool(case_pass),
        }
    params, action = synthetic_params("layer1_exit")
    resident_geometry = analyze_geometry(params, np.zeros((1, 0)), action, 1.0)
    cases["resident"] = {
        "horizon": 0.05,
        "category": classify_horizon(resident_geometry, 0, 0.05),
        "pass": classify_horizon(resident_geometry, 0, 0.05) == "resident",
    }
    report = {
        "protocol": PROTOCOL,
        "smooth_quadratic": smooth,
        "synthetic_relu": cases,
        "pass": bool(smooth["pass"] and all(row["pass"] for row in cases.values())),
        "written_at": now_iso(),
    }
    if not report["pass"]:
        raise AssertionError(f"harness failed: {report}")
    write_json_create(out_dir / "HARNESS.json", report)
    return report


def load_qlearning_rows(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(dataset, "r") as handle:
        observations = np.array(handle["observations"], dtype=np.float32)
        actions = np.array(handle["actions"], dtype=np.float32)
        if "timeouts" not in handle:
            raise ValueError("pilot requires the stored timeout mask")
        timeouts = np.array(handle["timeouts"], dtype=np.float32)
    keep = ~timeouts[:-1].astype(bool)
    filtered_observations = observations[:-1][keep]
    filtered_actions = actions[:-1][keep]
    mean = filtered_observations.mean(axis=0)
    std = filtered_observations.std(axis=0) + np.float32(1e-3)
    return filtered_observations, filtered_actions, mean, std


def extract_q1(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    tree = payload["critic_params"]
    if isinstance(tree, dict) and set(tree) == {"params"}:
        tree = tree["params"]
    expected = ("q1_l0", "q1_l1", "q1_out")
    if not all(key in tree for key in expected):
        raise KeyError(f"unexpected Q1 tree: {tree.keys()}")
    return {
        "w1": np.asarray(tree["q1_l0"]["kernel"], dtype=np.float64),
        "b1": np.asarray(tree["q1_l0"]["bias"], dtype=np.float64),
        "w2": np.asarray(tree["q1_l1"]["kernel"], dtype=np.float64),
        "b2": np.asarray(tree["q1_l1"]["bias"], dtype=np.float64),
        "w3": np.asarray(tree["q1_out"]["kernel"], dtype=np.float64),
        "b3": np.asarray(tree["q1_out"]["bias"], dtype=np.float64),
    }


def jax_gradients(
    params: dict[str, np.ndarray], states: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    p = {key: jnp.asarray(value, dtype=jnp.float64) for key, value in params.items()}

    def one_q(state: jax.Array, action: jax.Array) -> jax.Array:
        inputs = jnp.concatenate((state, action))
        h1 = jax.nn.relu(inputs @ p["w1"] + p["b1"])
        h2 = jax.nn.relu(h1 @ p["w2"] + p["b2"])
        return jnp.squeeze(h2 @ p["w3"] + p["b3"])

    fn = jax.jit(jax.vmap(jax.grad(one_q, argnums=1)))
    return np.asarray(fn(jnp.asarray(states), jnp.asarray(actions)), dtype=np.float64)


def select_pilot_indices(
    actions: np.ndarray, exclusion_paths: list[Path]
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    excluded: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in exclusion_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"required archived index file missing: {resolved}")
        values = np.asarray(np.load(resolved), dtype=np.int64)
        excluded.update(int(value) for value in values)
        records.append(
            {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "n_indices": int(len(values)),
            }
        )
    eligible = np.flatnonzero(np.max(np.abs(actions), axis=-1) <= ACTION_ELIGIBILITY)
    eligible = np.asarray(
        [value for value in eligible if int(value) not in excluded], dtype=np.int64
    )
    if len(eligible) < N_PILOT:
        raise RuntimeError(f"only {len(eligible)} eligible new indices")
    rng = np.random.default_rng(PILOT_SAMPLE_SEED)
    selected = rng.choice(eligible, size=N_PILOT, replace=False).astype(np.int64)
    if excluded.intersection(int(value) for value in selected):
        raise AssertionError("pilot index exclusion failed")
    return selected, records, len(excluded)


def validate_pilot_inputs(
    checkpoint: Path, config_path: Path, dataset: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (checkpoint, config_path, dataset):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "env": ENVIRONMENT,
        "seed": SEED,
        "tau": PILOT_TAU,
        "normalize": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key}={config.get(key)!r}, expected {expected!r}")
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local research checkpoint
    if int(payload.get("step", -1)) != CHECKPOINT_STEP:
        raise ValueError(f"checkpoint step is {payload.get('step')}")
    if float(payload.get("max_action", math.nan)) != BOX:
        raise ValueError(f"checkpoint max_action is {payload.get('max_action')}")
    embedded = payload.get("config", {})
    for key, expected in required.items():
        if embedded.get(key) != expected:
            raise ValueError(
                f"embedded config {key}={embedded.get(key)!r}, expected {expected!r}"
            )
    return config, payload


def category_counts(
    geometry: dict[str, np.ndarray], horizon: float
) -> dict[str, int]:
    counts = {name: 0 for name in CATEGORIES}
    for row in range(len(geometry["first_hit_time"])):
        counts[classify_horizon(geometry, row, horizon)] += 1
    if sum(counts.values()) != len(geometry["first_hit_time"]):
        raise AssertionError("category denominator mismatch")
    return counts


def run_pilot(
    out_dir: Path,
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    exclusion_paths: list[Path],
) -> dict[str, Any]:
    design_path = out_dir / "DESIGN_LOCK.json"
    harness_path = out_dir / "HARNESS.json"
    if not design_path.is_file() or not harness_path.is_file():
        raise FileNotFoundError("DESIGN_LOCK.json and passing HARNESS.json are required")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    harness = json.loads(harness_path.read_text(encoding="utf-8"))
    if not harness.get("pass"):
        raise RuntimeError("harness did not pass")
    locked_source_hashes = source_hashes()
    if design["source_sha256"] != locked_source_hashes:
        raise RuntimeError("source changed after DESIGN_LOCK")
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"pilot requires CPU backend, got {jax.default_backend()}")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("pilot requires jax_enable_x64=true")

    source_snapshots = write_source_snapshots(out_dir)
    provenance = git_provenance()
    input_lock = write_input_lock(
        out_dir, checkpoint, config_path, dataset, exclusion_paths
    )
    config, payload = validate_pilot_inputs(checkpoint, config_path, dataset)
    observations, actions, mean, std = load_qlearning_rows(dataset)
    stored_mean = np.asarray(payload["mean"], dtype=np.float32)
    stored_std = np.asarray(payload["std"], dtype=np.float32)
    if not np.allclose(mean, stored_mean, rtol=0.0, atol=2e-5):
        raise ValueError("dataset mean does not match checkpoint normalization")
    if not np.allclose(std, stored_std, rtol=0.0, atol=2e-5):
        raise ValueError("dataset std does not match checkpoint normalization")
    selected, exclusion_records, exclusion_union = select_pilot_indices(
        actions, exclusion_paths
    )
    expected_counts = {
        record["basename"]: record["n_indices"] for record in PINNED_EXCLUSIONS
    }
    actual_counts = {
        Path(record["path"]).name: record["n_indices"] for record in exclusion_records
    }
    if actual_counts != expected_counts:
        raise AssertionError("canonical exclusion file counts changed")
    if exclusion_union != PINNED_EXCLUSION_UNION:
        raise AssertionError("canonical seed0/seed1 exclusion union is not 1022")
    index_path = out_dir / "pilot_state_indices.npy"
    if index_path.exists():
        raise FileExistsError(index_path)
    np.save(index_path, selected, allow_pickle=False)
    normalized_states = (
        observations[selected].astype(np.float64) - stored_mean.astype(np.float64)
    ) / stored_std.astype(np.float64)
    anchors = actions[selected].astype(np.float64)
    params = extract_q1(payload)
    q_anchor, _, _ = q1_forward(params, normalized_states, anchors)
    c_ref = float(np.mean(np.abs(q_anchor)) + EPS)
    geometry = analyze_geometry(params, normalized_states, anchors, c_ref)
    grad_jax = jax_gradients(params, normalized_states, anchors)
    grad_error = np.max(np.abs(geometry["grad"] - grad_jax), axis=-1)
    grad_bound = 1e-9 * np.maximum(1.0, np.max(np.abs(grad_jax), axis=-1))
    if not bool(np.all(grad_error <= grad_bound)):
        raise AssertionError(
            f"analytic/JAX gradient mismatch max={float(np.max(grad_error))}"
        )

    state_path = out_dir / "STATE_GEOMETRY.npz"
    if state_path.exists():
        raise FileExistsError(state_path)
    np.savez_compressed(
        state_path,
        dataset_row=selected,
        normalized_state=normalized_states,
        anchor_action=anchors,
        q_anchor=geometry["q_anchor"],
        grad=geometry["grad"],
        velocity=geometry["velocity"],
        relu_hit_time=geometry["relu_hit_time"],
        box_hit_time=geometry["box_hit_time"],
        first_hit_time=geometry["first_hit_time"],
        first_hit_code=geometry["first_hit_code"],
        first_layer=geometry["first_layer"],
        first_unit=geometry["first_unit"],
        boundary_ambiguous=geometry["boundary_ambiguous"],
        nonfinite=geometry["nonfinite"],
        anchor_min_scaled_margin=geometry["anchor_min_scaled_margin"],
        grad_jax=grad_jax,
        c_ref=np.array(c_ref, dtype=np.float64),
    )

    cell_path = out_dir / "residence_cells.csv"
    fieldnames = [
        "environment",
        "seed",
        "T",
        "K",
        "h",
        "n_states",
        *[f"full_{name}" for name in CATEGORIES],
        *[f"step_{name}" for name in CATEGORIES],
        "full_residence_fraction",
        "step_residence_fraction",
        "full_affine_q_max_abs_residual",
    ]
    rows: list[dict[str, Any]] = []
    with cell_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for total in TIMES:
            full_counts = category_counts(geometry, total)
            full_resident = np.array(
                [
                    classify_horizon(geometry, row, total) == "resident"
                    for row in range(N_PILOT)
                ]
            )
            endpoint = anchors + total * geometry["velocity"]
            q_endpoint, _, _ = q1_forward(params, normalized_states, endpoint)
            q_affine = geometry["q_anchor"] + total * np.sum(
                geometry["grad"] * geometry["velocity"], axis=-1
            )
            affine_residual = (
                float(np.max(np.abs(q_endpoint[full_resident] - q_affine[full_resident])))
                if np.any(full_resident)
                else None
            )
            for k in SUBSTEPS:
                step = total / k
                step_counts = category_counts(geometry, step)
                row = {
                    "environment": ENVIRONMENT,
                    "seed": SEED,
                    "T": total,
                    "K": k,
                    "h": step,
                    "n_states": N_PILOT,
                    **{f"full_{name}": full_counts[name] for name in CATEGORIES},
                    **{f"step_{name}": step_counts[name] for name in CATEGORIES},
                    "full_residence_fraction": full_counts["resident"] / N_PILOT,
                    "step_residence_fraction": step_counts["resident"] / N_PILOT,
                    "full_affine_q_max_abs_residual": affine_residual,
                }
                writer.writerow(row)
                rows.append(row)

    input_doc = {
        "protocol": PROTOCOL,
        "development_only": True,
        "scientific_admissible": False,
        "input_lock_sha256": input_lock["input_lock_sha256"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "config_content_sha256": sha256_bytes(canonical_bytes(config)),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset.resolve()),
        "checkpoint_step": CHECKPOINT_STEP,
        "environment": ENVIRONMENT,
        "seed": SEED,
        "tau": PILOT_TAU,
        "processed_transition_rows": int(len(actions)),
        "normalization_mean_sha256": sha256_array(stored_mean),
        "normalization_std_sha256": sha256_array(stored_std),
        "sample_seed": PILOT_SAMPLE_SEED,
        "selected_indices_sha256": sha256_array(selected),
        "selected_index_file_sha256": sha256_file(index_path),
        "excluded_archived_index_files": exclusion_records,
        "excluded_union_n_indices": exclusion_union,
    }
    write_json_create(out_dir / "PILOT_INPUTS.json", input_doc)

    summary = {
        "protocol": PROTOCOL,
        "status": "development_pilot_complete",
        "development_only": True,
        "scientific_admissible": False,
        "n_states": N_PILOT,
        "n_cells": len(rows),
        "C_ref": c_ref,
        "analytic_vs_jax_grad_max_abs": float(np.max(grad_error)),
        "coverage_gate": None,
        "learned_slope": None,
        "cells": rows,
        "written_at": now_iso(),
    }
    write_json_create(out_dir / "SUMMARY.json", summary)

    if source_hashes() != locked_source_hashes:
        raise RuntimeError("source changed during pilot")
    if {name: row["sha256"] for name, row in source_snapshots.items()} != locked_source_hashes:
        raise AssertionError("source snapshot hashes do not match DESIGN_LOCK")
    artifacts = {}
    for name in (
        "DESIGN_LOCK.json",
        "HARNESS.json",
        "INPUT_LOCK.json",
        "PILOT_INPUTS.json",
        "pilot_state_indices.npy",
        "STATE_GEOMETRY.npz",
        "residence_cells.csv",
        "SUMMARY.json",
        *[row["path"] for row in source_snapshots.values()],
    ):
        artifacts[name] = sha256_file(out_dir / name)
    manifest = {
        "protocol": PROTOCOL,
        "status": "pilot_complete",
        "development_only": True,
        "scientific_admissible": False,
        "reason": (
            "already-exposed HalfCheetah engineering pilot; final contract excludes "
            "the HalfCheetah family"
        ),
        "design_sha256": design["design_sha256"],
        "input_lock_sha256": input_lock["input_lock_sha256"],
        "source_sha256": locked_source_hashes,
        "source_snapshots": source_snapshots,
        "git_provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "counts": {"states": N_PILOT, "cells": len(rows)},
        "artifacts": artifacts,
        "written_at": now_iso(),
    }
    write_json_create(out_dir / "MANIFEST.json", manifest)
    write_json_create(
        out_dir / "STATUS.json",
        {
            "protocol": PROTOCOL,
            "status": "pilot_complete",
            "development_only": True,
            "scientific_admissible": False,
            "written_at": now_iso(),
        },
    )
    return manifest


# --- Final scientific audit implementation --------------------------------


def dataset_filename(env: str) -> str:
    name, tail = env.split("-", 1)
    quality = tail.rsplit("-v2", 1)[0]
    return f"{name}_{quality.replace('-', '_')}-v2.hdf5"


def final_sample_seed(env: str, seed: int) -> int:
    return FINAL_SAMPLE_SEED_BASE + FINAL_ENVIRONMENTS.index(env) * 10 + int(seed)


def unique_horizons() -> list[float]:
    horizons = {float(total) for total in TIMES}
    horizons.update(float(total) / k for total in TIMES for k in SUBSTEPS)
    return sorted(horizons)


def read_dev_bundle_env(bundle_dir: Path) -> str | None:
    for name, key_path in (
        ("PILOT_INPUTS.json", ("environment",)),
        ("DESIGN_LOCK.json", ("pilot_contract", "environment")),
    ):
        path = bundle_dir / name
        if not path.is_file():
            continue
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in key_path:
            if not isinstance(payload, dict):
                payload = None
                break
            payload = payload.get(key)
        if isinstance(payload, str):
            return payload
    return None


def build_exclusion_inventory(
    archived_index_dir: Path, dev_bundle_glob: str
) -> list[dict[str, Any]]:
    """Enumerate every archived fixed_operator and development pilot index set.

    Hashes raw bytes; env tags control which rows are numerically applied per run
    while the full inventory is retained as auditable provenance.
    """
    inventory: list[dict[str, Any]] = []
    archived_index_dir = archived_index_dir.resolve()
    if not archived_index_dir.is_dir():
        raise FileNotFoundError(f"archived index dir missing: {archived_index_dir}")
    for path in sorted(archived_index_dir.glob("*.npy")):
        stem = path.stem
        if "_seed" not in stem:
            raise ValueError(f"unrecognized archived index name: {path.name}")
        env, seed_token = stem.rsplit("_seed", 1)
        values = np.asarray(np.load(path), dtype=np.int64)
        inventory.append(
            {
                "role": "archived_fixed_operator",
                "env": env,
                "seed": int(seed_token),
                "path": str(path),
                "basename": path.name,
                "sha256": sha256_file(path),
                "n_indices": int(len(values)),
            }
        )
    glob_path = Path(dev_bundle_glob)
    for bundle_dir in sorted(glob_path.parent.glob(glob_path.name)):
        index_path = bundle_dir / "pilot_state_indices.npy"
        if not index_path.is_file():
            continue
        bundle_id = bundle_dir.name.rsplit(".", 1)[-1]
        environment = read_dev_bundle_env(bundle_dir)
        if environment is None:
            raise ValueError(
                f"development index environment is missing or unparseable: {bundle_dir}"
            )
        values = np.asarray(np.load(index_path), dtype=np.int64)
        inventory.append(
            {
                "role": "dev_pilot_bundle",
                "env": environment,
                "bundle_id": bundle_id,
                "path": str(index_path),
                "basename": f"{bundle_dir.name}/pilot_state_indices.npy",
                "sha256": sha256_file(index_path),
                "n_indices": int(len(values)),
            }
        )
    return inventory


def applied_exclusions_for_env(
    inventory: list[dict[str, Any]], env: str
) -> tuple[set[int], list[dict[str, Any]]]:
    excluded: set[int] = set()
    applied: list[dict[str, Any]] = []
    for record in inventory:
        if record.get("env") != env:
            continue
        values = np.asarray(np.load(record["path"]), dtype=np.int64)
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise AssertionError(f"exclusion file changed after freeze: {record['path']}")
        if len(values) != record["n_indices"]:
            raise AssertionError(f"exclusion count changed after freeze: {record['path']}")
        excluded.update(int(value) for value in values)
        applied.append(record)
    return excluded, applied


def select_final_indices(
    actions: np.ndarray, excluded: set[int], n_states: int, sample_seed: int
) -> tuple[np.ndarray, int]:
    eligible = np.flatnonzero(np.max(np.abs(actions), axis=-1) <= ACTION_ELIGIBILITY)
    eligible = np.asarray(
        [value for value in eligible if int(value) not in excluded], dtype=np.int64
    )
    if len(eligible) < n_states:
        raise RuntimeError(f"only {len(eligible)} eligible new indices for seed {sample_seed}")
    rng = np.random.default_rng(sample_seed)
    selected = rng.choice(eligible, size=n_states, replace=False).astype(np.int64)
    if excluded.intersection(int(value) for value in selected):
        raise AssertionError("final index exclusion failed")
    return selected, int(len(eligible))


def validate_final_inputs(
    checkpoint: Path, config_path: Path, dataset: Path, env: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (checkpoint, config_path, dataset):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "env": env,
        "seed": seed,
        "tau": FINAL_TAU,
        **FINAL_REQUIRED_FIXED_CONFIG,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key}={config.get(key)!r}, expected {expected!r}")
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local research checkpoint
    if int(payload.get("step", -1)) != CHECKPOINT_STEP:
        raise ValueError(f"checkpoint step is {payload.get('step')}")
    if float(payload.get("max_action", math.nan)) != BOX:
        raise ValueError(f"checkpoint max_action is {payload.get('max_action')}")
    embedded = payload.get("config", {})
    for key, expected in required.items():
        if embedded.get(key) != expected:
            raise ValueError(
                f"embedded config {key}={embedded.get(key)!r}, expected {expected!r}"
            )
    return config, payload


def survival_curve(
    geometry: dict[str, np.ndarray], horizons: list[float]
) -> list[dict[str, Any]]:
    n_states = len(geometry["first_hit_time"])
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        counts = category_counts(geometry, horizon)
        rows.append(
            {
                "horizon": float(horizon),
                "n_resident": int(counts["resident"]),
                "resident_fraction": counts["resident"] / n_states,
                "categories": counts,
            }
        )
    return rows


def write_residence_cells(
    cell_path: Path,
    geometry: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
    normalized_states: np.ndarray,
    anchors: np.ndarray,
    env: str,
    seed: int,
    n_states: int,
) -> list[dict[str, Any]]:
    fieldnames = [
        "environment",
        "seed",
        "T",
        "K",
        "h",
        "n_states",
        *[f"full_{name}" for name in CATEGORIES],
        *[f"step_{name}" for name in CATEGORIES],
        "full_residence_fraction",
        "step_residence_fraction",
        "full_affine_q_max_abs_residual",
    ]
    rows: list[dict[str, Any]] = []
    with cell_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for total in TIMES:
            full_counts = category_counts(geometry, total)
            full_resident = np.array(
                [
                    classify_horizon(geometry, row, total) == "resident"
                    for row in range(n_states)
                ]
            )
            endpoint = anchors + total * geometry["velocity"]
            q_endpoint, _, _ = q1_forward(params, normalized_states, endpoint)
            q_affine = geometry["q_anchor"] + total * np.sum(
                geometry["grad"] * geometry["velocity"], axis=-1
            )
            affine_residual = (
                float(np.max(np.abs(q_endpoint[full_resident] - q_affine[full_resident])))
                if np.any(full_resident)
                else None
            )
            for k in SUBSTEPS:
                step = total / k
                step_counts = category_counts(geometry, step)
                row = {
                    "environment": env,
                    "seed": seed,
                    "T": total,
                    "K": k,
                    "h": step,
                    "n_states": n_states,
                    **{f"full_{name}": full_counts[name] for name in CATEGORIES},
                    **{f"step_{name}": step_counts[name] for name in CATEGORIES},
                    "full_residence_fraction": full_counts["resident"] / n_states,
                    "step_residence_fraction": step_counts["resident"] / n_states,
                    "full_affine_q_max_abs_residual": affine_residual,
                }
                writer.writerow(row)
                rows.append(row)
    return rows


def run_one_final(
    sub_dir: Path,
    env: str,
    seed: int,
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    sub_dir.mkdir(parents=True, exist_ok=False)
    config, payload = validate_final_inputs(checkpoint, config_path, dataset, env, seed)
    observations, actions, mean, std = load_qlearning_rows(dataset)
    stored_mean = np.asarray(payload["mean"], dtype=np.float32)
    stored_std = np.asarray(payload["std"], dtype=np.float32)
    if not np.allclose(mean, stored_mean, rtol=0.0, atol=2e-5):
        raise ValueError(f"{env} seed{seed}: dataset mean != checkpoint normalization")
    if not np.allclose(std, stored_std, rtol=0.0, atol=2e-5):
        raise ValueError(f"{env} seed{seed}: dataset std != checkpoint normalization")

    excluded, applied_records = applied_exclusions_for_env(inventory, env)
    sample_seed = final_sample_seed(env, seed)
    selected, n_eligible = select_final_indices(
        actions, excluded, FINAL_N_STATES, sample_seed
    )
    index_path = sub_dir / "state_indices.npy"
    np.save(index_path, selected, allow_pickle=False)

    normalized_states = (
        observations[selected].astype(np.float64) - stored_mean.astype(np.float64)
    ) / stored_std.astype(np.float64)
    anchors = actions[selected].astype(np.float64)
    params = extract_q1(payload)
    q_anchor, _, _ = q1_forward(params, normalized_states, anchors)
    c_ref = float(np.mean(np.abs(q_anchor)) + EPS)
    geometry = analyze_geometry(params, normalized_states, anchors, c_ref)
    grad_jax = jax_gradients(params, normalized_states, anchors)
    grad_error = np.max(np.abs(geometry["grad"] - grad_jax), axis=-1)
    grad_bound = 1e-9 * np.maximum(1.0, np.max(np.abs(grad_jax), axis=-1))
    if not bool(np.all(grad_error <= grad_bound)):
        raise AssertionError(
            f"{env} seed{seed}: analytic/JAX gradient mismatch max={float(np.max(grad_error))}"
        )

    np.savez_compressed(
        sub_dir / "STATE_GEOMETRY.npz",
        dataset_row=selected,
        normalized_state=normalized_states,
        anchor_action=anchors,
        q_anchor=geometry["q_anchor"],
        grad=geometry["grad"],
        velocity=geometry["velocity"],
        relu_hit_time=geometry["relu_hit_time"],
        box_hit_time=geometry["box_hit_time"],
        first_hit_time=geometry["first_hit_time"],
        first_hit_code=geometry["first_hit_code"],
        first_layer=geometry["first_layer"],
        first_unit=geometry["first_unit"],
        boundary_ambiguous=geometry["boundary_ambiguous"],
        nonfinite=geometry["nonfinite"],
        anchor_min_scaled_margin=geometry["anchor_min_scaled_margin"],
        grad_jax=grad_jax,
        c_ref=np.array(c_ref, dtype=np.float64),
    )

    cells = write_residence_cells(
        sub_dir / "residence_cells.csv",
        geometry,
        params,
        normalized_states,
        anchors,
        env,
        seed,
        FINAL_N_STATES,
    )
    horizons = unique_horizons()
    survival = survival_curve(geometry, horizons)
    full_counts_by_T = {
        f"{total}": category_counts(geometry, total) for total in TIMES
    }

    run_inputs = {
        "protocol": FINAL_PROTOCOL,
        "environment": env,
        "seed": seed,
        "tau": FINAL_TAU,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint.resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "config_content_sha256": sha256_bytes(canonical_bytes(config)),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset.resolve()),
        "checkpoint_step": CHECKPOINT_STEP,
        "processed_transition_rows": int(len(actions)),
        "normalization_mean_sha256": sha256_array(stored_mean),
        "normalization_std_sha256": sha256_array(stored_std),
        "sample_seed": sample_seed,
        "n_states": FINAL_N_STATES,
        "n_eligible": n_eligible,
        "selected_indices_sha256": sha256_array(selected),
        "selected_index_file_sha256": sha256_file(index_path),
        "applied_exclusions": applied_records,
        "applied_exclusion_union_n_indices": len(excluded),
        "q1_parameter_shapes": {
            name: list(np.asarray(value).shape) for name, value in params.items()
        },
    }
    write_json_create(sub_dir / "RUN_INPUTS.json", run_inputs)

    run_summary = {
        "protocol": FINAL_PROTOCOL,
        "environment": env,
        "seed": seed,
        "n_states": FINAL_N_STATES,
        "n_eligible": n_eligible,
        "sample_seed": sample_seed,
        "C_ref": c_ref,
        "analytic_vs_jax_grad_max_abs": float(np.max(grad_error)),
        "full_category_counts_by_T": full_counts_by_T,
        "survival": survival,
        "n_cells": len(cells),
        "status": "run_complete_pending_verification",
        "scientific_admissible": False,
        "scientific_admissible_when_verified": True,
        "written_at": now_iso(),
    }
    write_json_create(sub_dir / "RUN_SUMMARY.json", run_summary)
    return run_summary


def build_input_inventory(
    results_root: Path, dataset_dir: Path, environments: list[str]
) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()
    for env in environments:
        for seed in FINAL_SEEDS:
            run_dir = results_root / f"{env}_tau1_seed{seed}"
            checkpoint = run_dir / "params_1000000.pkl"
            config_path = run_dir / "config.json"
            if not checkpoint.is_file() or not config_path.is_file():
                raise FileNotFoundError(f"missing checkpoint/config for {env} seed{seed}")
            checkpoints.append(
                {
                    "env": env,
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "config": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                }
            )
        dataset = dataset_dir / dataset_filename(env)
        if not dataset.is_file():
            raise FileNotFoundError(f"missing dataset for {env}: {dataset}")
        key = str(dataset.resolve())
        if key not in seen_datasets:
            seen_datasets.add(key)
            datasets.append(
                {
                    "env": env,
                    "dataset": key,
                    "dataset_sha256": sha256_file(dataset),
                }
            )
    return {"checkpoints": checkpoints, "datasets": datasets}


def write_final_design_lock(
    out_dir: Path,
    environments: list[str],
    inventory: list[dict[str, Any]],
    input_inventory: dict[str, Any],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "protocol": FINAL_PROTOCOL,
        "status": "final_protocol_frozen_before_sampling",
        "scientific_admissible_when_verified": True,
        "reuses_pilot_geometry_and_residence_math": True,
        "written_at": now_iso(),
        "estimand": {
            "learned_primary": "relu_activation_region_residence",
            "direction": "v = action_dim * grad_a Q1(s,a0) / C_ref",
            "C_ref": "mean_i(abs(Q1(s_i,a0_i))) + 1e-6",
            "full_residence": "T is strictly before first ReLU/action-box exit",
            "first_step_residence": "T/K is strictly before first exit",
            "unique_horizon_estimand": "empirical survival curve P(t_exit > t)",
            "cell_dependence": (
                "repeated (T,K) cells with the same horizon t=T or t=T/K are "
                "deterministic views, never independent evidence"
            ),
            "boundary_rows": "retained scope diagnostics, never numerical errors",
            "implicit_scope": "same-region local backward-Euler branch only",
            "learned_slope": None,
            "learned_support_gate": None,
            "global_proximal_claim": False,
        },
        "grid": {"T": list(TIMES), "K": list(SUBSTEPS)},
        "numeric_contract": {
            "precision": "float64",
            "action_box": [-BOX, BOX],
            "action_eligibility": ACTION_ELIGIBILITY,
            "categories": list(CATEGORIES),
            "denominator_rule": "all sampled anchors retained in exactly one category",
        },
        "checkpoint_config_contract": FINAL_REQUIRED_FIXED_CONFIG,
        "environments": list(environments),
        "excluded_family": FINAL_EXCLUDED_FAMILY,
        "seeds": list(FINAL_SEEDS),
        "n_runs": len(environments) * len(FINAL_SEEDS),
        "n_states_per_run": FINAL_N_STATES,
        "sample_seed_base": FINAL_SAMPLE_SEED_BASE,
        "per_run_sample_seeds": {
            f"{env}_seed{seed}": final_sample_seed(env, seed)
            for env in environments
            for seed in FINAL_SEEDS
        },
        "exclusion_inventory": inventory,
        "exclusion_inventory_n_files": len(inventory),
        "known_dev_bundle_ids": list(KNOWN_DEV_BUNDLE_IDS),
        "exclude_all_archived_and_development_attempt_indices": True,
        "indices_created_only_after_final_protocol_freeze": True,
        "primary_summary": "unique-horizon empirical survival curve P(t_exit > t)",
        "repeated_cell_policy": "collapse equal horizons; do not count as independent evidence",
        "input_inventory": input_inventory,
        "source_sha256": source_hashes(final_source_files()),
        "coverage_gate": None,
        "learned_slope": None,
    }
    document["design_sha256"] = sha256_bytes(canonical_bytes(document))
    write_json_create(out_dir / "FINAL_DESIGN_LOCK.json", document)
    return document


def pooled_survival(
    per_run: list[dict[str, Any]], horizons: list[float], subset: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    runs = per_run if subset is None else subset
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        n_resident = 0
        n_total = 0
        categories = {name: 0 for name in CATEGORIES}
        for run in runs:
            match = next(
                item for item in run["survival"] if math.isclose(item["horizon"], horizon)
            )
            n_resident += int(match["n_resident"])
            n_total += int(run["n_states"])
            for name in CATEGORIES:
                categories[name] += int(match["categories"][name])
        rows.append(
            {
                "horizon": float(horizon),
                "n_resident": n_resident,
                "n_total": n_total,
                "resident_fraction": (n_resident / n_total) if n_total else 0.0,
                "categories": categories,
            }
        )
    return rows


def task_equal_survival(
    task_survival: dict[str, list[dict[str, Any]]], horizons: list[float]
) -> list[dict[str, Any]]:
    tasks = sorted(task_survival)
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        matches = [
            next(
                row
                for row in task_survival[task]
                if math.isclose(row["horizon"], horizon)
            )
            for task in tasks
        ]
        rows.append(
            {
                "horizon": float(horizon),
                "n_tasks": len(tasks),
                "resident_fraction": float(
                    np.mean([row["resident_fraction"] for row in matches])
                ),
                "category_fractions": {
                    name: float(
                        np.mean(
                            [
                                row["categories"][name] / row["n_total"]
                                for row in matches
                            ]
                        )
                    )
                    for name in CATEGORIES
                },
            }
        )
    return rows


def run_final(
    out_dir: Path,
    results_root: Path,
    dataset_dir: Path,
    archived_index_dir: Path,
    dev_bundle_glob: str,
) -> dict[str, Any]:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"final audit requires CPU backend, got {jax.default_backend()}")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("final audit requires jax_enable_x64=true")
    provenance = git_provenance()
    require_clean_origin_main(provenance)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"final audit requires an empty directory: {out_dir}")

    environments = list(FINAL_ENVIRONMENTS)
    if any(env.startswith(f"{FINAL_EXCLUDED_FAMILY}-") for env in environments):
        raise AssertionError("final environments must exclude the HalfCheetah family")

    inventory = build_exclusion_inventory(archived_index_dir, dev_bundle_glob)
    present_ids = {
        record["bundle_id"]
        for record in inventory
        if record["role"] == "dev_pilot_bundle"
    }
    missing_ids = [bundle for bundle in KNOWN_DEV_BUNDLE_IDS if bundle not in present_ids]
    if missing_ids:
        raise RuntimeError(f"exclusion inventory missing dev bundles: {missing_ids}")
    archived_envs = {
        record["env"] for record in inventory if record["role"] == "archived_fixed_operator"
    }
    for env in environments:
        if env not in archived_envs:
            raise RuntimeError(f"archived exclusion indices missing for {env}")

    input_inventory = build_input_inventory(results_root, dataset_dir, environments)
    # Freeze the protocol (raw-byte hashes only) BEFORE any semantic checkpoint read.
    design = write_final_design_lock(out_dir, environments, inventory, input_inventory)
    source_snapshots = write_source_snapshots(out_dir, final_source_files())
    locked_source_hashes = source_hashes(final_source_files())
    if design["source_sha256"] != locked_source_hashes:
        raise RuntimeError("source changed after FINAL_DESIGN_LOCK")
    run_harness(out_dir, gate_filename="FINAL_DESIGN_LOCK.json")

    per_run: list[dict[str, Any]] = []
    for env in environments:
        dataset = dataset_dir / dataset_filename(env)
        for seed in FINAL_SEEDS:
            run_dir = results_root / f"{env}_tau1_seed{seed}"
            checkpoint = run_dir / "params_1000000.pkl"
            config_path = run_dir / "config.json"
            summary = run_one_final(
                out_dir / f"{env}_seed{seed}",
                env,
                seed,
                checkpoint,
                config_path,
                dataset,
                inventory,
            )
            per_run.append(summary)

    horizons = unique_horizons()
    families = sorted({env.split("-", 1)[0] for env in environments})
    task_survival = {
        env: pooled_survival(
            per_run,
            horizons,
            [run for run in per_run if run["environment"] == env],
        )
        for env in environments
    }
    family_survival = {
        family: pooled_survival(
            per_run,
            horizons,
            [run for run in per_run if run["environment"].split("-", 1)[0] == family],
        )
        for family in families
    }
    n_anchors_total = sum(run["n_states"] for run in per_run)
    summary = {
        "protocol": FINAL_PROTOCOL,
        "status": FINAL_RAW_STATUS,
        "scientific_admissible": False,
        "scientific_admissible_when_verified": True,
        "n_runs": len(per_run),
        "n_states_per_run": FINAL_N_STATES,
        "n_anchors_total": n_anchors_total,
        "environments": environments,
        "seeds": list(FINAL_SEEDS),
        "horizons": horizons,
        "per_run": per_run,
        "pooled_survival": pooled_survival(per_run, horizons),
        "task_survival": task_survival,
        "task_equal_survival": task_equal_survival(task_survival, horizons),
        "family_survival": family_survival,
        "coverage_gate": None,
        "learned_slope": None,
        "written_at": now_iso(),
    }
    write_json_create(out_dir / "SUMMARY.json", summary)

    if source_hashes(final_source_files()) != locked_source_hashes:
        raise RuntimeError("source changed during final audit")

    artifacts: dict[str, str] = {}
    root_files = ["FINAL_DESIGN_LOCK.json", "HARNESS.json", "SUMMARY.json"]
    root_files.extend(row["path"] for row in source_snapshots.values())
    for name in root_files:
        artifacts[name] = sha256_file(out_dir / name)
    for run in per_run:
        prefix = f"{run['environment']}_seed{run['seed']}"
        for leaf in (
            "state_indices.npy",
            "STATE_GEOMETRY.npz",
            "residence_cells.csv",
            "RUN_INPUTS.json",
            "RUN_SUMMARY.json",
        ):
            rel = f"{prefix}/{leaf}"
            artifacts[rel] = sha256_file(out_dir / rel)

    manifest = {
        "protocol": FINAL_PROTOCOL,
        "status": FINAL_RAW_STATUS,
        "scientific_admissible": False,
        "scientific_admissible_when_verified": True,
        "reason": (
            "prelocked fresh-index Hopper/Walker ReLU residence audit on "
            "previously studied critics; HalfCheetah and every archived or "
            "development-attempt index set excluded"
        ),
        "design_sha256": design["design_sha256"],
        "source_sha256": locked_source_hashes,
        "source_snapshots": source_snapshots,
        "git_provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "counts": {
            "runs": len(per_run),
            "states_per_run": FINAL_N_STATES,
            "anchors_total": n_anchors_total,
        },
        "artifacts": artifacts,
        "written_at": now_iso(),
    }
    write_json_create(out_dir / "MANIFEST.json", manifest)
    write_json_create(
        out_dir / "STATUS.json",
        {
            "protocol": FINAL_PROTOCOL,
            "status": FINAL_RAW_STATUS,
            "scientific_admissible": False,
            "scientific_admissible_when_verified": True,
            "written_at": now_iso(),
        },
    )
    return manifest


def main() -> int:
    jax.config.update("jax_enable_x64", True)
    args = parse_args()
    if args.phase == "final":
        manifest = run_final(
            args.out_dir,
            args.results_root,
            args.dataset_dir,
            args.archived_index_dir,
            args.dev_bundle_glob,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "scientific_admissible": manifest["scientific_admissible"],
                    "counts": manifest["counts"],
                },
                indent=2,
            )
        )
        return 0
    if args.phase in ("design", "all"):
        write_design_lock(args.out_dir)
    if args.phase in ("harness", "all"):
        run_harness(args.out_dir)
    if args.phase in ("pilot", "all"):
        manifest = run_pilot(
            args.out_dir,
            args.checkpoint,
            args.config,
            args.dataset,
            args.exclude_indices,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "development_only": manifest["development_only"],
                    "scientific_admissible": manifest["scientific_admissible"],
                    "counts": manifest["counts"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
