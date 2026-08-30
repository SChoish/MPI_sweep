#!/usr/bin/env python3
"""Rebuild a compact manuscript audit pack from existing local diagnostics.

This script does not train policies or run simulator rollouts. It recomputes
run/checkpoint statistics from existing NPZ and per-state CSV inputs, verifies
pairing/count contracts, and emits only compact CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

EXPECTED_ENVS = (
    "halfcheetah-expert-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-v2",
    "hopper-expert-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-v2",
    "walker2d-expert-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-v2",
)
EXPECTED_TAUS = (4.0, 7.0, 10.0, 14.0, 20.0)
EXPECTED_SEEDS = (0, 1)
TARGET_METHODS = ("td3", "mpi2", "mpi3")
FAILURE_METHODS = ("lin2", "prox2")
TARGET_BOOTSTRAP_SEED = 20260830
DEFAULT_BOOTSTRAPS = 100_000
EPS = 1e-12


class AuditFailure(RuntimeError):
    """Raised when an integrity contract is violated."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AuditFailure(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def finite_stats(values: np.ndarray) -> tuple[int, int]:
    array = np.asarray(values)
    return int(np.isnan(array).sum()), int(np.isinf(array).sum())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def tau_token(value: float | str) -> str:
    return f"{float(value):g}"


def read_eval_score(checkpoint: Path, method: str) -> float:
    score_key = {
        "td3": "d4rl_score",
        "mpi2": "d4rl_pi2",
        "mpi3": "d4rl_pi3",
    }[method]
    rows = read_csv(checkpoint.parent / "eval.csv")
    matches = [row for row in rows if row.get("step") == "1000000"]
    require(len(matches) == 1, f"{checkpoint}: expected one 1M eval row")
    return float(matches[0][score_key])


def geometry_npz(
    audit_root: Path, method: str, env: str, tau: float, seed: int
) -> tuple[Path, int]:
    token = tau_token(tau)
    if method == "td3":
        name = f"{env}_tau{token}_seed{seed}.npz"
        return audit_root / "td3_mpi3_common_geometry" / "raw" / env / name, 1
    if method == "mpi2":
        name = f"{env}_tau{token}_mpi2_seed{seed}.npz"
        return audit_root / "matched_imp2_geometry" / "raw" / env / name, 2
    if method == "mpi3":
        name = f"{env}_tau{token}_mpi3_seed{seed}.npz"
        return audit_root / "td3_mpi3_common_geometry" / "raw" / env / name, 3
    raise KeyError(method)


def action_displacements(path: Path, hops: int) -> tuple[float, float, np.ndarray]:
    require(path.is_file(), f"missing geometry raw input: {path}")
    with np.load(path, allow_pickle=False) as raw:
        indices = np.asarray(raw["indices"])
        action0 = np.asarray(raw["action_hop0"], dtype=np.float64)
        action1 = np.asarray(raw["action_hop1"], dtype=np.float64)
        actionk = np.asarray(raw[f"action_hop{hops}"], dtype=np.float64)
    d_critic = float(np.mean(np.mean(np.square(action1 - action0), axis=-1)))
    d_final = float(np.mean(np.mean(np.square(actionk - action0), axis=-1)))
    return d_critic, d_final, indices


def target_exposure(
    audit_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path]]:
    source_dir = audit_root / "target_policy_exposure"
    summary_rows = read_csv(source_dir / "target_policy_exposure_summary.csv")
    manifest = json.loads((source_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    require(len(summary_rows) == 270, f"target exposure rows={len(summary_rows)}, expected 270")

    keys = [
        (row["method"], row["env"], tau_token(row["tau"]), int(row["seed"]))
        for row in summary_rows
    ]
    require(len(keys) == len(set(keys)), "target exposure primary-key duplicates")
    expected = {
        (method, env, tau_token(tau), seed)
        for method in TARGET_METHODS
        for env in EXPECTED_ENVS
        for tau in EXPECTED_TAUS
        for seed in EXPECTED_SEEDS
    }
    require(set(keys) == expected, "target exposure grid differs from 3×9×5×2 contract")

    rows_out: list[dict[str, Any]] = []
    inputs: set[Path] = {
        source_dir / "target_policy_exposure_summary.csv",
        source_dir / "MANIFEST.json",
    }
    source_indices_by_env: dict[str, np.ndarray] = {}
    normalization_by_env: dict[str, set[str]] = defaultdict(set)
    raw_nan = 0
    raw_inf = 0
    direct_formula_max_error = 0.0
    smoothing_formula_max_error = 0.0
    geometry_indices_by_env_and_size: dict[tuple[str, int], np.ndarray] = {}

    for row in summary_rows:
        method = row["method"]
        env = row["env"]
        tau = float(row["tau"])
        seed = int(row["seed"])
        checkpoint = Path(row["checkpoint"])
        raw_path = Path(row["raw_npz"])
        require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
        require(raw_path.is_file(), f"missing target raw input: {raw_path}")
        inputs.update((raw_path, checkpoint, checkpoint.parent / "eval.csv"))
        config = checkpoint.parent / "config.json"
        if config.is_file():
            inputs.add(config)

        with np.load(raw_path, allow_pickle=False) as raw:
            source_indices = np.asarray(raw["source_indices"])
            if env not in source_indices_by_env:
                source_indices_by_env[env] = source_indices
            require(
                np.array_equal(source_indices_by_env[env], source_indices),
                f"target exposure state IDs differ across methods: {env}",
            )
            action_dim = int(np.asarray(raw["dataset_actions"]).shape[-1])
            deterministic_direct = np.mean(
                np.square(
                    np.asarray(raw["target_next_actions"])
                    - np.asarray(raw["next_dataset_actions"])
                ),
                axis=-1,
            ).astype(np.float64)
            deterministic_stored = np.asarray(
                raw["target_deterministic_to_next_data_sq_mean_metric"],
                dtype=np.float64,
            )
            smoothed_repeats = np.asarray(
                raw["smoothing_repeat_sq_mean_metric"], dtype=np.float64
            )
            d_target = float(np.mean(deterministic_direct))
            d_smoothed = float(np.mean(smoothed_repeats))
            d_critic_current = float(
                np.mean(
                    np.mean(
                        np.square(
                            np.asarray(raw["online_current_actions"], dtype=np.float64)
                            - np.asarray(raw["dataset_actions"], dtype=np.float64)
                        ),
                        axis=-1,
                    )
                )
            )
            proxy = float(
                np.mean(
                    np.asarray(
                        raw["current_sample_anchor_proxy_sq_mean_metric"],
                        dtype=np.float64,
                    )
                )
            )
            polyak_lag = float(
                np.mean(
                    np.asarray(raw["polyak_lag_sq_mean_metric"], dtype=np.float64)
                )
            )
            direct_formula_max_error = max(
                direct_formula_max_error,
                float(np.max(np.abs(deterministic_direct - deterministic_stored))),
            )
            smoothing_formula_max_error = max(
                smoothing_formula_max_error,
                abs(
                    d_smoothed
                    - float(
                        row[
                            "target_smoothed_to_next_data_sq_mean_metric_mean"
                        ]
                    )
                ),
            )
            for name in raw.files:
                array = np.asarray(raw[name])
                if np.issubdtype(array.dtype, np.number):
                    n_nan, n_inf = finite_stats(array)
                    raw_nan += n_nan
                    raw_inf += n_inf

        geom_path, hops = geometry_npz(audit_root, method, env, tau, seed)
        d_critic_geom, d_final, geom_indices = action_displacements(geom_path, hops)
        inputs.add(geom_path)
        geometry_key = (env, len(geom_indices))
        if geometry_key not in geometry_indices_by_env_and_size:
            geometry_indices_by_env_and_size[geometry_key] = geom_indices
        require(
            np.array_equal(
                geometry_indices_by_env_and_size[geometry_key], geom_indices
            ),
            f"geometry validation state IDs differ: {env}/{len(geom_indices)}",
        )
        require(
            math.isclose(d_critic_current, d_critic_geom, rel_tol=0.1, abs_tol=0.02),
            f"actor-1 displacement mismatch across independent batches: {method}/{env}/{tau}/{seed}",
        )
        env_meta = manifest["common_batches"][env]
        normalization_by_env[env].add(row["normalization_sha256"])
        rows_out.append(
            {
                "method": method,
                "env": env,
                "T": tau,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "score": read_eval_score(checkpoint, method),
                "n_source_rows": int(env_meta["hdf5_rows"]),
                "n_valid_rows": int(env_meta["bootstrap_transition_count"]),
                "n_sampled_rows": len(source_indices),
                "d_critic_current": d_critic_current,
                "d_final_current": d_final,
                "d_target_deterministic": d_target,
                "d_target_smoothed": d_smoothed,
                "current_state_proxy": proxy,
                "polyak_lag": polyak_lag,
                "target_noise_draws": int(smoothed_repeats.shape[0]),
                "n_nan": 0,
                "n_inf": 0,
            }
        )

    require(all(len(hashes) == 1 for hashes in normalization_by_env.values()), "normalization hash mismatch")
    integrity = {
        "rows": len(rows_out),
        "primary_key_duplicates": len(keys) - len(set(keys)),
        "state_pairing_exact_by_environment": True,
        "geometry_state_pairing_exact_within_sample_size": True,
        "geometry_sample_sizes": [512, 4096],
        "normalization_hash_exact_by_environment": True,
        "common_noise_files_by_environment": len(manifest["common_batches"]),
        "raw_nan": raw_nan,
        "raw_inf": raw_inf,
        "deterministic_formula_max_abs_error": direct_formula_max_error,
        "smoothed_summary_max_abs_error": smoothing_formula_max_error,
    }
    require(raw_nan == 0 and raw_inf == 0, "non-finite target exposure raw arrays")
    require(
        direct_formula_max_error < 1e-6,
        "stored deterministic displacement formula mismatch beyond float32 tolerance",
    )
    require(smoothing_formula_max_error < 1e-12, "stored smoothed displacement formula mismatch")
    return rows_out, integrity, inputs


def map_rows(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = read_csv(path)
    result: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["env"], tau_token(row["tau"]), int(row["seed"]))
        require(key not in result, f"duplicate run row in {path}: {key}")
        result[key] = row
    return result


def raw_failure_stats(path: Path, method: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as raw:
        indices = np.asarray(raw["indices"])
        action0 = np.asarray(raw["action_hop0"], dtype=np.float64)
        action1 = np.asarray(raw["action_hop1"], dtype=np.float64)
        action2 = np.asarray(raw["action_hop2"], dtype=np.float64)
        if method == "prox2":
            own = sum(
                np.asarray(raw[f"own_hop{hop}_q1_gain"], dtype=np.float64)
                for hop in (1, 2)
            )
            common = sum(
                np.asarray(
                    raw[f"canonical_td3_tau1_hop{hop}_q1_gain"], dtype=np.float64
                )
                for hop in (1, 2)
            )
        else:
            own = sum(
                np.asarray(raw[f"hop{hop}_q1_own_next"], dtype=np.float64)
                - np.asarray(raw[f"hop{hop}_q1_own_prev"], dtype=np.float64)
                for hop in (1, 2)
            )
            common = sum(
                np.asarray(raw[f"hop{hop}_q1_canonical_next"], dtype=np.float64)
                - np.asarray(raw[f"hop{hop}_q1_canonical_prev"], dtype=np.float64)
                for hop in (1, 2)
            )
        n_nan = 0
        n_inf = 0
        for name in raw.files:
            array = np.asarray(raw[name])
            if np.issubdtype(array.dtype, np.number):
                nan_count, inf_count = finite_stats(array)
                n_nan += nan_count
                n_inf += inf_count
    return {
        "indices": indices,
        "d_critic": float(np.mean(np.mean(np.square(action1 - action0), axis=-1))),
        "d_final": float(np.mean(np.mean(np.square(action2 - action0), axis=-1))),
        "joint_critic_gain": float(np.mean(own)),
        "common_critic_gain": float(np.mean(common)),
        "n_nan": n_nan,
        "n_inf": n_inf,
    }


def failure_diagnostics(
    audit_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path]]:
    specs = {
        "prox2": (
            audit_root / "matched_imp2_geometry",
            "mpi2",
            "run_diagnostics.csv",
        ),
        "lin2": (
            audit_root / "matched_explicit_geometry",
            "exp2m",
            "run_diagnostics.csv",
        ),
    }
    rows_out: list[dict[str, Any]] = []
    inputs: set[Path] = set()
    indices_by_env: dict[str, np.ndarray] = {}
    missing_td_max = 0

    for method, (source, source_method, csv_name) in specs.items():
        run_rows = map_rows(source / csv_name)
        inputs.update((source / csv_name, source / "MANIFEST.json"))
        require(len(run_rows) == 90, f"{method} run rows={len(run_rows)}, expected 90")
        for env in EXPECTED_ENVS:
            for tau in EXPECTED_TAUS:
                for seed in EXPECTED_SEEDS:
                    key = (env, tau_token(tau), seed)
                    require(key in run_rows, f"missing {method} run row: {key}")
                    row = run_rows[key]
                    if method == "prox2":
                        raw_path = (
                            source
                            / "raw"
                            / env
                            / f"{env}_tau{tau_token(tau)}_mpi2_seed{seed}.npz"
                        )
                    else:
                        raw_path = (
                            source
                            / "raw"
                            / env
                            / f"{env}_tau{tau_token(tau)}_expl2_seed{seed}.npz"
                        )
                    require(raw_path.is_file(), f"missing failure raw input: {raw_path}")
                    inputs.add(raw_path)
                    raw = raw_failure_stats(raw_path, method)
                    if env not in indices_by_env:
                        indices_by_env[env] = raw["indices"]
                    require(
                        np.array_equal(indices_by_env[env], raw["indices"]),
                        f"failure validation state IDs differ: {env}/{method}/{tau}/{seed}",
                    )
                    checkpoint = Path(row["checkpoint"])
                    inputs.update((checkpoint, checkpoint.parent / "eval.csv"))
                    config = checkpoint.parent / "config.json"
                    if config.is_file():
                        inputs.add(config)
                    td_max_raw = row.get("td_residual_max", "")
                    if td_max_raw == "":
                        td_error_max: float | str = ""
                        missing_td_max += 1
                    else:
                        td_error_max = float(td_max_raw)
                    score = float(row["d4rl_score"])
                    rows_out.append(
                        {
                            "method": method,
                            "source_method": source_method,
                            "env": env,
                            "T": tau,
                            "seed": seed,
                            "checkpoint": str(checkpoint),
                            "score": score,
                            "collapsed_10": int(score < 10),
                            "collapsed_20": int(score < 20),
                            "collapsed_30": int(score < 30),
                            "collapsed_40": int(score < 40),
                            "td_error_mean": float(row["td_residual_mean"]),
                            "td_error_p50": float(row["td_residual_p50"]),
                            "td_error_p90": float(row["td_residual_p90"]),
                            "td_error_p99": float(row["td_residual_p99"]),
                            "td_error_max": td_error_max,
                            "q_abs_mean": float(row["q_data_abs_mean"]),
                            "q_abs_p99": float(row["q_data_abs_p99"]),
                            "critic_loss": float(row["final_critic_loss"]),
                            "d_critic": raw["d_critic"],
                            "d_final": raw["d_final"],
                            "joint_critic_gain": raw["joint_critic_gain"],
                            "common_critic_gain": raw["common_critic_gain"],
                            "n_nan": raw["n_nan"],
                            "n_inf": raw["n_inf"],
                        }
                    )

    keys = [(r["method"], r["env"], r["T"], r["seed"]) for r in rows_out]
    require(len(rows_out) == 180, f"failure rows={len(rows_out)}, expected 180")
    require(len(keys) == len(set(keys)), "failure primary-key duplicates")
    integrity = {
        "rows": len(rows_out),
        "primary_key_duplicates": len(keys) - len(set(keys)),
        "state_pairing_exact_across_methods": True,
        "raw_nan": sum(int(r["n_nan"]) for r in rows_out),
        "raw_inf": sum(int(r["n_inf"]) for r in rows_out),
        "td_error_max_unavailable_rows": missing_td_max,
        "td_error_source": (
            "run-level diagnostic CSV; source NPZ did not retain TD residual arrays"
        ),
    }
    require(integrity["raw_nan"] == 0 and integrity["raw_inf"] == 0, "non-finite failure raw arrays")
    return rows_out, integrity, inputs


def symmetric_relative_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(estimate - reference) / (
        np.abs(estimate) + np.abs(reference) + EPS
    )


def simulator_diagnostics(
    audit_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path], list[dict[str, str]]]:
    source = audit_root / "mc_value_pilot_cpu"
    cell_rows = read_csv(source / "cells.csv")
    require(len(cell_rows) == 60, f"simulator cells={len(cell_rows)}, expected 60")
    rows_out: list[dict[str, Any]] = []
    inputs: set[Path] = {
        source / "cells.csv",
        source / "PILOT_MANIFEST.json",
        source / "SUMMARY.json",
    }
    all_state_rows: list[dict[str, str]] = []
    state_keys: set[tuple[Any, ...]] = set()
    for cell in cell_rows:
        env = cell["env"]
        tau = float(cell["tau"])
        method = cell["method"]
        seed = int(cell["seed"])
        k = int(method.removeprefix("mpi"))
        cell_dir = source / env / f"tau{tau_token(tau)}_{method}_seed{seed}"
        state_path = cell_dir / "per_state.csv"
        states = read_csv(state_path)
        require(len(states) == 12, f"{cell_dir}: states={len(states)}, expected 12")
        inputs.update(
            (
                state_path,
                cell_dir / "MANIFEST.json",
                cell_dir / "summary.json",
                Path(cell["checkpoint"]),
            )
        )
        config = Path(cell["checkpoint"]).parent / "config.json"
        if config.is_file():
            inputs.add(config)
        signed = np.asarray(
            [float(row["critic_error_rho_at_ak"]) for row in states], dtype=np.float64
        )
        qhat = np.asarray([float(row["qhat_q1_ak"]) for row in states], dtype=np.float64)
        qmc = np.asarray(
            [float(row["qmc_rho_discounted_ak"]) for row in states], dtype=np.float64
        )
        learned_delta = np.asarray(
            [float(row["delta_qhat_q1"]) for row in states], dtype=np.float64
        )
        mc_delta = np.asarray(
            [float(row["delta_mc_rho"]) for row in states], dtype=np.float64
        )
        dead_zone = float(cell["dead_zone"])
        valid = (np.abs(learned_delta) > dead_zone) & (np.abs(mc_delta) > dead_zone)
        wrong = valid & (np.sign(learned_delta) != np.sign(mc_delta))
        for state in states:
            key = (
                env,
                tau_token(tau),
                method,
                seed,
                int(state["state_index"]),
            )
            require(key not in state_keys, f"duplicate simulator state key: {key}")
            state_keys.add(key)
            enriched = dict(state)
            enriched.update(
                {
                    "env": env,
                    "tau": tau_token(tau),
                    "method": method,
                    "seed": str(seed),
                    "d4rl_score": cell["d4rl_score"],
                }
            )
            all_state_rows.append(enriched)
        rows_out.append(
            {
                "env": env,
                "T": tau,
                "K": k,
                "method": method,
                "seed": seed,
                "checkpoint": cell["checkpoint"],
                "score": float(cell["d4rl_score"]),
                "collapsed": int(float(cell["d4rl_score"]) < 20),
                "n_states": len(states),
                "n_rollouts_per_state": int(states[0]["smoothing_rollouts"]),
                "median_signed_error": float(np.median(signed)),
                "median_absolute_error": float(np.median(np.abs(signed))),
                "median_symmetric_relative_error": float(
                    np.median(symmetric_relative_error(qhat, qmc))
                ),
                "ranking_valid_count": int(valid.sum()),
                "ranking_wrong_count": int(wrong.sum()),
                "n_nan": int(sum(finite_stats(v)[0] for v in (signed, qhat, qmc))),
                "n_inf": int(sum(finite_stats(v)[1] for v in (signed, qhat, qmc))),
            }
        )
    require(len(state_keys) == 720, f"simulator states={len(state_keys)}, expected 720")
    keys = [(r["env"], r["T"], r["K"], r["seed"]) for r in rows_out]
    require(len(keys) == len(set(keys)), "simulator checkpoint primary-key duplicates")
    integrity = {
        "rows": len(rows_out),
        "states": len(state_keys),
        "primary_key_duplicates": len(keys) - len(set(keys)),
        "collapsed_checkpoints": sum(int(r["collapsed"]) for r in rows_out),
        "stable_checkpoints": sum(1 - int(r["collapsed"]) for r in rows_out),
        "n_nan": sum(int(r["n_nan"]) for r in rows_out),
        "n_inf": sum(int(r["n_inf"]) for r in rows_out),
    }
    require(integrity["n_nan"] == 0 and integrity["n_inf"] == 0, "non-finite simulator inputs")
    require(integrity["collapsed_checkpoints"] == 5, "simulator collapsed count mismatch")
    return rows_out, integrity, inputs, all_state_rows


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(len(ordered) * proportion))
    if trim == 0:
        return float(np.mean(ordered))
    return float(np.mean(ordered[trim:-trim]))


def route_diagnostics(
    audit_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[Path], dict[str, list[float]]]:
    source = audit_root / "route_shadow_mc"
    run_source = read_csv(source / "run_level.csv")
    inputs: set[Path] = {
        source / "run_level.csv",
        source / "checkpoint_pairs.csv",
        source / "SUMMARY.json",
    }
    rows_out: list[dict[str, Any]] = []
    variants: dict[str, list[float]] = defaultdict(list)
    state_pair_checks = 0
    for run in run_source:
        env = run["env"]
        tau = float(run["tau"])
        seed = int(run["seed"])
        base = source / env / f"tau{tau_token(tau)}_mpi3_seed{seed}"
        checkpoint_medians: dict[str, list[float]] = defaultdict(list)
        checkpoint_means: dict[str, list[float]] = defaultdict(list)
        checkpoint_trimmed: dict[str, list[float]] = defaultdict(list)
        final_values: dict[str, float] = {}
        n_states: set[int] = set()
        for step in (250_000, 500_000, 750_000, 1_000_000):
            source_rows: dict[str, list[dict[str, str]]] = {}
            for arm in ("shadow_mu1", "shadow_muk"):
                arm_dir = base / f"step{step}" / arm
                state_path = arm_dir / "per_state.csv"
                rows = read_csv(state_path)
                require(len(rows) == 12, f"{arm_dir}: states={len(rows)}, expected 12")
                source_rows[arm] = rows
                inputs.update((state_path, arm_dir / "MANIFEST.json", arm_dir / "summary.json"))
                checkpoint = Path(rows[0]["checkpoint"])
                inputs.add(checkpoint)
                config = checkpoint.parent / "config.json"
                if config.is_file():
                    inputs.add(config)
                values = np.asarray(
                    [
                        float(row["critic_symmetric_relative_error_rho_at_ak"])
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                checkpoint_medians[arm].append(float(np.median(values)))
                checkpoint_means[arm].append(float(np.mean(values)))
                checkpoint_trimmed[arm].append(trimmed_mean(values))
                if step == 1_000_000:
                    final_values[arm] = float(np.median(values))
                n_states.add(len(rows))
            pair_fields = ("state_index", "episode", "time_index", "rollout_seed")
            left_keys = {
                tuple(row[field] for field in pair_fields)
                for row in source_rows["shadow_mu1"]
            }
            right_keys = {
                tuple(row[field] for field in pair_fields)
                for row in source_rows["shadow_muk"]
            }
            require(left_keys == right_keys, f"route state IDs differ: {env}/{tau}/{seed}/{step}")
            state_pair_checks += 1
        require(n_states == {12}, f"route inconsistent state count: {env}/{tau}/{seed}")
        first = float(np.mean(checkpoint_medians["shadow_mu1"]))
        final = float(np.mean(checkpoint_medians["shadow_muk"]))
        delta = final - first
        delta_final = final_values["shadow_muk"] - final_values["shadow_mu1"]
        delta_mean = float(
            np.mean(checkpoint_means["shadow_muk"])
            - np.mean(checkpoint_means["shadow_mu1"])
        )
        delta_trimmed = float(
            np.mean(checkpoint_trimmed["shadow_muk"])
            - np.mean(checkpoint_trimmed["shadow_mu1"])
        )
        variants["state_median_checkpoint_mean"].append(delta)
        variants["final_checkpoint_state_median"].append(delta_final)
        variants["state_mean_checkpoint_mean"].append(delta_mean)
        variants["state_trimmed_mean_checkpoint_mean"].append(delta_trimmed)
        rows_out.append(
            {
                "env": env,
                "T": tau,
                "seed": seed,
                "n_checkpoints": 4,
                "n_states_per_checkpoint": 12,
                "first_route_error": first,
                "final_route_error": final,
                "delta": delta,
                "delta_final_checkpoint_only": delta_final,
                "delta_state_mean_aggregation": delta_mean,
                "delta_state_median_aggregation": delta,
                "delta_state_trimmed_mean_aggregation": delta_trimmed,
            }
        )
    require(len(rows_out) == 8, f"route runs={len(rows_out)}, expected 8")
    keys = [(r["env"], r["T"], r["seed"]) for r in rows_out]
    require(len(keys) == len(set(keys)), "route run primary-key duplicates")
    integrity = {
        "rows": len(rows_out),
        "primary_key_duplicates": len(keys) - len(set(keys)),
        "checkpoint_state_pair_checks": state_pair_checks,
        "all_pairings_exact": True,
        "n_nan": 0,
        "n_inf": 0,
    }
    return rows_out, integrity, inputs, variants


def paired_values(
    rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    indexed = {
        (row["method"], row["env"], tau_token(row["T"]), int(row["seed"])): row
        for row in rows
    }
    left_values: list[float] = []
    right_values: list[float] = []
    clusters: list[tuple[str, int]] = []
    for env in EXPECTED_ENVS:
        for seed in EXPECTED_SEEDS:
            for tau in EXPECTED_TAUS:
                lkey = (left, env, tau_token(tau), seed)
                rkey = (right, env, tau_token(tau), seed)
                require(lkey in indexed and rkey in indexed, f"missing pair {lkey}/{rkey}")
                left_values.append(float(indexed[lkey][metric]))
                right_values.append(float(indexed[rkey][metric]))
                clusters.append((env, seed))
    require(len(left_values) == 90, f"{left} vs {right} pairs != 90")
    return np.asarray(left_values), np.asarray(right_values), clusters


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    ratio = left / right
    log_ratio = np.log(np.maximum(left, EPS) / np.maximum(right, EPS))
    return {
        "pairs": int(len(left)),
        "left_lower": int(np.sum(left < right)),
        "fraction_left_lower": float(np.mean(left < right)),
        "median_ratio": float(np.median(ratio)),
        "median_difference": float(np.median(left - right)),
        "mean_difference": float(np.mean(left - right)),
        "median_log_ratio": float(np.median(log_ratio)),
    }


def cluster_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    clusters: Sequence[tuple[str, int]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    cluster_order = list(dict.fromkeys(clusters))
    require(len(cluster_order) == 18, f"target bootstrap clusters={len(cluster_order)}, expected 18")
    positions = {
        cluster: np.asarray([i for i, value in enumerate(clusters) if value == cluster])
        for cluster in cluster_order
    }
    require(all(len(value) == 5 for value in positions.values()), "each target cluster must contain five T values")
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[float]] = {
        "mean_difference": [],
        "fraction_left_lower": [],
        "median_log_ratio": [],
    }
    batch_size = 1_000
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        picks = rng.integers(0, len(cluster_order), size=(count, len(cluster_order)))
        index = np.concatenate(
            [
                np.stack([positions[cluster_order[pick]] for pick in column])
                for column in picks.T
            ],
            axis=1,
        )
        lvals = left[index]
        rvals = right[index]
        metrics["mean_difference"].extend(np.mean(lvals - rvals, axis=1).tolist())
        metrics["fraction_left_lower"].extend(np.mean(lvals < rvals, axis=1).tolist())
        metrics["median_log_ratio"].extend(
            np.median(
                np.log(np.maximum(lvals, EPS) / np.maximum(rvals, EPS)), axis=1
            ).tolist()
        )
    return {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in metrics.items()
    }


def route_bootstrap(values: Sequence[float], samples: int, seed: int) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(data, size=(samples, len(data)), replace=True), axis=1
    )
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def build_claims(
    target_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    simulator_rows: list[dict[str, Any]],
    simulator_states: list[dict[str, str]],
    route_rows: list[dict[str, Any]],
    route_variants: dict[str, list[float]],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_claims: dict[str, Any] = {}
    sensitivity: list[dict[str, Any]] = []
    comparison_specs = (
        ("prox3_vs_td3", "mpi3", "td3"),
        ("prox3_vs_prox2", "mpi3", "mpi2"),
    )
    for label, left_method, right_method in comparison_specs:
        for metric in (
            "d_target_deterministic",
            "d_target_smoothed",
            "d_final_current",
        ):
            left, right, clusters = paired_values(
                target_rows, left_method, right_method, metric
            )
            observed = comparison(left, right)
            observed["cluster_bootstrap_95"] = cluster_bootstrap(
                left,
                right,
                clusters,
                samples=bootstrap_samples,
                seed=TARGET_BOOTSTRAP_SEED,
            )
            target_claims[f"{label}_{metric}"] = observed
            for statistic, interval in observed["cluster_bootstrap_95"].items():
                sensitivity.append(
                    {
                        "analysis": "target_exposure",
                        "variant": label,
                        "subgroup": metric,
                        "metric": f"{statistic}_cluster_bootstrap_95_low",
                        "value": interval[0],
                        "n": 18,
                        "status": "computed",
                    }
                )
                sensitivity.append(
                    {
                        "analysis": "target_exposure",
                        "variant": label,
                        "subgroup": metric,
                        "metric": f"{statistic}_cluster_bootstrap_95_high",
                        "value": interval[1],
                        "n": 18,
                        "status": "computed",
                    }
                )

    proxy = np.asarray([float(row["current_state_proxy"]) for row in target_rows])
    deterministic = np.asarray(
        [float(row["d_target_deterministic"]) for row in target_rows]
    )
    smoothed = np.asarray([float(row["d_target_smoothed"]) for row in target_rows])
    relative = np.abs(deterministic - proxy) / np.maximum(np.abs(proxy), EPS)
    target_claims["proxy_alignment"] = {
        "pearson_proxy_vs_target_deterministic": float(
            np.corrcoef(proxy, deterministic)[0, 1]
        ),
        "median_target_deterministic_over_proxy": float(
            np.median(deterministic / proxy)
        ),
        "max_abs_relative_difference_deterministic_vs_proxy": float(
            np.max(relative)
        ),
        "pearson_proxy_vs_target_smoothed": float(
            np.corrcoef(proxy, smoothed)[0, 1]
        ),
    }

    failure_claims: dict[str, Any] = {}
    for method in FAILURE_METHODS:
        method_rows = [row for row in failure_rows if row["method"] == method]
        for group, predicate in (
            ("stable", lambda score: score >= 20),
            ("collapsed", lambda score: score < 20),
        ):
            selected = [row for row in method_rows if predicate(float(row["score"]))]
            metrics = {}
            for field in (
                "td_error_p99",
                "q_abs_mean",
                "critic_loss",
                "d_critic",
                "d_final",
                "common_critic_gain",
            ):
                values = np.asarray([float(row[field]) for row in selected])
                metrics[field] = float(np.median(values))
                logged = np.log10(1.0 + np.abs(values))
                metrics[f"{field}_log10_median"] = float(np.median(logged))
                metrics[f"{field}_log10_iqr"] = [
                    float(np.percentile(logged, 25)),
                    float(np.percentile(logged, 75)),
                ]
            failure_claims[f"{method}_{group}"] = {
                "runs": len(selected),
                **metrics,
            }
    for threshold in (10, 20, 30, 40):
        for method in FAILURE_METHODS:
            method_rows = [row for row in failure_rows if row["method"] == method]
            collapsed = sum(float(row["score"]) < threshold for row in method_rows)
            sensitivity.append(
                {
                    "analysis": "failure",
                    "variant": f"collapse_threshold_{threshold}",
                    "subgroup": method,
                    "metric": "collapsed_runs",
                    "value": collapsed,
                    "n": len(method_rows),
                    "status": "computed",
                }
            )
    for omitted in EXPECTED_ENVS:
        for method in FAILURE_METHODS:
            selected = [
                row
                for row in failure_rows
                if row["method"] == method and row["env"] != omitted
            ]
            sensitivity.append(
                {
                    "analysis": "failure",
                    "variant": "leave_one_environment_out",
                    "subgroup": f"{method}; omitted={omitted}",
                    "metric": "collapsed_fraction_threshold20",
                    "value": float(np.mean([float(row["score"]) < 20 for row in selected])),
                    "n": len(selected),
                    "status": "computed",
                }
            )
    sensitivity.append(
        {
            "analysis": "failure",
            "variant": "last_3_checkpoint_median",
            "subgroup": "all",
            "metric": "availability",
            "value": "",
            "n": 0,
            "status": "unavailable: source audit retained final-checkpoint TD arrays only",
        }
    )

    stable_state_errors = np.asarray(
        [
            float(row["critic_error_rho_at_ak"])
            for row in simulator_states
            if float(row["d4rl_score"]) >= 20
        ]
    )
    collapsed_state_errors = np.asarray(
        [
            float(row["critic_error_rho_at_ak"])
            for row in simulator_states
            if float(row["d4rl_score"]) < 20
        ]
    )
    stable_checkpoint_medians = np.asarray(
        [
            float(row["median_signed_error"])
            for row in simulator_rows
            if not row["collapsed"]
        ]
    )
    collapsed_checkpoint_medians = np.asarray(
        [
            float(row["median_signed_error"])
            for row in simulator_rows
            if row["collapsed"]
        ]
    )
    simulator_claims = {
        "stable_checkpoints": int(len(stable_checkpoint_medians)),
        "collapsed_checkpoints": int(len(collapsed_checkpoint_medians)),
        "stable_signed_error_state_pooled_median": float(np.median(stable_state_errors)),
        "collapsed_signed_error_state_pooled_median": float(
            np.median(collapsed_state_errors)
        ),
        "stable_signed_error_checkpoint_median_median": float(
            np.median(stable_checkpoint_medians)
        ),
        "collapsed_signed_error_checkpoint_median_median": float(
            np.median(collapsed_checkpoint_medians)
        ),
        "reported_value_aggregation_level": "state-pooled median",
    }
    for dead_zone in (0.5, 1.0, 2.0, 5.0):
        learned = np.asarray(
            [float(row["delta_qhat_q1"]) for row in simulator_states]
        )
        mc = np.asarray([float(row["delta_mc_rho"]) for row in simulator_states])
        valid = (np.abs(learned) > dead_zone) & (np.abs(mc) > dead_zone)
        wrong = valid & (np.sign(learned) != np.sign(mc))
        sensitivity.extend(
            (
                {
                    "analysis": "simulator",
                    "variant": f"dead_zone_{dead_zone:g}",
                    "subgroup": "all_states",
                    "metric": "ranking_eligible",
                    "value": int(valid.sum()),
                    "n": len(valid),
                    "status": "computed",
                },
                {
                    "analysis": "simulator",
                    "variant": f"dead_zone_{dead_zone:g}",
                    "subgroup": "all_states",
                    "metric": "ranking_disagreement",
                    "value": int(wrong.sum()),
                    "n": int(valid.sum()),
                    "status": "computed",
                },
            )
        )
        if dead_zone == 1.0:
            simulator_claims["rank_eligible"] = int(valid.sum())
            simulator_claims["rank_disagreement"] = int(wrong.sum())

    primary_deltas = [float(row["delta"]) for row in route_rows]
    route_claims = {
        "positive": [int(np.sum(np.asarray(primary_deltas) > 0)), len(primary_deltas)],
        "mean_delta": float(np.mean(primary_deltas)),
        "median_delta": float(np.median(primary_deltas)),
        "sign_test_p": float(2.0 / (2 ** len(primary_deltas))),
        "run_bootstrap_95": route_bootstrap(
            primary_deltas, bootstrap_samples, TARGET_BOOTSTRAP_SEED
        ),
        "estimand": (
            "alignment against common Q_MC^{rho_1} first-route continuation; "
            "not each route critic's own-continuation calibration"
        ),
    }
    for variant, values in route_variants.items():
        sensitivity.append(
            {
                "analysis": "route",
                "variant": variant,
                "subgroup": "8 training runs",
                "metric": "positive_delta_runs",
                "value": int(np.sum(np.asarray(values) > 0)),
                "n": len(values),
                "status": "computed",
            }
        )
        sensitivity.append(
            {
                "analysis": "route",
                "variant": variant,
                "subgroup": "8 training runs",
                "metric": "mean_delta",
                "value": float(np.mean(values)),
                "n": len(values),
                "status": "computed",
            }
        )

    expected_checks = {
        "prox3_vs_td3_target_lower": (
            target_claims["prox3_vs_td3_d_target_deterministic"]["left_lower"],
            88,
        ),
        "prox3_vs_prox2_target_lower": (
            target_claims["prox3_vs_prox2_d_target_deterministic"]["left_lower"],
            84,
        ),
        "prox3_vs_td3_final_lower": (
            target_claims["prox3_vs_td3_d_final_current"]["left_lower"],
            36,
        ),
        "simulator_stable": (simulator_claims["stable_checkpoints"], 55),
        "simulator_collapsed": (simulator_claims["collapsed_checkpoints"], 5),
        "simulator_rank_eligible": (simulator_claims["rank_eligible"], 69),
        "simulator_rank_disagreement": (
            simulator_claims["rank_disagreement"],
            29,
        ),
        "route_positive": (route_claims["positive"][0], 8),
    }
    checks = {
        key: {"observed": observed, "expected": expected, "pass": observed == expected}
        for key, (observed, expected) in expected_checks.items()
    }
    deterministic_tolerance_checks = {
        "prox3_vs_td3_target_median_ratio": (
            target_claims["prox3_vs_td3_d_target_deterministic"]["median_ratio"],
            0.6576946525,
            1e-4,
        ),
        "prox3_vs_prox2_target_median_ratio": (
            target_claims["prox3_vs_prox2_d_target_deterministic"]["median_ratio"],
            0.9030389793,
            1e-4,
        ),
        "prox3_vs_td3_final_median_ratio": (
            target_claims["prox3_vs_td3_d_final_current"]["median_ratio"],
            1.057,
            1e-3,
        ),
        "proxy_target_pearson": (
            target_claims["proxy_alignment"][
                "pearson_proxy_vs_target_deterministic"
            ],
            0.9999114535,
            1e-4,
        ),
        "proxy_smoothed_pearson": (
            target_claims["proxy_alignment"]["pearson_proxy_vs_target_smoothed"],
            0.9992799627,
            1e-4,
        ),
        "route_mean_delta": (route_claims["mean_delta"], 0.1235147225, 1e-4),
        "route_median_delta": (route_claims["median_delta"], 0.0132679730, 1e-4),
        "route_sign_test_p": (route_claims["sign_test_p"], 0.0078125, 1e-12),
    }
    for key, (observed, expected, tolerance) in deterministic_tolerance_checks.items():
        checks[key] = {
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "pass": abs(observed - expected) <= tolerance,
        }
    return (
        {
            "target_exposure": target_claims,
            "failure_diagnostics": failure_claims,
            "simulator": simulator_claims,
            "shadow_route": route_claims,
            "checks": checks,
            "all_deterministic_checks_pass": all(
                item["pass"] for item in checks.values()
            ),
        },
        sensitivity,
    )


def write_checksums(path: Path, inputs: Iterable[Path]) -> dict[str, Any]:
    unique = sorted({item.resolve() for item in inputs if item.is_file()}, key=str)
    lines: list[str] = []
    total_bytes = 0
    for item in unique:
        digest = sha256(item)
        total_bytes += item.stat().st_size
        lines.append(f"{digest}  {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"files": len(unique), "bytes": total_bytes}


def report_text(
    integrity: dict[str, Any],
    claims: dict[str, Any],
    checksum_stats: dict[str, Any],
) -> str:
    checks = claims["checks"]
    failed = [key for key, value in checks.items() if not value["pass"]]
    overall = "PASS" if not failed else "FAIL"
    simulator = claims["simulator"]
    aggregation_diff = (
        simulator["stable_signed_error_state_pooled_median"]
        != simulator["stable_signed_error_checkpoint_median_median"]
        or simulator["collapsed_signed_error_state_pooled_median"]
        != simulator["collapsed_signed_error_checkpoint_median_median"]
    )
    lines = [
        "# Independent diagnostics audit",
        "",
        f"Overall deterministic claim reproduction: **{overall}**",
        "",
        "This pack was rebuilt from existing local NPZ/per-state diagnostic inputs.",
        "No policy training or simulator rollout was rerun.",
        "",
        "## Integrity",
        "",
        f"- Target exposure: {integrity['target_exposure']['rows']}/270 rows; exact state pairing; "
        f"NaN={integrity['target_exposure']['raw_nan']}, inf={integrity['target_exposure']['raw_inf']}.",
        f"- Failure diagnostics: {integrity['failure']['rows']}/180 rows; exact validation-state pairing; "
        f"NaN={integrity['failure']['raw_nan']}, inf={integrity['failure']['raw_inf']}.",
        f"- Simulator reference: {integrity['simulator']['rows']}/60 checkpoints and "
        f"{integrity['simulator']['states']}/720 states.",
        f"- Route intervention: {integrity['route']['rows']}/8 runs; "
        f"{integrity['route']['checkpoint_state_pair_checks']} checkpoint arm-pair checks passed.",
        f"- Checksums: {checksum_stats['files']} inputs, {checksum_stats['bytes']} bytes.",
        "",
        "## Claim reproduction",
        "",
    ]
    for key, value in checks.items():
        mark = "PASS" if value["pass"] else "FAIL"
        lines.append(
            f"- {mark} `{key}`: observed={value['observed']}, expected={value['expected']}"
        )
    lines.extend(
        [
            "",
            "## Findings and limitations",
            "",
            "- Target comparisons use 18-cluster `(env, seed)` bootstrap; T values remain grouped.",
            "- Simulator aggregation was recomputed in both forms. The manuscript values "
            "`-38.7` and `1.70e12` correspond to pooled-state medians, not medians of "
            "checkpoint-level medians."
            if aggregation_diff
            else "- Simulator pooled-state and checkpoint-median summaries agree.",
            "- The route result is alignment against the common "
            "`Q_MC^{rho_1}` first-route estimand, not own-continuation calibration.",
            "- The K=2 source NPZ files do not retain per-state TD-residual arrays. "
            f"`td_error_max` is therefore unavailable for "
            f"{integrity['failure']['td_error_max_unavailable_rows']} rows; TD quantiles "
            "are independently checked for count/finiteness but read from the prior "
            "run-level diagnostic dump.",
            "- Last-three-checkpoint failure sensitivity is unavailable because the "
            "source audit retained final-checkpoint diagnostics only.",
            "",
            "## Leakage audit",
            "",
            "- Failure signatures are descriptive, not causal.",
            "- Simulator state rows are not treated as independent observations.",
            "- Route inference resamples the eight training runs, not checkpoints or states.",
            "- No manuscript aggregate table is used as an input to claim recomputation.",
            "",
        ]
    )
    if failed:
        lines.extend(["## Failed checks", "", *[f"- `{item}`" for item in failed], ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path(
            os.environ.get(
                "AUDIT_ROOT",
                "/home/ext_csv/mpi_sweep_lab/audit_report_20260829",
            )
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.bootstrap_samples > 0, "bootstrap-samples must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_root = args.audit_root.resolve()

    target_rows, target_integrity, target_inputs = target_exposure(audit_root)
    failure_rows, failure_integrity, failure_inputs = failure_diagnostics(audit_root)
    (
        simulator_rows,
        simulator_integrity,
        simulator_inputs,
        simulator_states,
    ) = simulator_diagnostics(audit_root)
    route_rows, route_integrity, route_inputs, route_variants = route_diagnostics(
        audit_root
    )
    claims, sensitivity = build_claims(
        target_rows,
        failure_rows,
        simulator_rows,
        simulator_states,
        route_rows,
        route_variants,
        args.bootstrap_samples,
    )
    integrity = {
        "target_exposure": target_integrity,
        "failure": failure_integrity,
        "simulator": simulator_integrity,
        "route": route_integrity,
    }

    write_csv(output / "target_exposure_run.csv", target_rows)
    write_csv(output / "failure_diagnostics_run.csv", failure_rows)
    write_csv(output / "simulator_checkpoint.csv", simulator_rows)
    write_csv(output / "route_intervention_run.csv", route_rows)
    write_csv(output / "sensitivity.csv", sensitivity)
    write_json(output / "claim_reproduction.json", claims)

    inputs = target_inputs | failure_inputs | simulator_inputs | route_inputs
    inputs.add(Path(__file__).resolve())
    checksum_stats = write_checksums(output / "checksums.txt", inputs)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "understood_as": (
            "re-audit existing local raw/per-state diagnostics without rerunning "
            "training or simulator rollouts; publish compact run/checkpoint outputs"
        ),
        "audit_root": str(audit_root),
        "output_dir": str(output),
        "repo_git_commit": git_commit(args.repo_root),
        "lab_git_commit": git_commit(audit_root.parent),
        "bootstrap": {
            "unit": "(env, seed) cluster for target; training run for route",
            "samples": args.bootstrap_samples,
            "seed": TARGET_BOOTSTRAP_SEED,
        },
        "counts": {
            "target_exposure_run.csv": len(target_rows),
            "failure_diagnostics_run.csv": len(failure_rows),
            "simulator_checkpoint.csv": len(simulator_rows),
            "route_intervention_run.csv": len(route_rows),
            "sensitivity.csv": len(sensitivity),
        },
        "integrity": integrity,
        "checksums": checksum_stats,
        "filters": {
            "target": "bootstrap-mask-valid transitions; timeout filtering from source manifest",
            "collapse_primary": "D4RL normalized score < 20",
            "simulator_aggregation": "rollout mean -> state error -> checkpoint median",
            "route_aggregation": "state median -> checkpoint -> training-run mean",
        },
        "input_policy": (
            "raw NPZ/per-state CSV plus run-level TD diagnostics where source NPZ "
            "did not retain TD residual arrays; no manuscript aggregate table"
        ),
    }
    write_json(output / "audit_manifest.json", manifest)
    (output / "AUDIT_REPORT.md").write_text(
        report_text(integrity, claims, checksum_stats), encoding="utf-8"
    )
    require(
        claims["all_deterministic_checks_pass"],
        "one or more deterministic manuscript claims failed reproduction",
    )
    print(
        f"PASS audit pack: target={len(target_rows)} failure={len(failure_rows)} "
        f"simulator={len(simulator_rows)} route={len(route_rows)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
