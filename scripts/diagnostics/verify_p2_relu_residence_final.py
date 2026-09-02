#!/usr/bin/env python3
"""Independent full-recomputation verifier for the P2 ReLU residence FINAL audit.

This module intentionally does not import ``run_p2_relu_residence``.  It reuses
the pilot verifier's independent NumPy geometry/oracle code (which is itself
independent of the runner) and adds the final 12-bundle protocol, exclusion,
per-run, and aggregate-survival checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_p2_relu_residence as vp  # noqa: E402


FINAL_PROTOCOL = "p2_relu_residence_final_v2"
FINAL_RAW_STATUS = "analysis_complete_pending_verification"
FINAL_N_STATES = 512
FINAL_SEEDS = (0, 1)
FINAL_TAU = 1.0
FINAL_EXCLUDED_FAMILY = "halfcheetah"
FINAL_SAMPLE_SEED_BASE = 20260903
FINAL_ENVIRONMENTS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
KNOWN_DEV_BUNDLE_IDS = (
    "VnOjA4",
    "InTWgf",
    "OLDBdx",
    "xlMNfc",
    "sqfIVj",
    "meAdJI",
)
CHECKPOINT_STEP = 1_000_000
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

TIMES = vp.TIMES
SUBSTEPS = vp.SUBSTEPS
CATEGORIES = vp.CATEGORIES
ACTION_ELIGIBILITY = vp.ACTION_ELIGIBILITY
BOX = vp.BOX
SOURCE_SNAPSHOT_DIR = vp.SOURCE_SNAPSHOT_DIR

sha256_file = vp.sha256_file
sha256_array = vp.sha256_array
sha256_bytes = vp.sha256_bytes
canonical_bytes = vp.canonical_bytes
write_json_create = vp.write_json_create
now_iso = vp.now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="recompute read-only and require the existing VERIFY.json to agree",
    )
    return parser.parse_args()


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


def applied_exclusions_for_env(
    inventory: list[dict[str, Any]], env: str
) -> tuple[set[int], list[dict[str, Any]]]:
    excluded: set[int] = set()
    applied: list[dict[str, Any]] = []
    for record in inventory:
        if record.get("env") != env:
            continue
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"exclusion file hash mismatch: {path}")
        values = np.asarray(np.load(path), dtype=np.int64)
        if len(values) != record["n_indices"]:
            raise AssertionError(f"exclusion count mismatch: {path}")
        excluded.update(int(value) for value in values)
        applied.append(record)
    return excluded, applied


def verify_design(out_dir: Path) -> dict[str, Any]:
    design = json.loads((out_dir / "FINAL_DESIGN_LOCK.json").read_text())
    without_hash = dict(design)
    claimed = without_hash.pop("design_sha256")
    if sha256_bytes(canonical_bytes(without_hash)) != claimed:
        raise AssertionError("FINAL_DESIGN_LOCK canonical hash mismatch")
    if design.get("protocol") != FINAL_PROTOCOL:
        raise AssertionError("final protocol mismatch")
    if design.get("learned_slope") is not None or design.get("coverage_gate") is not None:
        raise AssertionError("final design contains a forbidden learned outcome gate")
    estimand = design.get("estimand", {})
    if estimand.get("learned_slope") is not None:
        raise AssertionError("learned slope must be absent")
    if estimand.get("learned_support_gate") is not None:
        raise AssertionError("learned support gate must be absent")
    if estimand.get("unique_horizon_estimand") != "empirical survival curve P(t_exit > t)":
        raise AssertionError("unique-horizon estimand missing")
    if "never independent evidence" not in estimand.get("cell_dependence", ""):
        raise AssertionError("repeated-cell dependence contract missing")
    if design.get("excluded_family") != FINAL_EXCLUDED_FAMILY:
        raise AssertionError("final excluded family mismatch")
    if design.get("n_runs") != len(FINAL_ENVIRONMENTS) * len(FINAL_SEEDS):
        raise AssertionError("final n_runs mismatch")
    if design.get("n_states_per_run") != FINAL_N_STATES:
        raise AssertionError("final n_states_per_run mismatch")
    if list(design.get("environments", [])) != list(FINAL_ENVIRONMENTS):
        raise AssertionError("final environments mismatch")
    if any(env.startswith(f"{FINAL_EXCLUDED_FAMILY}-") for env in design["environments"]):
        raise AssertionError("final environments must exclude HalfCheetah")
    if list(design.get("seeds", [])) != list(FINAL_SEEDS):
        raise AssertionError("final seeds mismatch")
    if design.get("primary_summary") != "unique-horizon empirical survival curve P(t_exit > t)":
        raise AssertionError("final primary-summary contract missing")
    if "do not count as independent evidence" not in design.get("repeated_cell_policy", ""):
        raise AssertionError("final repeated-cell policy missing")
    if design.get("indices_created_only_after_final_protocol_freeze") is not True:
        raise AssertionError("final freeze-before-sampling declaration missing")
    if design.get("exclude_all_archived_and_development_attempt_indices") is not True:
        raise AssertionError("final exclude-all declaration missing")
    if int(design.get("sample_seed_base")) != FINAL_SAMPLE_SEED_BASE:
        raise AssertionError("final sample seed base mismatch")
    if design.get("checkpoint_config_contract") != FINAL_REQUIRED_FIXED_CONFIG:
        raise AssertionError("final checkpoint config contract mismatch")
    expected_seeds = {
        f"{env}_seed{seed}": final_sample_seed(env, seed)
        for env in FINAL_ENVIRONMENTS
        for seed in FINAL_SEEDS
    }
    if design.get("per_run_sample_seeds") != expected_seeds:
        raise AssertionError("per-run sample seed map mismatch")
    return design


def verify_exclusion_inventory(design: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = design["exclusion_inventory"]
    if design.get("exclusion_inventory_n_files") != len(inventory):
        raise AssertionError("exclusion inventory count mismatch")
    present_dev = {
        record["bundle_id"]
        for record in inventory
        if record["role"] == "dev_pilot_bundle"
    }
    missing = [bundle for bundle in KNOWN_DEV_BUNDLE_IDS if bundle not in present_dev]
    if missing:
        raise AssertionError(f"exclusion inventory missing dev bundles: {missing}")
    archived_by_env: dict[str, set[int]] = {}
    for record in inventory:
        path = Path(record["path"])
        if not path.is_file():
            raise AssertionError(f"exclusion file missing: {path}")
        if sha256_file(path) != record["sha256"]:
            raise AssertionError(f"exclusion file hash mismatch: {path}")
        values = np.asarray(np.load(path), dtype=np.int64)
        if len(values) != record["n_indices"]:
            raise AssertionError(f"exclusion n_indices mismatch: {path}")
        if record["role"] == "archived_fixed_operator":
            archived_by_env.setdefault(record["env"], set()).add(int(record["seed"]))
    for env in FINAL_ENVIRONMENTS:
        if archived_by_env.get(env) != set(FINAL_SEEDS):
            raise AssertionError(f"archived seed0/seed1 exclusions missing for {env}")
    return inventory


def verify_input_inventory(design: dict[str, Any]) -> None:
    inventory = design["input_inventory"]
    checkpoints = {
        (row["env"], int(row["seed"])): row for row in inventory["checkpoints"]
    }
    if set(checkpoints) != {
        (env, seed) for env in FINAL_ENVIRONMENTS for seed in FINAL_SEEDS
    }:
        raise AssertionError("input inventory checkpoint grid mismatch")
    for row in inventory["checkpoints"]:
        if sha256_file(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise AssertionError(f"input checkpoint hash mismatch: {row['checkpoint']}")
        if sha256_file(Path(row["config"])) != row["config_sha256"]:
            raise AssertionError(f"input config hash mismatch: {row['config']}")
    dataset_envs = {row["env"] for row in inventory["datasets"]}
    if dataset_envs != set(FINAL_ENVIRONMENTS):
        raise AssertionError("input inventory dataset grid mismatch")
    for row in inventory["datasets"]:
        if sha256_file(Path(row["dataset"])) != row["dataset_sha256"]:
            raise AssertionError(f"input dataset hash mismatch: {row['dataset']}")


def verify_final_git_provenance(provenance: Any) -> None:
    vp.verify_git_provenance(provenance)
    if provenance.get("git_dirty") is not False:
        raise AssertionError("final run did not record a fully clean worktree")
    if provenance.get("git_tracked_dirty") is not False:
        raise AssertionError("final run did not record a clean tracked worktree")
    if provenance.get("head_matches_origin_main") is not True:
        raise AssertionError("final run did not record HEAD == origin/main")
    if provenance.get("git_revision") != provenance.get("origin_main_revision"):
        raise AssertionError("final run revision/origin-main provenance mismatch")


def verify_run(
    out_dir: Path, design: dict[str, Any], env: str, seed: int, inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    sub = out_dir / f"{env}_seed{seed}"
    for name in (
        "state_indices.npy",
        "STATE_GEOMETRY.npz",
        "residence_cells.csv",
        "RUN_INPUTS.json",
        "RUN_SUMMARY.json",
    ):
        if not (sub / name).is_file():
            raise FileNotFoundError(sub / name)
    run_inputs = json.loads((sub / "RUN_INPUTS.json").read_text())
    run_summary = json.loads((sub / "RUN_SUMMARY.json").read_text())

    design_ckpt = {
        (row["env"], int(row["seed"])): row
        for row in design["input_inventory"]["checkpoints"]
    }[(env, seed)]
    checkpoint = Path(run_inputs["checkpoint"])
    config_path = Path(run_inputs["config"])
    dataset = Path(run_inputs["dataset"])
    if sha256_file(checkpoint) != design_ckpt["checkpoint_sha256"]:
        raise AssertionError(f"{env} seed{seed}: checkpoint drifted from design lock")
    if run_inputs["checkpoint_sha256"] != design_ckpt["checkpoint_sha256"]:
        raise AssertionError(f"{env} seed{seed}: RUN_INPUTS checkpoint hash mismatch")
    if sha256_file(config_path) != design_ckpt["config_sha256"]:
        raise AssertionError(f"{env} seed{seed}: config drifted from design lock")
    dataset_hash = {
        row["env"]: row["dataset_sha256"] for row in design["input_inventory"]["datasets"]
    }[env]
    if sha256_file(dataset) != dataset_hash:
        raise AssertionError(f"{env} seed{seed}: dataset drifted from design lock")
    if run_inputs["dataset_sha256"] != dataset_hash:
        raise AssertionError(f"{env} seed{seed}: RUN_INPUTS dataset hash mismatch")

    config = json.loads(config_path.read_text())
    required_config = {
        "env": env,
        "seed": seed,
        "tau": FINAL_TAU,
        **FINAL_REQUIRED_FIXED_CONFIG,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise AssertionError(f"{env} seed{seed}: config {key} mismatch")
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - declared trusted local checkpoint
    if int(payload.get("step", -1)) != CHECKPOINT_STEP:
        raise AssertionError(f"{env} seed{seed}: checkpoint step mismatch")
    if float(payload.get("max_action", math.nan)) != BOX:
        raise AssertionError(f"{env} seed{seed}: checkpoint max_action mismatch")

    observations, actions, mean, std = vp.load_rows(dataset)
    if sha256_array(np.asarray(payload["mean"], dtype=np.float32)) != run_inputs[
        "normalization_mean_sha256"
    ]:
        raise AssertionError(f"{env} seed{seed}: normalization mean fingerprint mismatch")
    if sha256_array(np.asarray(payload["std"], dtype=np.float32)) != run_inputs[
        "normalization_std_sha256"
    ]:
        raise AssertionError(f"{env} seed{seed}: normalization std fingerprint mismatch")
    if not np.allclose(mean, payload["mean"], rtol=0.0, atol=2e-5):
        raise AssertionError(f"{env} seed{seed}: recomputed normalization mean mismatch")
    if not np.allclose(std, payload["std"], rtol=0.0, atol=2e-5):
        raise AssertionError(f"{env} seed{seed}: recomputed normalization std mismatch")

    excluded, applied = applied_exclusions_for_env(inventory, env)
    if [record["basename"] for record in applied] != [
        record["basename"] for record in run_inputs["applied_exclusions"]
    ]:
        raise AssertionError(f"{env} seed{seed}: applied exclusion inventory mismatch")
    if run_inputs.get("applied_exclusion_union_n_indices") != len(excluded):
        raise AssertionError(f"{env} seed{seed}: applied exclusion union mismatch")

    sample_seed = final_sample_seed(env, seed)
    if run_inputs.get("sample_seed") != sample_seed:
        raise AssertionError(f"{env} seed{seed}: sample seed mismatch")
    eligible = np.flatnonzero(np.max(np.abs(actions), axis=-1) <= ACTION_ELIGIBILITY)
    eligible = np.array([value for value in eligible if int(value) not in excluded])
    if run_inputs.get("n_eligible") != int(len(eligible)):
        raise AssertionError(f"{env} seed{seed}: eligible count mismatch")
    regenerated = np.random.default_rng(sample_seed).choice(
        eligible, size=FINAL_N_STATES, replace=False
    ).astype(np.int64)
    selected = np.asarray(np.load(sub / "state_indices.npy"), dtype=np.int64)
    if not np.array_equal(regenerated, selected):
        raise AssertionError(f"{env} seed{seed}: selected indices do not regenerate")
    if excluded.intersection(int(value) for value in selected):
        raise AssertionError(f"{env} seed{seed}: selected rows overlap excluded rows")
    if sha256_array(selected) != run_inputs["selected_indices_sha256"]:
        raise AssertionError(f"{env} seed{seed}: selected index content hash mismatch")

    states = (
        observations[selected].astype(np.float64)
        - np.asarray(payload["mean"], dtype=np.float64)
    ) / np.asarray(payload["std"], dtype=np.float64)
    anchors = actions[selected].astype(np.float64)
    params = vp.extract_q1(payload)
    expected_shapes = {
        name: list(np.asarray(value).shape) for name, value in params.items()
    }
    if run_inputs.get("q1_parameter_shapes") != expected_shapes:
        raise AssertionError(f"{env} seed{seed}: Q1 parameter schema mismatch")
    q_anchor = vp.direct_forward(params, states, anchors)[0]
    c_ref = float(np.mean(np.abs(q_anchor)) + 1e-6)
    geometry = vp.independent_geometry(params, states, anchors, c_ref)

    stored = np.load(sub / "STATE_GEOMETRY.npz")
    expected_arrays = {
        "dataset_row": selected,
        "normalized_state": states,
        "anchor_action": anchors,
        **geometry,
    }
    if set(stored.files) != set(expected_arrays) | {"grad_jax", "c_ref"}:
        raise AssertionError(f"{env} seed{seed}: STATE_GEOMETRY schema mismatch")
    for name, expected in expected_arrays.items():
        actual = stored[name]
        if expected.dtype.kind in "biu":
            equal = np.array_equal(actual, expected)
        else:
            rtol = 1e-9 if name == "anchor_min_scaled_margin" else 1e-11
            equal = np.allclose(actual, expected, rtol=rtol, atol=1e-11, equal_nan=True)
        if not equal:
            raise AssertionError(f"{env} seed{seed}: geometry mismatch {name}")
    if not math.isclose(float(stored["c_ref"]), c_ref, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{env} seed{seed}: C_ref mismatch")
    grad_jax = np.asarray(stored["grad_jax"], dtype=np.float64)
    if not np.allclose(grad_jax, geometry["grad"], rtol=1e-10, atol=1e-10):
        raise AssertionError(f"{env} seed{seed}: stored JAX gradient mismatch")
    analytic_jax_error = float(np.max(np.abs(grad_jax - geometry["grad"])))

    fd_checked, fd_max_error = vp.finite_difference_checks(params, states, anchors, geometry)
    exit_checked, exit_max_error = vp.exit_bracket_checks(params, states, anchors, geometry)
    if fd_checked == 0 or exit_checked == 0:
        raise AssertionError(f"{env} seed{seed}: zero FD/ReLU checks")
    event_report = vp.box_and_combined_event_checks(params, states, anchors, geometry)

    with (sub / "residence_cells.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != vp.CELL_FIELDS:
            raise AssertionError(f"{env} seed{seed}: residence CSV header mismatch")
        rows = list(reader)
    if len(rows) != len(TIMES) * len(SUBSTEPS):
        raise AssertionError(f"{env} seed{seed}: residence cell count mismatch")
    by_key = {(float(row["T"]), int(row["K"])): row for row in rows}
    if set(by_key) != {(total, k) for total in TIMES for k in SUBSTEPS}:
        raise AssertionError(f"{env} seed{seed}: residence cell grid mismatch")
    affine_checked = 0
    for total in TIMES:
        full_expected = vp.counts(geometry, total)
        residual, integrity_tol, comparison_tol = vp.affine_residual_reference(
            params, states, anchors, geometry, total
        )
        if residual is not None:
            affine_checked += 1
        prior_step_resident = -1
        first_full = None
        for k in SUBSTEPS:
            row = by_key[(total, k)]
            step = total / k
            step_expected = vp.counts(geometry, step)
            expected_row = {
                "environment": env,
                "seed": seed,
                "T": total,
                "K": k,
                "h": step,
                "n_states": FINAL_N_STATES,
                **{f"full_{name}": full_expected[name] for name in CATEGORIES},
                **{f"step_{name}": step_expected[name] for name in CATEGORIES},
                "full_residence_fraction": full_expected["resident"] / FINAL_N_STATES,
                "step_residence_fraction": step_expected["resident"] / FINAL_N_STATES,
                "full_affine_q_max_abs_residual": residual,
            }
            vp.assert_cell_matches(row, expected_row, f"{env} seed{seed} T={total} K={k}", comparison_tol)
            if sum(int(row[f"full_{name}"]) for name in CATEGORIES) != FINAL_N_STATES:
                raise AssertionError(f"{env} seed{seed}: full categories not exhaustive")
            if sum(int(row[f"step_{name}"]) for name in CATEGORIES) != FINAL_N_STATES:
                raise AssertionError(f"{env} seed{seed}: step categories not exhaustive")
            current = int(row["step_resident"])
            if current < prior_step_resident:
                raise AssertionError(f"{env} seed{seed}: step residence not monotone in K")
            prior_step_resident = current
            signature = tuple(int(row[f"full_{name}"]) for name in CATEGORIES)
            if first_full is None:
                first_full = signature
            elif signature != first_full:
                raise AssertionError(f"{env} seed{seed}: full residence changed with K")

    # Recompute survival curve and cross-check RUN_SUMMARY.
    horizons = unique_horizons()
    expected_survival = []
    for horizon in horizons:
        run_counts = vp.counts(geometry, horizon)
        expected_survival.append(
            {
                "horizon": float(horizon),
                "n_resident": int(run_counts["resident"]),
                "resident_fraction": run_counts["resident"] / FINAL_N_STATES,
                "categories": run_counts,
            }
        )
    stored_survival = run_summary.get("survival", [])
    if len(stored_survival) != len(expected_survival):
        raise AssertionError(f"{env} seed{seed}: survival horizon count mismatch")
    for expected, actual in zip(expected_survival, stored_survival, strict=True):
        if not math.isclose(actual["horizon"], expected["horizon"], rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{env} seed{seed}: survival horizon mismatch")
        if int(actual["n_resident"]) != expected["n_resident"]:
            raise AssertionError(f"{env} seed{seed}: survival resident count mismatch")
        if not math.isclose(
            float(actual["resident_fraction"]), expected["resident_fraction"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(f"{env} seed{seed}: survival fraction mismatch")
        if {k: int(v) for k, v in actual["categories"].items()} != expected["categories"]:
            raise AssertionError(f"{env} seed{seed}: survival categories mismatch")
    for total in TIMES:
        expected_counts = vp.counts(geometry, total)
        actual_counts = run_summary["full_category_counts_by_T"][f"{total}"]
        if {k: int(v) for k, v in actual_counts.items()} != expected_counts:
            raise AssertionError(f"{env} seed{seed}: full_category_counts_by_T mismatch")
    if not math.isclose(float(run_summary["C_ref"]), c_ref, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{env} seed{seed}: RUN_SUMMARY C_ref mismatch")
    if not math.isclose(
        float(run_summary["analytic_vs_jax_grad_max_abs"]), analytic_jax_error, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(f"{env} seed{seed}: RUN_SUMMARY analytic/JAX error mismatch")
    if run_summary.get("status") != "run_complete_pending_verification":
        raise AssertionError(f"{env} seed{seed}: RUN_SUMMARY status mismatch")
    if run_summary.get("scientific_admissible") is not False:
        raise AssertionError(f"{env} seed{seed}: raw run cannot self-admit")
    if run_summary.get("scientific_admissible_when_verified") is not True:
        raise AssertionError(f"{env} seed{seed}: verification gate missing")
    if run_summary.get("n_states") != FINAL_N_STATES:
        raise AssertionError(f"{env} seed{seed}: RUN_SUMMARY n_states mismatch")

    return {
        "environment": env,
        "seed": seed,
        "n_states": FINAL_N_STATES,
        "survival": expected_survival,
        "fd_checked": fd_checked,
        "fd_max_error": fd_max_error,
        "exit_checked": exit_checked,
        "exit_max_error": exit_max_error,
        "finite_box_rows_checked": event_report["finite_box_rows_checked"],
        "combined_event_rows_checked": event_report["combined_event_rows_checked"],
        "combined_ties_checked": event_report["combined_ties_checked"],
        "affine_residuals_checked": affine_checked,
    }


def recompute_pooled(
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


def recompute_task_equal(
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


def assert_survival_equal(expected: list[dict[str, Any]], actual: Any, label: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AssertionError(f"{label}: survival length mismatch")
    for exp, act in zip(expected, actual, strict=True):
        if not math.isclose(float(act["horizon"]), exp["horizon"], rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{label}: horizon mismatch")
        if int(act["n_resident"]) != exp["n_resident"] or int(act["n_total"]) != exp["n_total"]:
            raise AssertionError(f"{label}: pooled count mismatch")
        if not math.isclose(
            float(act["resident_fraction"]), exp["resident_fraction"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(f"{label}: pooled fraction mismatch")
        if {name: int(value) for name, value in act["categories"].items()} != exp[
            "categories"
        ]:
            raise AssertionError(f"{label}: pooled category mismatch")


def assert_task_equal_equal(
    expected: list[dict[str, Any]], actual: Any, label: str
) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AssertionError(f"{label}: task-equal length mismatch")
    for exp, act in zip(expected, actual, strict=True):
        if not math.isclose(
            float(act["horizon"]), exp["horizon"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(f"{label}: task-equal horizon mismatch")
        if int(act["n_tasks"]) != exp["n_tasks"]:
            raise AssertionError(f"{label}: task count mismatch")
        if not math.isclose(
            float(act["resident_fraction"]),
            exp["resident_fraction"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"{label}: task-equal resident fraction mismatch")
        for name in CATEGORIES:
            if not math.isclose(
                float(act["category_fractions"][name]),
                exp["category_fractions"][name],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"{label}: task-equal category mismatch {name}")


def verify_final(out_dir: Path, *, check_existing: bool = False) -> dict[str, Any]:
    required = (
        "FINAL_DESIGN_LOCK.json",
        "HARNESS.json",
        "SUMMARY.json",
        "MANIFEST.json",
        "STATUS.json",
    )
    for name in required:
        if not (out_dir / name).is_file():
            raise FileNotFoundError(out_dir / name)
    verify_path = out_dir / "VERIFY.json"
    if check_existing and not verify_path.is_file():
        raise FileNotFoundError(verify_path)
    if not check_existing and verify_path.exists():
        raise FileExistsError(verify_path)

    design = verify_design(out_dir)
    inventory = verify_exclusion_inventory(design)
    verify_input_inventory(design)

    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    status = json.loads((out_dir / "STATUS.json").read_text())
    summary = json.loads((out_dir / "SUMMARY.json").read_text())
    runtime = manifest.get("runtime", {})
    if runtime.get("backend") != "cpu":
        raise AssertionError("manifest CPU runtime missing")
    if runtime.get("jax_enable_x64") is not True:
        raise AssertionError("manifest x64 runtime missing")
    for doc in (manifest, status, summary):
        if doc.get("status") != FINAL_RAW_STATUS:
            raise AssertionError("raw final status mismatch")
        if doc.get("scientific_admissible") is not False:
            raise AssertionError("raw final artifacts cannot self-admit")
        if doc.get("scientific_admissible_when_verified") is not True:
            raise AssertionError("raw final verification gate missing")
    if summary.get("coverage_gate") is not None or summary.get("learned_slope") is not None:
        raise AssertionError("final summary contains a forbidden learned outcome gate")
    if manifest.get("design_sha256") != design["design_sha256"]:
        raise AssertionError("manifest design hash mismatch")

    verify_final_git_provenance(manifest.get("git_provenance"))
    vp.verify_harness(json.loads((out_dir / "HARNESS.json").read_text()))

    # Source snapshots + artifact inventory.
    if design["source_sha256"] != manifest["source_sha256"]:
        raise AssertionError("design/manifest source hashes differ")
    snapshots = manifest.get("source_snapshots", {})
    if set(snapshots) != set(manifest["source_sha256"]):
        raise AssertionError("source snapshot inventory mismatch")
    for name, digest in manifest["source_sha256"].items():
        expected_path = f"{SOURCE_SNAPSHOT_DIR}/{name}"
        if snapshots[name] != {"path": expected_path, "sha256": digest}:
            raise AssertionError(f"source snapshot record mismatch: {name}")
        if sha256_file(out_dir / expected_path) != digest:
            raise AssertionError(f"source snapshot content mismatch: {name}")
    verifier_name = Path(__file__).name
    if sha256_file(Path(__file__).resolve()) != manifest["source_sha256"].get(verifier_name):
        raise AssertionError("invoke the exact snapshotted final verifier")

    expected_artifacts = {"FINAL_DESIGN_LOCK.json", "HARNESS.json", "SUMMARY.json"}
    expected_artifacts.update(
        f"{SOURCE_SNAPSHOT_DIR}/{name}" for name in design["source_sha256"]
    )
    for env in FINAL_ENVIRONMENTS:
        for seed in FINAL_SEEDS:
            prefix = f"{env}_seed{seed}"
            for leaf in (
                "state_indices.npy",
                "STATE_GEOMETRY.npz",
                "residence_cells.csv",
                "RUN_INPUTS.json",
                "RUN_SUMMARY.json",
            ):
                expected_artifacts.add(f"{prefix}/{leaf}")
    vp.verify_artifact_inventory(out_dir, manifest.get("artifacts"), expected_artifacts)

    per_run: list[dict[str, Any]] = []
    fd_total = 0
    exit_total = 0
    box_total = 0
    combined_total = 0
    ties_total = 0
    for env in FINAL_ENVIRONMENTS:
        for seed in FINAL_SEEDS:
            record = verify_run(out_dir, design, env, seed, inventory)
            per_run.append(record)
            fd_total += record["fd_checked"]
            exit_total += record["exit_checked"]
            box_total += record["finite_box_rows_checked"]
            combined_total += record["combined_event_rows_checked"]
            ties_total += record["combined_ties_checked"]

    # Cross-check aggregate survival against SUMMARY.per_run and pooled/family curves.
    summary_runs = {
        (row["environment"], int(row["seed"])): row for row in summary["per_run"]
    }
    expected_run_keys = {
        (env, seed) for env in FINAL_ENVIRONMENTS for seed in FINAL_SEEDS
    }
    if set(summary_runs) != expected_run_keys:
        raise AssertionError("SUMMARY per-run key grid mismatch")
    for record in per_run:
        stored_run = summary_runs[(record["environment"], record["seed"])]
        sub_run = json.loads(
            (
                out_dir
                / f"{record['environment']}_seed{record['seed']}"
                / "RUN_SUMMARY.json"
            ).read_text()
        )
        if canonical_bytes(stored_run) != canonical_bytes(sub_run):
            raise AssertionError("SUMMARY per-run copy differs from RUN_SUMMARY")
        assert_run_survival = stored_run["survival"]
        if len(assert_run_survival) != len(record["survival"]):
            raise AssertionError("SUMMARY per-run survival length mismatch")
        for exp, act in zip(record["survival"], assert_run_survival, strict=True):
            if int(act["n_resident"]) != exp["n_resident"]:
                raise AssertionError("SUMMARY per-run survival mismatch")

    horizons = unique_horizons()
    if [float(value) for value in summary.get("horizons", [])] != horizons:
        raise AssertionError("SUMMARY horizon grid mismatch")
    expected_pooled = recompute_pooled(per_run, horizons)
    assert_survival_equal(expected_pooled, summary.get("pooled_survival"), "pooled")
    expected_tasks = {
        env: recompute_pooled(
            per_run,
            horizons,
            [run for run in per_run if run["environment"] == env],
        )
        for env in FINAL_ENVIRONMENTS
    }
    if set(summary.get("task_survival", {})) != set(FINAL_ENVIRONMENTS):
        raise AssertionError("SUMMARY task-survival grid mismatch")
    for env, expected_task in expected_tasks.items():
        assert_survival_equal(
            expected_task, summary["task_survival"][env], f"task/{env}"
        )
    expected_task_equal = recompute_task_equal(expected_tasks, horizons)
    assert_task_equal_equal(
        expected_task_equal,
        summary.get("task_equal_survival"),
        "task_equal",
    )
    families = sorted({env.split("-", 1)[0] for env in FINAL_ENVIRONMENTS})
    if set(summary.get("family_survival", {})) != set(families):
        raise AssertionError("SUMMARY family-survival grid mismatch")
    for family in families:
        subset = [run for run in per_run if run["environment"].split("-", 1)[0] == family]
        expected_family = recompute_pooled(per_run, horizons, subset)
        assert_survival_equal(
            expected_family, summary["family_survival"].get(family), f"family/{family}"
        )
    if summary.get("n_runs") != len(per_run) or summary.get("n_anchors_total") != sum(
        run["n_states"] for run in per_run
    ):
        raise AssertionError("SUMMARY run/anchor totals mismatch")

    report = {
        "protocol": FINAL_PROTOCOL,
        "status": "verified_complete",
        "pass": True,
        "scientific_admissible": True,
        "n_runs": len(per_run),
        "n_states_per_run": FINAL_N_STATES,
        "n_anchors_total": sum(run["n_states"] for run in per_run),
        "independent_geometry_recompute_pass": True,
        "finite_difference_pass": True,
        "finite_difference_coordinates_checked": fd_total,
        "exit_bracket_pass": True,
        "relu_exits_bracketed_within_Tmax": exit_total,
        "finite_box_rows_checked": box_total,
        "combined_event_rows_checked": combined_total,
        "combined_ties_checked": ties_total,
        "pooled_survival_pass": True,
        "task_survival_pass": True,
        "task_equal_survival_pass": True,
        "aggregate_boundary_categories_pass": True,
        "family_survival_pass": True,
        "manifest_sha256": sha256_file(out_dir / "MANIFEST.json"),
        "design_sha256": design["design_sha256"],
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "written_at": now_iso(),
    }
    if check_existing:
        vp.verify_stored_report(verify_path, report)
    else:
        write_json_create(verify_path, report)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = verify_final(args.out_dir, check_existing=args.check_existing)
    except Exception as exc:
        print(f"FAIL P2 ReLU residence FINAL: {type(exc).__name__}: {exc}")
        return 1
    print(
        "PASS P2 ReLU residence FINAL: "
        f"{report['n_runs']} bundles, {report['n_anchors_total']} anchors, "
        f"{report['finite_difference_coordinates_checked']} FD coordinates verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
