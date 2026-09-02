#!/usr/bin/env python3
"""Independently verify the compact P1 release's internal consistency.

The compact archive omits raw arrays and checkpoints.  This verifier therefore
checks its exact grid, hash chain, retained arithmetic, derived tables, and
manuscript-facing summaries without importing the P1 runner or raw verifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "p1_target_value_audit_v2"
ENVIRONMENTS = (
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
METHOD_HOPS = {"td3": 1, "p3": 3, "p4": 4}
TAUS = (4.0, 7.0, 10.0, 14.0, 20.0)
SEEDS = (0, 1)
OWN_SCOPE = "within_run_target_critic"
COMMON_SCOPE = "common_td3_target_critic"
RESIDUAL_SCOPE = "within_run_online_critic"
CHECKPOINT_STEP = 1_000_000
N_ROWS = 4096
SHA256 = re.compile(r"[0-9a-f]{64}")

COMPACT_FILES = {
    "EXPECTED_GRID.json",
    "FROZEN_PROTOCOL.json",
    "MANIFEST.json",
    "README.md",
    "RUN_STATUS.json",
    "VERIFY.json",
    "comparator_residuals.csv",
    "geometry_correlations.csv",
    "method_contrasts.csv",
    "residual_aggregates.csv",
    "same_next_state_geometry.csv",
    "td_target_value.csv",
}
CSV_FIELDS = {
    "same_next_state_geometry.csv": tuple(
        (
            "key,method,environment,tau,seed,checkpoint_step,checkpoint_file_sha256,"
            "weights_sha256,config_sha256,dataset_sha256,normalization_sha256,"
            "selected_transition_sha256,common_noise_sha256,external_score,stability,"
            "n_rows,action_dim,target_to_next_data_rms,final_to_next_data_rms,"
            "final_to_target_rms,smoothed_target_to_next_data_rms,final_target_ratio,"
            "target_saturation_fraction,final_saturation_fraction,"
            "smoothed_target_saturation_fraction,smoothed_dataset_saturation_fraction,"
            "all_finite,raw_npz,raw_sha256"
        ).split(",")
    ),
    "td_target_value.csv": tuple(
        (
            "key,method,environment,tau,seed,checkpoint_step,checkpoint_file_sha256,"
            "weights_sha256,config_sha256,dataset_sha256,normalization_sha256,"
            "selected_transition_sha256,common_noise_sha256,external_score,stability,"
            "critic_scope,critic_params_sha256,gamma,action_clip_target_fraction,"
            "action_clip_dataset_fraction,n_rows,delta_y_rms,delta_y_abs_mean,"
            "target_actor_rms,target_dataset_rms,target_actor_abs_mean,"
            "target_dataset_abs_mean,q_actor_abs_mean,q_dataset_abs_mean,all_finite,"
            "action_clip_target_fraction_no_noise,action_clip_dataset_fraction_no_noise,"
            "delta_y_rms_no_noise,delta_y_abs_mean_no_noise,"
            "smoothing_delta_y_rms_change,smoothing_delta_y_rms_ratio,"
            "no_noise_all_finite,raw_npz,raw_sha256"
        ).split(",")
    ),
    "comparator_residuals.csv": tuple(
        (
            "key,method,environment,tau,seed,checkpoint_step,checkpoint_file_sha256,"
            "weights_sha256,config_sha256,dataset_sha256,normalization_sha256,"
            "selected_transition_sha256,common_noise_sha256,external_score,stability,"
            "critic_scope,critic_params_sha256,intervention,"
            "historical_training_step_recovered,hop,total_hops,tau_step,C_k,q_weight,"
            "optimizer_step_before,optimizer_step_after,r_before,slack_before,r_after,"
            "slack_after,actor_loss_before,actor_loss_after,comparator_loss,q_mean_before,"
            "q_mean_after,reference_q_mean,transport_before,transport_after,all_finite,"
            "raw_npz,raw_sha256"
        ).split(",")
    ),
    "method_contrasts.csv": tuple(
        (
            "level,method_minus,method_reference,critic_scope,environment,tau,seed,"
            "delta_y_rms_difference,delta_y_rms_no_noise_difference,"
            "smoothing_delta_y_rms_change_difference,seed_interpretation,"
            "generalization_scope"
        ).split(",")
    ),
    "residual_aggregates.csv": tuple(
        (
            "method,hop,tau,environment,stability,n_seed_cells,r_before_mean,"
            "slack_before_mean,r_after_mean,slack_after_mean"
        ).split(",")
    ),
    "geometry_correlations.csv": tuple(
        (
            "method,stability,n_cells,"
            "pearson_final_target_ratio_vs_within_run_delta_y_rms,"
            "external_stability_definition"
        ).split(",")
    ),
}
HEADLINE = {
    "p3": {
        "target": (88, 0.6503870640408389),
        "final": (37, 1.0602393618282206),
        "final_over_target": (90, 1.2972705782079244),
        "common": (86, 0.7174210425586443),
        "common_no_noise": (86, 0.7015723762795683),
        "zero_slack": 85,
        "median_slack": 0.0005192383836414294,
        "positive_slack_decreases": (95, 95),
    },
    "p4": {
        "target": (89, 0.5645070565332423),
        "final": (44, 1.0252651612633423),
        "final_over_target": (90, 1.3648187375341083),
        "common": (88, 0.6171611189767942),
        "common_no_noise": (88, 0.6065054083885606),
        "zero_slack": 64,
        "median_slack": 0.00145720429610674,
        "positive_slack_decreases": (205, 206),
    },
    "td3_critic_explosions": 31,
    "td3_log_scale_delta_correlation": 0.9986391753735672,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS[path.name]:
            raise AssertionError(f"{path.name}: exact CSV schema mismatch")
        return list(reader)


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{label}: not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise AssertionError(f"{label}: not finite: {value!r}")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise AssertionError(f"{label}: Boolean is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{label}: not an integer: {value!r}") from error
    if str(result) != str(value):
        raise AssertionError(f"{label}: noncanonical integer: {value!r}")
    return result


def _boolean(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AssertionError(f"{label}: noncanonical Boolean: {value!r}")


def _close(actual: float, expected: float, label: str, tolerance: float = 2e-6) -> None:
    scale = max(1.0, abs(actual), abs(expected))
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise AssertionError(f"{label}: nonfinite comparison")
    if abs(actual - expected) > tolerance * scale:
        raise AssertionError(f"{label}: {actual} != {expected}")


def _strict_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def _base_key(row: Mapping[str, str]) -> tuple[str, str, float, int]:
    return (
        row["method"],
        row["environment"],
        _finite(row["tau"], f"{row.get('key')} tau"),
        _integer(row["seed"], f"{row.get('key')} seed"),
    )


def _key_text(key: tuple[str, str, float, int]) -> str:
    method, environment, tau, seed = key
    tau_text = str(int(tau)) if tau.is_integer() else str(tau)
    return f"{method}|{environment}|{tau_text}|{seed}"


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def _verify_metadata(
    compact_dir: Path,
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    actual_files = {
        path.relative_to(compact_dir).as_posix()
        for path in compact_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != COMPACT_FILES:
        raise AssertionError("P1 compact inventory contains a missing or extra file")

    expected_grid = _read_json(compact_dir / "EXPECTED_GRID.json")
    protocol = _read_json(compact_dir / "FROZEN_PROTOCOL.json")
    manifest = _read_json(compact_dir / "MANIFEST.json")
    status = _read_json(compact_dir / "RUN_STATUS.json")
    receipt = _read_json(compact_dir / "VERIFY.json")

    if set(expected_grid) != {
        "analysis_started", "cells", "checkpoint_step", "environments", "methods",
        "mode", "n_expected", "no_substitution", "protocol", "root", "seeds", "taus",
    }:
        raise AssertionError("P1 EXPECTED_GRID schema mismatch")
    expected_grid_header = {
        "protocol": PROTOCOL,
        "mode": "dry_run",
        "n_expected": 270,
        "checkpoint_step": CHECKPOINT_STEP,
        "environments": list(ENVIRONMENTS),
        "seeds": list(SEEDS),
        "taus": list(TAUS),
        "no_substitution": True,
        "analysis_started": False,
    }
    for field, expected in expected_grid_header.items():
        if expected_grid.get(field) != expected:
            raise AssertionError(f"P1 EXPECTED_GRID header mismatch: {field}")
    expected_methods = {
        "td3": {
            "hops": 1,
            "result_dir": "results_qnorm",
            "score_key": "d4rl_score",
            "tag": "",
        },
        "p3": {
            "hops": 3,
            "result_dir": "results_mpi3",
            "score_key": "d4rl_pi3",
            "tag": "mpi3",
        },
        "p4": {
            "hops": 4,
            "result_dir": "results/mpi4_norm",
            "score_key": "d4rl_pi4",
            "tag": "mpi4",
        },
    }
    if expected_grid.get("methods") != expected_methods:
        raise AssertionError("P1 EXPECTED_GRID method contract mismatch")
    cells = expected_grid.get("cells")
    if not isinstance(cells, list) or len(cells) != 270:
        raise AssertionError("P1 EXPECTED_GRID cell count mismatch")
    cell_keys = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "checkpoint_path",
            "config_path",
            "environment",
            "eval_path",
            "hops",
            "key",
            "method",
            "result_dir",
            "root",
            "run_dir",
            "run_name",
            "seed",
            "tau",
        }:
            raise AssertionError("P1 EXPECTED_GRID cell schema mismatch")
        key = (
            cell["method"], cell["environment"], _finite(cell["tau"], "grid tau"),
            _integer(cell["seed"], "grid seed"),
        )
        if cell["key"] != _key_text(key) or cell["hops"] != METHOD_HOPS.get(key[0]):
            raise AssertionError(f"P1 EXPECTED_GRID cell identity mismatch: {key}")
        cell_keys.add(key)
    expected_keys = set(product(METHOD_HOPS, ENVIRONMENTS, TAUS, SEEDS))
    if cell_keys != expected_keys or len(cell_keys) != len(cells):
        raise AssertionError("P1 EXPECTED_GRID exact key grid mismatch")

    expected_counts = {
        "checkpoints": 270,
        "value_rows": 540,
        "geometry_rows": 270,
        "residual_rows": 450,
        "raw_npz": 270,
    }
    if (
        protocol.get("protocol") != PROTOCOL
        or protocol.get("locked") is not True
        or protocol.get("grid")
        != {
            "checkpoint_step": CHECKPOINT_STEP,
            "environments": list(ENVIRONMENTS),
            "methods": list(METHOD_HOPS),
            "seeds": list(SEEDS),
            "taus": list(TAUS),
        }
        or protocol.get("expected_counts") != expected_counts
        or protocol.get("value_scopes") != [OWN_SCOPE, COMMON_SCOPE]
        or protocol.get("common_critic")
        != "matched TD3 target critic at identical environment/tau/seed"
        or protocol.get("geometry_state")
        != "same frozen next-state batch for target and final actors"
        or protocol.get("collapse_label") != "external normalized return < 20"
    ):
        raise AssertionError("P1 frozen protocol identity/grid mismatch")
    noise = protocol.get("common_noise", {})
    if (
        noise.get("same_clipped_epsilon_for_actor_and_recorded_next_action") is not True
        or noise.get("same_epsilon_across_methods_and_seeds_within_environment") is not True
    ):
        raise AssertionError("P1 common-noise contract mismatch")
    limits = protocol.get("interpretation_limits", {})
    if (
        "post-hoc mechanism readouts" not in str(limits.get("causality"))
        or "not a global optimality gap or bound on epsilon_k"
        not in str(limits.get("residual"))
        or "accuracy estimate" not in str(limits.get("critic"))
    ):
        raise AssertionError("P1 frozen interpretation limits mismatch")
    residual_contract = protocol.get("residual", {})
    if residual_contract != {
        "critic_scope": RESIDUAL_SCOPE,
        "fresh_optimizer_allowed": False,
        "historical_training_step_recovered": False,
        "hops": "k>=2 only",
        "intervention": "posthoc_final_checkpoint_one_adam_step",
    }:
        raise AssertionError("P1 frozen residual contract mismatch")

    if set(status) != {
        "analysis_started", "finite_value_gate", "independent_verifier_required",
        "run_allowed", "state", "written_at",
    } or any(
        (
            status.get("analysis_started") is not True,
            status.get("finite_value_gate") is not True,
            status.get("independent_verifier_required") is not True,
            status.get("run_allowed") is not True,
            status.get("state") != "analysis_complete_unverified",
        )
    ):
        raise AssertionError("P1 create-only RUN_STATUS gate mismatch")

    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("status") != "analysis_complete"
        or manifest.get("cpu_only") is not True
        or manifest.get("jax_backend") != "cpu"
        or manifest.get("finite_value_gate") is not True
        or manifest.get("historical_training_step_recovered") is not False
        or manifest.get("counts") != expected_counts
        or manifest.get("derived_counts")
        != {
            "geometry_correlation_rows": 9,
            "method_contrast_rows": 580,
            "residual_aggregate_rows": 256,
        }
        or manifest.get("residual_intervention")
        != "posthoc_final_checkpoint_one_adam_step"
    ):
        raise AssertionError("P1 MANIFEST status/count/runtime mismatch")
    manifest_limits = manifest.get("interpretation_limits", {})
    for field in (
        "broader_task_generalization", "causal_return_effect", "critic_accuracy",
        "epsilon_k_bound", "global_optimality_gap",
    ):
        if manifest_limits.get(field) is not False:
            raise AssertionError(f"P1 MANIFEST interpretation limit mismatch: {field}")

    protocol_hash = sha256_file(compact_dir / "FROZEN_PROTOCOL.json")
    manifest_hash = sha256_file(compact_dir / "MANIFEST.json")
    inventory_hash = manifest.get("inventory_sha256")
    if (
        not isinstance(inventory_hash, str)
        or SHA256.fullmatch(inventory_hash) is None
        or protocol.get("checkpoint_manifest_sha256") != inventory_hash
        or manifest.get("protocol_sha256") != protocol_hash
    ):
        raise AssertionError("P1 retained inventory/protocol hash chain mismatch")
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("pass") is not True
        or receipt.get("artifact_integrity_pass") is not True
        or receipt.get("scientific_inclusion_gate_pass") is not True
        or receipt.get("finite_value_gate") is not True
        or receipt.get("artifact_only") is not True
        or receipt.get("imports_analysis_runner") is not False
        or receipt.get("counts") != expected_counts
        or receipt.get("inventory_sha256") != inventory_hash
        or receipt.get("protocol_sha256") != protocol_hash
        or receipt.get("manifest_sha256") != manifest_hash
    ):
        raise AssertionError("P1 independent VERIFY receipt mismatch")
    source_verifier = repo / "scripts" / "diagnostics" / "verify_p1_target_value_audit.py"
    source_runner = repo / "scripts" / "diagnostics" / "run_p1_target_value_audit.py"
    if (
        not source_verifier.is_file()
        or receipt.get("verifier_source_sha256") != sha256_file(source_verifier)
        or not source_runner.is_file()
        or protocol.get("diagnostic_code_sha256") != sha256_file(source_runner)
    ):
        raise AssertionError("P1 released runner/verifier source hash mismatch")

    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(CSV_FIELDS):
        raise AssertionError("P1 scientific CSV inventory mismatch")
    for name, digest in artifact_hashes.items():
        if SHA256.fullmatch(str(digest)) is None or sha256_file(compact_dir / name) != digest:
            raise AssertionError(f"P1 scientific CSV hash mismatch: {name}")
    raw_hashes = manifest.get("raw_sha256")
    if not isinstance(raw_hashes, dict) or len(raw_hashes) != 270:
        raise AssertionError("P1 raw-NPZ receipt inventory mismatch")
    for relative, digest in raw_hashes.items():
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".npz"
            or SHA256.fullmatch(str(digest)) is None
        ):
            raise AssertionError(f"P1 unsafe or invalid raw-NPZ receipt: {relative}")
    return protocol, manifest, receipt


def _verify_primary_rows(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    geometry_rows: Sequence[dict[str, str]],
    value_rows: Sequence[dict[str, str]],
    residual_rows: Sequence[dict[str, str]],
) -> tuple[
    dict[tuple[str, str, float, int], dict[str, str]],
    dict[tuple[str, str, float, int, str], dict[str, str]],
]:
    expected_base = set(product(METHOD_HOPS, ENVIRONMENTS, TAUS, SEEDS))
    geometry = {_base_key(row): row for row in geometry_rows}
    if len(geometry_rows) != 270 or len(geometry) != 270 or set(geometry) != expected_base:
        raise AssertionError("P1 geometry rows are not the exact 3x9x5x2 grid")
    value = {(*_base_key(row), row["critic_scope"]): row for row in value_rows}
    expected_value = {
        (*key, scope)
        for key in expected_base
        for scope in (OWN_SCOPE, COMMON_SCOPE)
    }
    if len(value_rows) != 540 or len(value) != 540 or set(value) != expected_value:
        raise AssertionError("P1 target-value rows are not the exact grid x two scopes")

    expected_residual = {
        (*key, hop)
        for key in expected_base
        if key[0] in ("p3", "p4")
        for hop in range(2, METHOD_HOPS[key[0]] + 1)
    }
    residual_keys = {
        (*_base_key(row), _integer(row["hop"], "residual hop"))
        for row in residual_rows
    }
    if (
        len(residual_rows) != 450
        or len(residual_keys) != 450
        or residual_keys != expected_residual
    ):
        raise AssertionError("P1 residual rows are not exact P3/P4 later-hop grid")

    sha_fields = (
        "checkpoint_file_sha256",
        "weights_sha256",
        "config_sha256",
        "dataset_sha256",
        "normalization_sha256", "selected_transition_sha256", "common_noise_sha256",
        "raw_sha256",
    )
    shared_fields = (
        "key",
        "method",
        "environment",
        "tau",
        "seed",
        "checkpoint_step",
        *sha_fields,
        "external_score",
        "stability",
        "raw_npz",
    )
    raw_hashes = manifest["raw_sha256"]
    environment_hashes: dict[str, dict[str, set[str]]] = {
        environment: {
            field: set()
            for field in (
                "dataset_sha256",
                "normalization_sha256",
                "selected_transition_sha256",
                "common_noise_sha256",
            )
        }
        for environment in ENVIRONMENTS
    }
    for key, row in geometry.items():
        if (
            row["key"] != _key_text(key)
            or _integer(row["checkpoint_step"], "geometry step") != CHECKPOINT_STEP
        ):
            raise AssertionError(f"P1 geometry identity mismatch: {key}")
        for field in sha_fields:
            if SHA256.fullmatch(row[field]) is None:
                raise AssertionError(f"P1 invalid {field}: {key}")
        if raw_hashes.get(row["raw_npz"]) != row["raw_sha256"]:
            raise AssertionError(f"P1 geometry raw receipt mismatch: {key}")
        for field in environment_hashes[key[1]]:
            environment_hashes[key[1]][field].add(row[field])
        score = _finite(row["external_score"], f"{key} external score")
        expected_stability = "collapsed" if score < 20.0 else "stable"
        if row["stability"] != expected_stability:
            raise AssertionError(f"P1 external stability label mismatch: {key}")
        if _integer(row["n_rows"], f"{key} geometry rows") != N_ROWS:
            raise AssertionError(f"P1 geometry sample count mismatch: {key}")
        if _integer(row["action_dim"], f"{key} action_dim") <= 0:
            raise AssertionError(f"P1 invalid action dimension: {key}")
        for field in (
            "target_to_next_data_rms", "final_to_next_data_rms", "final_to_target_rms",
            "smoothed_target_to_next_data_rms",
        ):
            if _finite(row[field], f"{key} {field}") < 0.0:
                raise AssertionError(f"P1 negative geometry RMS: {key} {field}")
        target = _finite(row["target_to_next_data_rms"], f"{key} target RMS")
        final = _finite(row["final_to_next_data_rms"], f"{key} final RMS")
        if target <= 0.0:
            raise AssertionError(f"P1 zero target-geometry denominator: {key}")
        _strict_close(
            _finite(row["final_target_ratio"], f"{key} final/target ratio"),
            final / target,
            f"P1 final/target ratio mismatch: {key}",
        )
        for field in (
            "target_saturation_fraction", "final_saturation_fraction",
            "smoothed_target_saturation_fraction",
            "smoothed_dataset_saturation_fraction",
        ):
            fraction = _finite(row[field], f"{key} {field}")
            if not 0.0 <= fraction <= 1.0:
                raise AssertionError(f"P1 saturation fraction out of range: {key} {field}")
        if _boolean(row["all_finite"], f"{key} geometry all_finite") is not True:
            raise AssertionError(f"P1 nonfinite geometry flag: {key}")
    if any(
        len(values) != 1
        for fields in environment_hashes.values()
        for values in fields.values()
    ):
        raise AssertionError("P1 common batch/noise hashes vary within an environment")
    if set(raw_hashes) != {row["raw_npz"] for row in geometry_rows}:
        raise AssertionError("P1 raw-NPZ receipts do not match the 270 geometry cells")

    value_numeric = (
        "gamma", "action_clip_target_fraction", "action_clip_dataset_fraction",
        "delta_y_rms", "delta_y_abs_mean", "target_actor_rms", "target_dataset_rms",
        "target_actor_abs_mean", "target_dataset_abs_mean", "q_actor_abs_mean",
        "q_dataset_abs_mean", "action_clip_target_fraction_no_noise",
        "action_clip_dataset_fraction_no_noise", "delta_y_rms_no_noise",
        "delta_y_abs_mean_no_noise", "smoothing_delta_y_rms_change",
        "smoothing_delta_y_rms_ratio",
    )
    for key_scope, row in value.items():
        key = key_scope[:4]
        base = geometry[key]
        if any(row[field] != base[field] for field in shared_fields):
            raise AssertionError(f"P1 value/geometry provenance mismatch: {key_scope}")
        if SHA256.fullmatch(row["critic_params_sha256"]) is None:
            raise AssertionError(f"P1 invalid critic fingerprint: {key_scope}")
        if _integer(row["n_rows"], f"{key_scope} value rows") != N_ROWS:
            raise AssertionError(f"P1 value sample count mismatch: {key_scope}")
        numbers = {field: _finite(row[field], f"{key_scope} {field}") for field in value_numeric}
        if numbers["gamma"] != 0.99:
            raise AssertionError(f"P1 gamma mismatch: {key_scope}")
        for field in (
            "action_clip_target_fraction", "action_clip_dataset_fraction",
            "action_clip_target_fraction_no_noise", "action_clip_dataset_fraction_no_noise",
        ):
            if not 0.0 <= numbers[field] <= 1.0:
                raise AssertionError(f"P1 clip fraction out of range: {key_scope} {field}")
        for field in (
            "delta_y_rms", "delta_y_abs_mean", "target_actor_rms", "target_dataset_rms",
            "target_actor_abs_mean", "target_dataset_abs_mean", "q_actor_abs_mean",
            "q_dataset_abs_mean", "delta_y_rms_no_noise", "delta_y_abs_mean_no_noise",
        ):
            if numbers[field] < 0.0:
                raise AssertionError(f"P1 negative magnitude: {key_scope} {field}")
        no_noise = numbers["delta_y_rms_no_noise"]
        if no_noise <= 0.0:
            raise AssertionError(f"P1 zero no-noise RMS denominator: {key_scope}")
        _strict_close(
            numbers["smoothing_delta_y_rms_change"],
            numbers["delta_y_rms"] - no_noise,
            f"P1 smoothing-change identity mismatch: {key_scope}",
        )
        _strict_close(
            numbers["smoothing_delta_y_rms_ratio"],
            numbers["delta_y_rms"] / no_noise,
            f"P1 smoothing-ratio identity mismatch: {key_scope}",
        )
        if (
            _boolean(row["all_finite"], f"{key_scope} all_finite") is not True
            or _boolean(row["no_noise_all_finite"], f"{key_scope} no_noise_all_finite")
            is not True
        ):
            raise AssertionError(f"P1 nonfinite value flag: {key_scope}")
    for environment, tau, seed in product(ENVIRONMENTS, TAUS, SEEDS):
        td3_own = value[("td3", environment, tau, seed, OWN_SCOPE)]
        td3_common = value[("td3", environment, tau, seed, COMMON_SCOPE)]
        if {field: td3_own[field] for field in td3_own if field != "critic_scope"} != {
            field: td3_common[field] for field in td3_common if field != "critic_scope"
        }:
            raise AssertionError(
                f"P1 TD3 own/common rows differ: {(environment, tau, seed)}"
            )
        common_hashes = {
            value[(method, environment, tau, seed, COMMON_SCOPE)]["critic_params_sha256"]
            for method in METHOD_HOPS
        }
        if common_hashes != {td3_own["critic_params_sha256"]}:
            raise AssertionError(
                f"P1 common-TD3 critic pairing mismatch: {(environment, tau, seed)}"
            )

    active_step = protocol["optimizer_state_contract"]["schedule_derived_active_step"]
    for row in residual_rows:
        key = _base_key(row)
        hop = _integer(row["hop"], f"{key} residual hop")
        base = geometry[key]
        if any(row[field] != base[field] for field in shared_fields):
            raise AssertionError(f"P1 residual/geometry provenance mismatch: {(*key, hop)}")
        if (
            row["critic_scope"] != RESIDUAL_SCOPE
            or row["intervention"] != "posthoc_final_checkpoint_one_adam_step"
            or _boolean(
                row["historical_training_step_recovered"],
                f"{key} historical training step",
            )
            is not False
            or _integer(row["total_hops"], f"{key} total hops") != METHOD_HOPS[key[0]]
            or _integer(row["optimizer_step_before"], f"{key} optimizer before") != active_step
            or _integer(row["optimizer_step_after"], f"{key} optimizer after") != active_step + 1
            or SHA256.fullmatch(row["critic_params_sha256"]) is None
        ):
            raise AssertionError(f"P1 comparator residual scope/step mismatch: {(*key, hop)}")
        total_hops = METHOD_HOPS[key[0]]
        tau_step = _finite(row["tau_step"], f"{key} tau_step")
        c_k = _finite(row["C_k"], f"{key} C_k")
        q_weight = _finite(row["q_weight"], f"{key} q_weight")
        if c_k <= 0.0:
            raise AssertionError(f"P1 nonpositive C_k: {(*key, hop)}")
        _strict_close(tau_step, key[2] / total_hops, f"P1 tau_step mismatch: {(*key, hop)}")
        _strict_close(q_weight, 2.0 * tau_step / c_k, f"P1 q_weight mismatch: {(*key, hop)}")
        numeric = {
            field: _finite(row[field], f"{(*key, hop)} {field}")
            for field in (
                "r_before", "slack_before", "r_after", "slack_after",
                "actor_loss_before", "actor_loss_after", "comparator_loss",
                "q_mean_before", "q_mean_after", "reference_q_mean",
                "transport_before", "transport_after",
            )
        }
        if numeric["transport_before"] < 0.0 or numeric["transport_after"] < 0.0:
            raise AssertionError(f"P1 negative transport term: {(*key, hop)}")
        _close(
            numeric["actor_loss_before"],
            -q_weight * numeric["q_mean_before"] + numeric["transport_before"],
            f"P1 actor-loss-before identity mismatch: {(*key, hop)}",
        )
        _close(
            numeric["actor_loss_after"],
            -q_weight * numeric["q_mean_after"] + numeric["transport_after"],
            f"P1 actor-loss-after identity mismatch: {(*key, hop)}",
        )
        _close(
            numeric["comparator_loss"],
            -q_weight * numeric["reference_q_mean"],
            f"P1 comparator-loss identity mismatch: {(*key, hop)}",
        )
        _strict_close(
            numeric["r_before"],
            numeric["actor_loss_before"] - numeric["comparator_loss"],
            f"P1 residual-before identity mismatch: {(*key, hop)}",
        )
        _strict_close(
            numeric["r_after"],
            numeric["actor_loss_after"] - numeric["comparator_loss"],
            f"P1 residual-after identity mismatch: {(*key, hop)}",
        )
        _strict_close(
            numeric["slack_before"], max(0.0, numeric["r_before"]),
            f"P1 slack-before identity mismatch: {(*key, hop)}",
        )
        _strict_close(
            numeric["slack_after"], max(0.0, numeric["r_after"]),
            f"P1 slack-after identity mismatch: {(*key, hop)}",
        )
        if _boolean(row["all_finite"], f"{(*key, hop)} all_finite") is not True:
            raise AssertionError(f"P1 nonfinite residual flag: {(*key, hop)}")
    return geometry, value


def _verify_method_contrasts(
    value: Mapping[tuple[str, str, float, int, str], Mapping[str, str]],
    rows: Sequence[Mapping[str, str]],
) -> None:
    metrics = {
        "delta_y_rms_difference": "delta_y_rms",
        "delta_y_rms_no_noise_difference": "delta_y_rms_no_noise",
        "smoothing_delta_y_rms_change_difference": "smoothing_delta_y_rms_change",
    }
    seed_text = "seeds are marginal run replicates within fixed task-budget cells"
    scope_text = "fixed nine-task, five-budget grid only"

    def semantic_key(row: Mapping[str, str]) -> tuple[object, ...]:
        return (
            row["level"], row["method_minus"], row["critic_scope"], row["environment"],
            None if row["tau"] == "" else _finite(row["tau"], "contrast tau"),
            None if row["seed"] == "" else _integer(row["seed"], "contrast seed"),
        )

    actual = {semantic_key(row): row for row in rows}
    expected_keys = set()
    for method, scope in product(("p3", "p4"), (OWN_SCOPE, COMMON_SCOPE)):
        expected_keys.update(
            ("cell", method, scope, environment, tau, seed)
            for environment, tau, seed in product(ENVIRONMENTS, TAUS, SEEDS)
        )
        expected_keys.update(
            ("task_budget_seed_mean", method, scope, environment, tau, None)
            for environment, tau in product(ENVIRONMENTS, TAUS)
        )
        expected_keys.update(
            ("task_marginal_mean", method, scope, environment, None, None)
            for environment in ENVIRONMENTS
        )
        expected_keys.add(("task_equal_overall", method, scope, "", None, None))
    if len(rows) != 580 or len(actual) != 580 or set(actual) != expected_keys:
        raise AssertionError("P1 method-contrast exact key grid mismatch")

    def difference(
        method: str,
        environment: str,
        tau: float,
        seed: int,
        scope: str,
        field: str,
    ) -> float:
        method_value = _finite(
            value[(method, environment, tau, seed, scope)][field],
            "method value",
        )
        td3_value = _finite(
            value[("td3", environment, tau, seed, scope)][field],
            "TD3 value",
        )
        return method_value - td3_value

    for key, row in actual.items():
        level, method, scope, environment, tau, seed = key
        if (
            row["method_reference"] != "td3"
            or row["seed_interpretation"] != seed_text
            or row["generalization_scope"] != scope_text
        ):
            raise AssertionError(f"P1 method-contrast semantics mismatch: {key}")
        if level == "cell":
            selected = [(str(environment), float(tau), int(seed))]
        elif level == "task_budget_seed_mean":
            selected = [(str(environment), float(tau), current) for current in SEEDS]
        elif level == "task_marginal_mean":
            selected = [
                (str(environment), current_tau, current_seed)
                for current_tau in TAUS
                for current_seed in SEEDS
            ]
        else:
            selected = list(product(ENVIRONMENTS, TAUS, SEEDS))
        for output, source in metrics.items():
            expected = _mean(
                difference(
                    str(method), env, current_tau, current_seed, str(scope), source
                )
                for env, current_tau, current_seed in selected
            )
            _close(
                _finite(row[output], f"{key} {output}"), expected,
                f"P1 method contrast mismatch: {key} {output}",
            )


def _verify_residual_aggregates(
    residual_rows: Sequence[Mapping[str, str]],
    rows: Sequence[Mapping[str, str]],
) -> None:
    groups: dict[tuple[str, int, float, str, str], list[Mapping[str, str]]] = {}
    for row in residual_rows:
        key = (
            row["method"], _integer(row["hop"], "aggregate source hop"),
            _finite(row["tau"], "aggregate source tau"), row["environment"],
            row["stability"],
        )
        groups.setdefault(key, []).append(row)
    actual = {
        (
            row["method"], _integer(row["hop"], "aggregate hop"),
            _finite(row["tau"], "aggregate tau"), row["environment"], row["stability"],
        ): row
        for row in rows
    }
    if len(rows) != 256 or len(actual) != 256 or set(actual) != set(groups):
        raise AssertionError("P1 residual-aggregate exact key grid mismatch")
    sources = {
        "r_before_mean": "r_before", "slack_before_mean": "slack_before",
        "r_after_mean": "r_after", "slack_after_mean": "slack_after",
    }
    for key, selected in groups.items():
        row = actual[key]
        if _integer(row["n_seed_cells"], f"{key} n_seed_cells") != len(selected):
            raise AssertionError(f"P1 residual aggregate count mismatch: {key}")
        for output, source in sources.items():
            _close(
                _finite(row[output], f"{key} {output}"),
                _mean(_finite(item[source], f"{key} {source}") for item in selected),
                f"P1 residual aggregate mismatch: {key} {output}",
            )


def _verify_geometry_correlations(
    geometry: Mapping[tuple[str, str, float, int], Mapping[str, str]],
    value: Mapping[tuple[str, str, float, int, str], Mapping[str, str]],
    rows: Sequence[Mapping[str, str]],
) -> None:
    actual = {(row["method"], row["stability"]): row for row in rows}
    expected_keys = set(product(METHOD_HOPS, ("all", "stable", "collapsed")))
    if len(rows) != 9 or len(actual) != 9 or set(actual) != expected_keys:
        raise AssertionError("P1 geometry-correlation exact key grid mismatch")
    for key, row in actual.items():
        method, stability = key
        selected = [
            (base_key, item)
            for base_key, item in geometry.items()
            if base_key[0] == method
            and (stability == "all" or item["stability"] == stability)
        ]
        x = np.asarray(
            [
                _finite(item["final_target_ratio"], "correlation ratio")
                for _, item in selected
            ]
        )
        y = np.asarray([
            _finite(value[(*base_key, OWN_SCOPE)]["delta_y_rms"], "correlation delta")
            for base_key, _ in selected
        ])
        expected = float(np.corrcoef(x, y)[0, 1])
        if _integer(row["n_cells"], f"{key} n_cells") != len(selected):
            raise AssertionError(f"P1 geometry correlation count mismatch: {key}")
        _close(
            _finite(
                row["pearson_final_target_ratio_vs_within_run_delta_y_rms"],
                f"{key} correlation",
            ),
            expected,
            f"P1 geometry correlation mismatch: {key}",
        )
        if row["external_stability_definition"] != "normalized return <20 is collapsed":
            raise AssertionError(f"P1 geometry correlation label mismatch: {key}")


def _verify_headlines(
    geometry: Mapping[tuple[str, str, float, int], Mapping[str, str]],
    value: Mapping[tuple[str, str, float, int, str], Mapping[str, str]],
    residual_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    cell_keys = list(product(ENVIRONMENTS, TAUS, SEEDS))
    report: dict[str, object] = {}
    for method in ("p3", "p4"):
        target_ratios = []
        final_ratios = []
        final_over_target = []
        for environment, tau, seed in cell_keys:
            baseline = geometry[("td3", environment, tau, seed)]
            candidate = geometry[(method, environment, tau, seed)]
            target_ratios.append(
                (
                    _finite(candidate["target_to_next_data_rms"], "candidate target")
                    / _finite(baseline["target_to_next_data_rms"], "TD3 target")
                )
                ** 2
            )
            final_ratios.append(
                (
                    _finite(candidate["final_to_next_data_rms"], "candidate final")
                    / _finite(baseline["final_to_next_data_rms"], "TD3 final")
                )
                ** 2
            )
            final_over_target.append(
                _finite(candidate["final_target_ratio"], "candidate final/target")
            )
        observed_geometry = {
            "target": (
                sum(value_ < 1.0 for value_ in target_ratios),
                statistics.median(target_ratios),
            ),
            "final": (
                sum(value_ < 1.0 for value_ in final_ratios),
                statistics.median(final_ratios),
            ),
            "final_over_target": (
                sum(value_ > 1.0 for value_ in final_over_target),
                statistics.median(final_over_target),
            ),
        }
        for label, (count, median) in observed_geometry.items():
            expected_count, expected_median = HEADLINE[method][label]
            if count != expected_count:
                raise AssertionError(f"P1 {method} {label} count mismatch")
            _strict_close(median, expected_median, f"P1 {method} {label} median mismatch")

        common_results = {}
        for metric, label in (
            ("delta_y_rms", "common"),
            ("delta_y_rms_no_noise", "common_no_noise"),
        ):
            ratios = []
            task_medians = []
            for environment in ENVIRONMENTS:
                task_ratios = []
                for tau, seed in product(TAUS, SEEDS):
                    candidate = _finite(
                        value[(method, environment, tau, seed, COMMON_SCOPE)][metric],
                        f"{method} common {metric}",
                    )
                    baseline = _finite(
                        value[("td3", environment, tau, seed, COMMON_SCOPE)][metric],
                        f"TD3 common {metric}",
                    )
                    if baseline <= 0.0:
                        raise AssertionError(
                            "P1 zero common-critic denominator: "
                            f"{(environment, tau, seed)}"
                        )
                    task_ratios.append(candidate / baseline)
                ratios.extend(task_ratios)
                task_medians.append(statistics.median(task_ratios))
            count = sum(ratio < 1.0 for ratio in ratios)
            median = statistics.median(ratios)
            expected_count, expected_median = HEADLINE[method][label]
            if count != expected_count or len(task_medians) != 9 or not all(
                task_median < 1.0 for task_median in task_medians
            ):
                raise AssertionError(f"P1 {method} {label} count/task-median mismatch")
            _strict_close(median, expected_median, f"P1 {method} {label} median mismatch")
            common_results[label] = {
                "lower": count,
                "median_ratio": median,
                "task_medians_below_one": sum(item < 1.0 for item in task_medians),
            }

        method_residuals = [row for row in residual_rows if row["method"] == method]
        slack_before = [_finite(row["slack_before"], f"{method} slack") for row in method_residuals]
        positive = [row for row in method_residuals if _finite(row["slack_before"], "slack") > 0.0]
        zero_slack = sum(slack == 0.0 for slack in slack_before)
        decreases = sum(
            _finite(row["slack_after"], "slack after")
            < _finite(row["slack_before"], "slack before")
            for row in positive
        )
        if zero_slack != HEADLINE[method]["zero_slack"]:
            raise AssertionError(f"P1 {method} zero-slack count mismatch")
        _strict_close(
            statistics.median(slack_before),
            HEADLINE[method]["median_slack"],
            f"P1 {method} median slack mismatch",
        )
        if (decreases, len(positive)) != HEADLINE[method]["positive_slack_decreases"]:
            raise AssertionError(f"P1 {method} positive-slack decrease count mismatch")
        report[method] = {
            **observed_geometry,
            **common_results,
            "zero_slack": zero_slack,
            "median_slack": statistics.median(slack_before),
            "positive_slack_decreases": (decreases, len(positive)),
        }

    td3_own = [
        value[("td3", environment, tau, seed, OWN_SCOPE)]
        for environment, tau, seed in cell_keys
    ]
    explosions = [
        row
        for row in td3_own
        if _finite(row["q_dataset_abs_mean"], "TD3 critic scale") > 1e9
    ]
    if len(explosions) != HEADLINE["td3_critic_explosions"] or any(
        row["stability"] != "collapsed" for row in explosions
    ):
        raise AssertionError("P1 TD3 critic-explosion threshold/count mismatch")
    critic_scale = np.asarray(
        [_finite(row["q_dataset_abs_mean"], "TD3 critic scale") for row in td3_own]
    )
    delta = np.asarray(
        [_finite(row["delta_y_rms"], "TD3 delta_y_rms") for row in td3_own]
    )
    if np.any(critic_scale <= 0.0) or np.any(delta <= 0.0):
        raise AssertionError("P1 log critic-scale correlation has nonpositive input")
    correlation = float(np.corrcoef(np.log(critic_scale), np.log(delta))[0, 1])
    _strict_close(
        correlation,
        HEADLINE["td3_log_scale_delta_correlation"],
        "P1 TD3 log critic-scale/delta correlation mismatch",
    )
    report["td3_critic_explosions"] = len(explosions)
    report["td3_log_scale_delta_correlation"] = correlation
    return report


def verify_p1_target_value_compact(diagnostics_dir: Path) -> dict[str, object]:
    """Verify retained P1 arithmetic and return the recomputed headline readout."""
    compact_dir = diagnostics_dir / "p1_target_value_audit"
    repo = diagnostics_dir.parents[1]
    protocol, manifest, _ = _verify_metadata(compact_dir, repo)
    tables = {name: _read_csv(compact_dir / name) for name in CSV_FIELDS}
    geometry, value = _verify_primary_rows(
        protocol,
        manifest,
        tables["same_next_state_geometry.csv"],
        tables["td_target_value.csv"],
        tables["comparator_residuals.csv"],
    )
    _verify_method_contrasts(value, tables["method_contrasts.csv"])
    _verify_residual_aggregates(
        tables["comparator_residuals.csv"],
        tables["residual_aggregates.csv"],
    )
    _verify_geometry_correlations(
        geometry,
        value,
        tables["geometry_correlations.csv"],
    )
    report = _verify_headlines(
        geometry,
        value,
        tables["comparator_residuals.csv"],
    )
    print(
        "PASS P1 compact: exact 270-cell grid, 540 value rows, 270 geometry "
        "rows, 450 residual rows, derived tables, headline ratios, critic tail, "
        "and verification-gated interpretation limits"
    )
    return report
