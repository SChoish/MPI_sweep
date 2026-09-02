#!/usr/bin/env python3
"""Independent artifact-only verifier for the strict P1 CPU audit.

This module deliberately does not import the analysis runner or checkpoint
model code. It reads only frozen JSON/CSV/NPZ artifacts and independently
recomputes hashes, keys, clipping geometry, TD-target RMS, and residuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
TAUS = (4.0, 7.0, 10.0, 14.0, 20.0)
SEEDS = (0, 1)
METHOD_HOPS = {"td3": 1, "p3": 3, "p4": 4}
CHECKPOINT_STEP = 1_000_000
COLLAPSE_THRESHOLD = 20.0
REQUIRED_CONFIG = {
    "normalize": True,
    "q_scale_norm": True,
    "integrator": "implicit",
    "max_timesteps": CHECKPOINT_STEP,
    "eval_episodes": 10,
    "batch_size": 256,
    "discount": 0.99,
    "polyak": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "lr": 3e-4,
    "n_jitted_updates": 8,
}
OPTIONAL_CONFIG_IF_PRESENT = {"method": "bar"}
OWN_SCOPE = "within_run_target_critic"
COMMON_SCOPE = "common_td3_target_critic"
RESIDUAL_SCOPE = "within_run_online_critic"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(named_arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named_arrays):
        array = np.ascontiguousarray(np.asarray(named_arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def check_file_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise AssertionError(f"missing artifact: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AssertionError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"empty CSV: {path}")
    return rows


def _cell_key(method: str, environment: str, tau: float, seed: int) -> str:
    return f"{method}|{environment}|{float(tau):g}|{int(seed)}"


def _expected_keys() -> set[str]:
    return {
        _cell_key(method, environment, tau, seed)
        for environment in ENVIRONMENTS
        for tau in TAUS
        for seed in SEEDS
        for method in METHOD_HOPS
    }


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def _run_name(method: str, environment: str, tau: float, seed: int) -> str:
    tags = {"td3": "", "p3": "mpi3", "p4": "mpi4"}
    middle = f"_{tags[method]}" if tags[method] else ""
    return f"{environment}_tau{float(tau):g}{middle}_seed{int(seed)}"


def _expected_record_metadata() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for environment in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS:
                for method, hops in METHOD_HOPS.items():
                    key = _cell_key(method, environment, tau, seed)
                    expected[key] = {
                        "key": key,
                        "method": method,
                        "environment": environment,
                        "tau": float(tau),
                        "seed": int(seed),
                        "hops": int(hops),
                        "run_name": _run_name(method, environment, tau, seed),
                    }
    return expected


def _config_contract_errors(
    config: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    locked = {
        "env": expected["environment"],
        "seed": expected["seed"],
        "tau": expected["tau"],
        "mpi_steps": expected["hops"],
        **REQUIRED_CONFIG,
    }
    errors: list[str] = []
    for key, value in locked.items():
        if key not in config:
            errors.append(f"missing config key {key}")
        elif not _same_value(config[key], value):
            errors.append(f"config {key}={config[key]!r} != {value!r}")
    for key, value in OPTIONAL_CONFIG_IF_PRESENT.items():
        if key in config and not _same_value(config[key], value):
            errors.append(f"config {key}={config[key]!r} != {value!r}")
    return errors


def _bool(text: Any) -> bool:
    if isinstance(text, bool):
        return text
    normalized = str(text).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise AssertionError(f"invalid boolean: {text!r}")


def _close(actual: float, expected: float, label: str, tolerance: float = 2e-6) -> None:
    if np.isnan(actual) and np.isnan(expected):
        return
    scale = max(1.0, abs(actual), abs(expected))
    if not np.isfinite(actual) or not np.isfinite(expected):
        if actual != expected:
            raise AssertionError(f"{label}: nonfinite mismatch {actual} != {expected}")
        return
    if abs(actual - expected) > tolerance * scale:
        raise AssertionError(f"{label}: {actual} != {expected}")


def _verify_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "analysis_complete":
        raise AssertionError("manifest does not declare a complete analysis")
    if manifest.get("cpu_only") is not True:
        raise AssertionError("manifest does not declare CPU-only analysis")
    if manifest.get("jax_backend") != "cpu":
        raise AssertionError("manifest JAX backend is not CPU")


def _verify_verifier_source(provenance: Mapping[str, Any]) -> str:
    sources = provenance.get("sources", {})
    expected = str(sources.get("verifier", {}).get("sha256", ""))
    actual = sha256_file(Path(__file__).resolve())
    if actual != expected:
        raise AssertionError(
            f"executing verifier source hash differs from frozen provenance: {actual} != {expected}"
        )
    return actual


def _verify_record_contract(
    key: str, row: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for field in ("key", "method", "environment", "seed", "hops", "run_name"):
        if not _same_value(row.get(field), expected[field]):
            raise AssertionError(f"fixed record metadata mismatch: {key} {field}")
    if not _same_value(row.get("tau"), expected["tau"]):
        raise AssertionError(f"fixed record metadata mismatch: {key} tau")
    if int(row.get("checkpoint_step", -1)) != CHECKPOINT_STEP:
        raise AssertionError(f"checkpoint step mismatch: {key}")
    if int(row.get("external_score_step", -1)) != CHECKPOINT_STEP:
        raise AssertionError(f"external score step mismatch: {key}")
    config = row.get("config")
    if not isinstance(config, Mapping):
        raise AssertionError(f"checkpoint config is missing or invalid: {key}")
    config_errors = _config_contract_errors(config, expected)
    if config_errors:
        raise AssertionError(f"checkpoint config mismatch: {key}: {'; '.join(config_errors)}")
    config_hash = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    if config_hash != row.get("config_sha256"):
        raise AssertionError(f"checkpoint config hash mismatch: {key}")
    expected_names = {
        "checkpoint_path": f"params_{CHECKPOINT_STEP}.pkl",
        "config_path": "config.json",
        "eval_path": "eval.csv",
    }
    resolved_paths: dict[str, Path] = {}
    for field, filename in expected_names.items():
        path = Path(str(row.get(field, "")))
        if not path.is_absolute() or path.name != filename:
            raise AssertionError(f"checkpoint companion path mismatch: {key} {field}")
        resolved_paths[field] = path
    run_dir = Path(str(row.get("run_dir", "")))
    if not run_dir.is_absolute() or any(
        path.parent != run_dir for path in resolved_paths.values()
    ):
        raise AssertionError(f"checkpoint companion directories differ: {key}")
    external_score = float(row.get("external_score", float("nan")))
    if not np.isfinite(external_score):
        raise AssertionError(f"external score is nonfinite: {key}")
    expected_collapsed = external_score < COLLAPSE_THRESHOLD
    if _bool(row.get("collapsed_lt20")) != expected_collapsed:
        raise AssertionError(f"collapse label mismatch: {key}")
    max_action = float(row.get("max_action", float("nan")))
    policy_noise = float(row.get("policy_noise", float("nan")))
    noise_clip = float(row.get("noise_clip", float("nan")))
    if not np.isfinite(max_action) or max_action <= 0.0:
        raise AssertionError(f"max_action is invalid: {key}")
    _close(
        policy_noise,
        float(config["policy_noise"]) * max_action,
        f"{key} policy_noise",
        tolerance=1e-12,
    )
    _close(
        noise_clip,
        float(config["noise_clip"]) * max_action,
        f"{key} noise_clip",
        tolerance=1e-12,
    )
    _close(float(row.get("discount")), float(config["discount"]), f"{key} discount")
    _close(
        float(row.get("learning_rate")),
        float(config["lr"]),
        f"{key} learning_rate",
        tolerance=1e-12,
    )
    if not row.get("resolved") or not row.get("residual_supported"):
        raise AssertionError(f"unsupported inventory cell: {key}")
    if row.get("residual_intervention") != "posthoc_final_checkpoint_one_adam_step":
        raise AssertionError(f"residual intervention mismatch: {key}")
    optimizer = row.get("optimizer_reconstruction", {})
    if optimizer.get("optimizer") != "optax.adam with imported training defaults":
        raise AssertionError(f"optimizer identity mismatch: {key}")
    _close(
        float(optimizer.get("learning_rate")),
        float(config["lr"]),
        f"{key} optimizer learning rate",
        tolerance=1e-12,
    )
    expected_actor_step = CHECKPOINT_STEP // int(config["policy_freq"])
    if int(optimizer.get("expected_actor_step", -1)) != expected_actor_step:
        raise AssertionError(f"expected optimizer step mismatch: {key}")
    actors = optimizer.get("actors", ())
    if len(actors) != int(expected["hops"]):
        raise AssertionError(f"optimizer actor count mismatch: {key}")
    for index, actor in enumerate(actors, start=1):
        if (
            int(actor.get("actor", -1)) != index
            or int(actor.get("actor_step", -1)) != expected_actor_step
            or int(actor.get("adam_count", -1)) != expected_actor_step
            or not _is_sha256(actor.get("optimizer_state_sha256"))
        ):
            raise AssertionError(f"optimizer reconstruction mismatch: {key} actor {index}")
    if not _is_sha256(optimizer.get("all_optimizer_states_sha256")):
        raise AssertionError(f"optimizer-state bundle hash missing: {key}")


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(left, dtype=np.float64)
                    - np.asarray(right, dtype=np.float64)
                )
            )
        )
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        return {name: np.asarray(pack[name]) for name in pack.files}


def _resolve_declared_path(base: Path, declared: str) -> Path:
    path = Path(declared)
    if path.is_absolute() or ".." in path.parts:
        raise AssertionError(f"artifact path is not bundle-relative: {declared}")
    candidate = (base / path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise AssertionError(f"artifact escapes bundle: {declared}") from error
    return candidate


def _verify_common_contracts(
    out_dir: Path, inventory: Mapping[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    contracts = inventory["common_contracts"]
    if set(contracts) != set(ENVIRONMENTS):
        raise AssertionError("common contract environments do not match the fixed grid")
    arrays_by_env: dict[str, dict[str, np.ndarray]] = {}
    transition_suffixes = (
        "observations",
        "actions",
        "next_observations",
        "next_actions",
        "rewards",
        "source_indices",
    )
    expected_array_names = {
        f"{prefix}_{suffix}"
        for prefix in ("value", "residual")
        for suffix in transition_suffixes
    } | {"standard_noise", "epsilon"}
    for environment in ENVIRONMENTS:
        contract = contracts[environment]
        path = _resolve_declared_path(out_dir, str(contract["common_npz"]))
        arrays = _load_npz(path)
        if set(arrays) != expected_array_names:
            raise AssertionError(f"common NPZ schema mismatch: {environment}")
        if sha256_arrays(arrays) != contract["common_content_sha256"]:
            raise AssertionError(f"common content hash mismatch: {environment}")
        selected = {
            key: value
            for key, value in arrays.items()
            if key.startswith("value_") or key.startswith("residual_")
        }
        if sha256_arrays(selected) != contract["selected_transition_sha256"]:
            raise AssertionError(f"selected transition hash mismatch: {environment}")
        if sha256_arrays(
            {
                "standard_noise": arrays["standard_noise"],
                "epsilon": arrays["epsilon"],
            }
        ) != contract["common_noise_sha256"]:
            raise AssertionError(f"common noise hash mismatch: {environment}")
        value_indices = set(map(int, arrays["value_source_indices"].tolist()))
        residual_indices = set(map(int, arrays["residual_source_indices"].tolist()))
        if value_indices & residual_indices:
            raise AssertionError(f"fixed audit batches overlap: {environment}")
        if len(value_indices) != int(contract["value_rows"]):
            raise AssertionError(f"value audit row count mismatch: {environment}")
        if len(residual_indices) != int(contract["residual_audit_rows"]):
            raise AssertionError(f"residual audit row count mismatch: {environment}")
        for prefix, expected_rows in (
            ("value", int(contract["value_rows"])),
            ("residual", int(contract["residual_audit_rows"])),
        ):
            for suffix in transition_suffixes:
                if np.asarray(arrays[f"{prefix}_{suffix}"]).shape[0] != expected_rows:
                    raise AssertionError(
                        f"{prefix} transition shape mismatch: {environment} {suffix}"
                    )
        if arrays["epsilon"].shape != arrays["standard_noise"].shape:
            raise AssertionError(f"noise shape mismatch: {environment}")
        expected_epsilon = np.clip(
            arrays["standard_noise"] * float(contract["policy_noise"]),
            -float(contract["noise_clip"]),
            float(contract["noise_clip"]),
        )
        np.testing.assert_allclose(
            arrays["epsilon"], expected_epsilon, rtol=0.0, atol=0.0
        )
        arrays_by_env[environment] = arrays
    return arrays_by_env


def _verify_value_and_geometry(
    *,
    inventory_records: Mapping[str, Mapping[str, Any]],
    value_rows: Sequence[Mapping[str, str]],
    geometry_rows: Sequence[Mapping[str, str]],
    arrays_by_env: Mapping[str, Mapping[str, np.ndarray]],
    raw_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    expected = _expected_keys()
    if len(value_rows) != 540:
        raise AssertionError(f"value rows={len(value_rows)} != 540")
    value_keys = {(row["key"], row["critic_scope"]) for row in value_rows}
    expected_value_keys = {
        (key, scope) for key in expected for scope in (OWN_SCOPE, COMMON_SCOPE)
    }
    if value_keys != expected_value_keys:
        raise AssertionError("value CSV key/scope set is incomplete or duplicated")
    if len(geometry_rows) != 270 or {row["key"] for row in geometry_rows} != expected:
        raise AssertionError("geometry CSV must contain exactly one row per fixed cell")
    value_lookup = {(row["key"], row["critic_scope"]): row for row in value_rows}
    base_raw_names = {
        "gamma",
        "rewards",
        "epsilon",
        "target_actions",
        "final_actions",
        "dataset_next_actions",
        "smoothed_target_actions",
        "smoothed_dataset_actions",
        "deterministic_target_actions",
        "deterministic_dataset_actions",
        "residual_hops",
        "residual_q_weight",
        "residual_C_k",
        "residual_q_reference",
        "residual_q_before",
        "residual_q_after",
        "residual_transport_before",
        "residual_transport_after",
    }
    value_raw_suffixes = (
        "q_actor",
        "q_dataset",
        "delta_y",
        "target_actor",
        "target_dataset",
        "q_actor_no_noise",
        "q_dataset_no_noise",
        "delta_y_no_noise",
        "target_actor_no_noise",
        "target_dataset_no_noise",
    )
    expected_raw_names = base_raw_names | {
        f"{scope}_{suffix}"
        for scope in (OWN_SCOPE, COMMON_SCOPE)
        for suffix in value_raw_suffixes
    }
    for geometry in geometry_rows:
        key = geometry["key"]
        record = inventory_records[key]
        expected_stability = (
            "collapsed" if float(record["external_score"]) < 20.0 else "stable"
        )
        if geometry["stability"] != expected_stability:
            raise AssertionError(f"external stability label mismatch: {key}")
        environment = str(record["environment"])
        common = arrays_by_env[environment]
        raw = raw_cache[str(geometry["raw_npz"])]
        if set(raw) != expected_raw_names:
            raise AssertionError(f"raw NPZ schema mismatch: {key}")
        expected_hops = np.arange(
            2, METHOD_HOPS[str(record["method"])] + 1, dtype=np.int16
        )
        np.testing.assert_array_equal(raw["residual_hops"], expected_hops)
        for name in (
            "residual_q_weight",
            "residual_C_k",
            "residual_q_reference",
            "residual_q_before",
            "residual_q_after",
            "residual_transport_before",
            "residual_transport_after",
        ):
            if np.asarray(raw[name]).shape[0] != expected_hops.size:
                raise AssertionError(f"raw residual shape mismatch: {key} {name}")
        np.testing.assert_array_equal(raw["epsilon"], common["epsilon"])
        np.testing.assert_array_equal(
            raw["dataset_next_actions"], common["value_next_actions"]
        )
        max_action = float(record["max_action"])
        expected_smoothed_target = np.clip(
            raw["target_actions"] + raw["epsilon"], -max_action, max_action
        )
        expected_smoothed_dataset = np.clip(
            raw["dataset_next_actions"] + raw["epsilon"], -max_action, max_action
        )
        expected_deterministic_target = np.clip(
            raw["target_actions"], -max_action, max_action
        )
        expected_deterministic_dataset = np.clip(
            raw["dataset_next_actions"], -max_action, max_action
        )
        np.testing.assert_allclose(
            raw["smoothed_target_actions"],
            expected_smoothed_target,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            raw["smoothed_dataset_actions"],
            expected_smoothed_dataset,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            raw["deterministic_target_actions"],
            expected_deterministic_target,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            raw["deterministic_dataset_actions"],
            expected_deterministic_dataset,
            rtol=0.0,
            atol=0.0,
        )
        target_rms = _rms(raw["target_actions"], raw["dataset_next_actions"])
        final_rms = _rms(raw["final_actions"], raw["dataset_next_actions"])
        n_rows, action_dim = np.asarray(raw["target_actions"]).shape
        if int(geometry["n_rows"]) != n_rows or int(geometry["action_dim"]) != action_dim:
            raise AssertionError(f"{key} geometry dimensions do not match raw arrays")
        _close(float(geometry["target_to_next_data_rms"]), target_rms, f"{key} target")
        _close(float(geometry["final_to_next_data_rms"]), final_rms, f"{key} final")
        _close(
            float(geometry["final_to_target_rms"]),
            _rms(raw["final_actions"], raw["target_actions"]),
            f"{key} final-target",
        )
        _close(
            float(geometry["smoothed_target_to_next_data_rms"]),
            _rms(raw["smoothed_target_actions"], raw["smoothed_dataset_actions"]),
            f"{key} smoothed",
        )
        ratio = final_rms / target_rms if target_rms > 0 else float("nan")
        _close(float(geometry["final_target_ratio"]), ratio, f"{key} ratio")
        saturation_metrics = {
            "target_saturation_fraction": np.mean(
                np.abs(raw["target_actions"]) >= max_action
            ),
            "final_saturation_fraction": np.mean(
                np.abs(raw["final_actions"]) >= max_action
            ),
            "smoothed_target_saturation_fraction": np.mean(
                np.abs(raw["smoothed_target_actions"]) >= max_action
            ),
            "smoothed_dataset_saturation_fraction": np.mean(
                np.abs(raw["smoothed_dataset_actions"]) >= max_action
            ),
        }
        for field, value in saturation_metrics.items():
            _close(float(geometry[field]), float(value), f"{key} {field}")
        geometry_finite = bool(
            all(
                np.all(np.isfinite(raw[name]))
                for name in (
                    "target_actions",
                    "final_actions",
                    "dataset_next_actions",
                    "smoothed_target_actions",
                    "smoothed_dataset_actions",
                )
            )
            and np.isfinite(ratio)
        )
        if _bool(geometry["all_finite"]) != geometry_finite:
            raise AssertionError(f"{key} geometry finite flag mismatch")
        gamma = float(np.asarray(raw["gamma"]))
        _close(gamma, float(record["discount"]), f"{key} raw gamma")
        rewards = np.asarray(raw["rewards"], dtype=np.float64).reshape(-1)
        for scope in (OWN_SCOPE, COMMON_SCOPE):
            row = value_lookup[(key, scope)]
            if row["stability"] != expected_stability:
                raise AssertionError(f"value stability label mismatch: {key}")
            expected_critic_hash = (
                record["target_critic_sha256"]
                if scope == OWN_SCOPE
                else inventory_records[
                    _cell_key(
                        "td3",
                        str(record["environment"]),
                        float(record["tau"]),
                        int(record["seed"]),
                    )
                ]["target_critic_sha256"]
            )
            if row["critic_params_sha256"] != expected_critic_hash:
                raise AssertionError(f"value critic fingerprint mismatch: {key} {scope}")
            _close(float(row["gamma"]), gamma, f"{key} {scope} gamma")
            q_actor = np.asarray(raw[f"{scope}_q_actor"], dtype=np.float64)
            q_dataset = np.asarray(raw[f"{scope}_q_dataset"], dtype=np.float64)
            delta = gamma * (q_actor - q_dataset)
            target_actor = rewards + gamma * q_actor
            target_dataset = rewards + gamma * q_dataset
            q_actor_no_noise = np.asarray(
                raw[f"{scope}_q_actor_no_noise"], dtype=np.float64
            )
            q_dataset_no_noise = np.asarray(
                raw[f"{scope}_q_dataset_no_noise"], dtype=np.float64
            )
            delta_no_noise = gamma * (q_actor_no_noise - q_dataset_no_noise)
            target_actor_no_noise = rewards + gamma * q_actor_no_noise
            target_dataset_no_noise = rewards + gamma * q_dataset_no_noise
            np.testing.assert_allclose(
                raw[f"{scope}_delta_y"], delta, rtol=2e-6, atol=2e-6
            )
            np.testing.assert_allclose(
                raw[f"{scope}_target_actor"], target_actor, rtol=2e-6, atol=2e-6
            )
            np.testing.assert_allclose(
                raw[f"{scope}_target_dataset"],
                target_dataset,
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                raw[f"{scope}_delta_y_no_noise"],
                delta_no_noise,
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                raw[f"{scope}_target_actor_no_noise"],
                target_actor_no_noise,
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                raw[f"{scope}_target_dataset_no_noise"],
                target_dataset_no_noise,
                rtol=2e-6,
                atol=2e-6,
            )
            expected_metrics = {
                "delta_y_rms": np.sqrt(np.mean(np.square(delta))),
                "delta_y_abs_mean": np.mean(np.abs(delta)),
                "target_actor_rms": np.sqrt(np.mean(np.square(target_actor))),
                "target_dataset_rms": np.sqrt(np.mean(np.square(target_dataset))),
                "target_actor_abs_mean": np.mean(np.abs(target_actor)),
                "target_dataset_abs_mean": np.mean(np.abs(target_dataset)),
                "q_actor_abs_mean": np.mean(np.abs(q_actor)),
                "q_dataset_abs_mean": np.mean(np.abs(q_dataset)),
            }
            for field, value in expected_metrics.items():
                _close(float(row[field]), float(value), f"{key} {scope} {field}")
            no_noise_rms = float(np.sqrt(np.mean(np.square(delta_no_noise))))
            no_noise_abs_mean = float(np.mean(np.abs(delta_no_noise)))
            _close(
                float(row["delta_y_rms_no_noise"]),
                no_noise_rms,
                f"{key} {scope} delta_y_rms_no_noise",
            )
            _close(
                float(row["delta_y_abs_mean_no_noise"]),
                no_noise_abs_mean,
                f"{key} {scope} delta_y_abs_mean_no_noise",
            )
            smoothing_change = float(expected_metrics["delta_y_rms"]) - no_noise_rms
            smoothing_ratio = (
                float(expected_metrics["delta_y_rms"]) / no_noise_rms
                if no_noise_rms > 0.0
                else float("nan")
            )
            _close(
                float(row["smoothing_delta_y_rms_change"]),
                smoothing_change,
                f"{key} {scope} smoothing change",
            )
            _close(
                float(row["smoothing_delta_y_rms_ratio"]),
                smoothing_ratio,
                f"{key} {scope} smoothing ratio",
            )
            clip_metrics = {
                "action_clip_target_fraction": np.mean(
                    np.abs(raw["target_actions"] + raw["epsilon"]) > max_action
                ),
                "action_clip_dataset_fraction": np.mean(
                    np.abs(raw["dataset_next_actions"] + raw["epsilon"])
                    > max_action
                ),
                "action_clip_target_fraction_no_noise": np.mean(
                    np.abs(raw["target_actions"]) > max_action
                ),
                "action_clip_dataset_fraction_no_noise": np.mean(
                    np.abs(raw["dataset_next_actions"]) > max_action
                ),
            }
            for field, value in clip_metrics.items():
                _close(float(row[field]), float(value), f"{key} {scope} {field}")
            no_noise_finite = bool(
                all(
                    np.all(np.isfinite(value))
                    for value in (
                        delta_no_noise,
                        target_actor_no_noise,
                        target_dataset_no_noise,
                    )
                )
            )
            if _bool(row["no_noise_all_finite"]) != no_noise_finite:
                raise AssertionError(f"{key} {scope} no-noise finite flag mismatch")
            expected_all_finite = bool(
                no_noise_finite
                and all(
                    np.all(np.isfinite(value))
                    for value in (delta, target_actor, target_dataset)
                )
                and np.isfinite(smoothing_change)
                and np.isfinite(smoothing_ratio)
            )
            if _bool(row["all_finite"]) != expected_all_finite:
                raise AssertionError(f"{key} {scope} finite flag mismatch")
            if int(row["n_rows"]) != rewards.size:
                raise AssertionError(f"{key} {scope} row count mismatch")
    for environment in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS:
                td3 = _cell_key("td3", environment, tau, seed)
                own = value_lookup[(td3, OWN_SCOPE)]
                common = value_lookup[(td3, COMMON_SCOPE)]
                if own["critic_params_sha256"] != common["critic_params_sha256"]:
                    raise AssertionError(f"TD3 common critic identity mismatch: {td3}")
                raw = raw_cache[str(own["raw_npz"])]
                np.testing.assert_array_equal(
                    raw[f"{OWN_SCOPE}_q_actor"], raw[f"{COMMON_SCOPE}_q_actor"]
                )
                np.testing.assert_array_equal(
                    raw[f"{OWN_SCOPE}_q_dataset"], raw[f"{COMMON_SCOPE}_q_dataset"]
                )
                np.testing.assert_array_equal(
                    raw[f"{OWN_SCOPE}_q_actor_no_noise"],
                    raw[f"{COMMON_SCOPE}_q_actor_no_noise"],
                )
                np.testing.assert_array_equal(
                    raw[f"{OWN_SCOPE}_q_dataset_no_noise"],
                    raw[f"{COMMON_SCOPE}_q_dataset_no_noise"],
                )
                common_hash = common["critic_params_sha256"]
                for method in ("p3", "p4"):
                    key = _cell_key(method, environment, tau, seed)
                    if value_lookup[(key, COMMON_SCOPE)]["critic_params_sha256"] != common_hash:
                        raise AssertionError(f"common critic differs: {key}")


def _verify_residuals(
    residual_rows: Sequence[Mapping[str, str]],
    raw_cache: Mapping[str, Mapping[str, np.ndarray]],
    inventory_records: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(residual_rows) != 450:
        raise AssertionError(f"residual rows={len(residual_rows)} != 450")
    expected = {
        (_cell_key(method, environment, tau, seed), hop)
        for environment in ENVIRONMENTS
        for tau in TAUS
        for seed in SEEDS
        for method, total_hops in METHOD_HOPS.items()
        for hop in range(2, total_hops + 1)
    }
    actual = {(row["key"], int(row["hop"])) for row in residual_rows}
    if actual != expected or any(hop < 2 for _, hop in actual):
        raise AssertionError("residual keys incomplete, duplicated, or include k=1")
    for row in residual_rows:
        if row["critic_scope"] != RESIDUAL_SCOPE:
            raise AssertionError("residual used an unexpected critic scope")
        if row["critic_params_sha256"] != inventory_records[row["key"]][
            "online_critic_sha256"
        ]:
            raise AssertionError(f"residual critic fingerprint mismatch: {row['key']}")
        if row["intervention"] != "posthoc_final_checkpoint_one_adam_step":
            raise AssertionError("residual intervention is mislabeled")
        if _bool(row["historical_training_step_recovered"]):
            raise AssertionError("artifact claims historical live-step recovery")
        expected_stability = (
            "collapsed" if float(row["external_score"]) < 20.0 else "stable"
        )
        if row["stability"] != expected_stability:
            raise AssertionError(f"residual stability label mismatch: {row['key']}")
        raw = raw_cache[str(row["raw_npz"])]
        hop = int(row["hop"])
        total_hops = METHOD_HOPS[row["method"]]
        if int(row["total_hops"]) != total_hops:
            raise AssertionError(f"total-hop mismatch: {row['key']} k={hop}")
        tau_step = float(row["tau"]) / total_hops
        _close(float(row["tau_step"]), tau_step, f"{row['key']} k={hop} tau_step")
        positions = np.flatnonzero(raw["residual_hops"] == hop)
        if positions.size != 1:
            raise AssertionError(f"raw residual hop mismatch: {row['key']} k={hop}")
        index = int(positions[0])
        q_ref = np.asarray(raw["residual_q_reference"][index], dtype=np.float64)
        q_before = np.asarray(raw["residual_q_before"][index], dtype=np.float64)
        q_after = np.asarray(raw["residual_q_after"][index], dtype=np.float64)
        tr_before = np.asarray(raw["residual_transport_before"][index], dtype=np.float64)
        tr_after = np.asarray(raw["residual_transport_after"][index], dtype=np.float64)
        C_k = float(np.mean(np.abs(q_ref))) + 1e-6
        q_weight = 2.0 * tau_step / C_k
        q_mean_before = float(np.mean(q_before))
        q_mean_after = float(np.mean(q_after))
        reference_q_mean = float(np.mean(q_ref))
        transport_before = float(np.mean(tr_before))
        transport_after = float(np.mean(tr_after))
        actor_loss_before = -q_weight * q_mean_before + transport_before
        actor_loss_after = -q_weight * q_mean_after + transport_after
        comparator_loss = -q_weight * reference_q_mean
        r_before = actor_loss_before - comparator_loss
        r_after = actor_loss_after - comparator_loss
        _close(
            float(np.asarray(raw["residual_C_k"])[index]),
            C_k,
            f"{row['key']} k={hop} raw C_k",
        )
        _close(
            float(np.asarray(raw["residual_q_weight"])[index]),
            q_weight,
            f"{row['key']} k={hop} raw q_weight",
        )
        _close(float(row["C_k"]), C_k, f"{row['key']} k={hop} C_k")
        _close(float(row["q_weight"]), q_weight, f"{row['key']} k={hop} q_weight")
        _close(float(row["r_before"]), float(r_before), f"{row['key']} k={hop} r_before")
        _close(float(row["r_after"]), float(r_after), f"{row['key']} k={hop} r_after")
        _close(float(row["slack_before"]), max(0.0, float(r_before)), "slack before")
        _close(float(row["slack_after"]), max(0.0, float(r_after)), "slack after")
        expected_fields = {
            "actor_loss_before": actor_loss_before,
            "actor_loss_after": actor_loss_after,
            "comparator_loss": comparator_loss,
            "q_mean_before": q_mean_before,
            "q_mean_after": q_mean_after,
            "reference_q_mean": reference_q_mean,
            "transport_before": transport_before,
            "transport_after": transport_after,
        }
        for field, value in expected_fields.items():
            _close(float(row[field]), value, f"{row['key']} k={hop} {field}")
        optimizer_actors = inventory_records[row["key"]]["optimizer_reconstruction"][
            "actors"
        ]
        expected_step = int(optimizer_actors[hop - 1]["actor_step"])
        if int(row["optimizer_step_before"]) != expected_step:
            raise AssertionError(f"stored optimizer step mismatch: {row['key']} k={hop}")
        if int(row["optimizer_step_after"]) != int(row["optimizer_step_before"]) + 1:
            raise AssertionError(f"not exactly one Adam step: {row['key']} k={hop}")
        finite = bool(
            all(
                np.all(np.isfinite(value))
                for value in (q_ref, q_before, q_after, tr_before, tr_after)
            )
            and all(np.isfinite(value) for value in expected_fields.values())
            and all(
                np.isfinite(value)
                for value in (C_k, q_weight, r_before, r_after)
            )
        )
        if _bool(row["all_finite"]) != finite:
            raise AssertionError(f"residual finite flag mismatch: {row['key']} k={hop}")


def _verify_method_contrasts(
    value_rows: Sequence[Mapping[str, str]],
    contrast_rows: Sequence[Mapping[str, str]],
) -> None:
    value = {
        (
            row["method"],
            row["environment"],
            float(row["tau"]),
            int(row["seed"]),
            row["critic_scope"],
        ): row
        for row in value_rows
    }
    metric_sources = {
        "delta_y_rms_difference": "delta_y_rms",
        "delta_y_rms_no_noise_difference": "delta_y_rms_no_noise",
        "smoothing_delta_y_rms_change_difference": "smoothing_delta_y_rms_change",
    }
    seed_text = "seeds are marginal run replicates within fixed task-budget cells"
    scope_text = "fixed nine-task, five-budget grid only"
    expected: list[dict[str, Any]] = []
    for method in ("p3", "p4"):
        for scope in (OWN_SCOPE, COMMON_SCOPE):
            cells: list[dict[str, Any]] = []
            for environment in ENVIRONMENTS:
                for tau in TAUS:
                    for seed in SEEDS:
                        method_row = value[(method, environment, tau, seed, scope)]
                        reference_row = value[("td3", environment, tau, seed, scope)]
                        cells.append(
                            {
                                "level": "cell",
                                "method_minus": method,
                                "method_reference": "td3",
                                "critic_scope": scope,
                                "environment": environment,
                                "tau": tau,
                                "seed": seed,
                                **{
                                    output: float(method_row[source])
                                    - float(reference_row[source])
                                    for output, source in metric_sources.items()
                                },
                            }
                        )
            expected.extend(cells)
            task_budget: list[dict[str, Any]] = []
            for environment in ENVIRONMENTS:
                for tau in TAUS:
                    selected = [
                        row
                        for row in cells
                        if row["environment"] == environment and row["tau"] == tau
                    ]
                    task_budget.append(
                        {
                            "level": "task_budget_seed_mean",
                            "method_minus": method,
                            "method_reference": "td3",
                            "critic_scope": scope,
                            "environment": environment,
                            "tau": tau,
                            "seed": "",
                            **{
                                output: float(
                                    np.mean([float(row[output]) for row in selected])
                                )
                                for output in metric_sources
                            },
                        }
                    )
            expected.extend(task_budget)
            task_means: list[dict[str, Any]] = []
            for environment in ENVIRONMENTS:
                selected = [row for row in cells if row["environment"] == environment]
                task_means.append(
                    {
                        "level": "task_marginal_mean",
                        "method_minus": method,
                        "method_reference": "td3",
                        "critic_scope": scope,
                        "environment": environment,
                        "tau": "",
                        "seed": "",
                        **{
                            output: float(
                                np.mean([float(row[output]) for row in selected])
                            )
                            for output in metric_sources
                        },
                    }
                )
            expected.extend(task_means)
            expected.append(
                {
                    "level": "task_equal_overall",
                    "method_minus": method,
                    "method_reference": "td3",
                    "critic_scope": scope,
                    "environment": "",
                    "tau": "",
                    "seed": "",
                    **{
                        output: float(
                            np.mean([float(row[output]) for row in task_means])
                        )
                        for output in metric_sources
                    },
                }
            )
    if len(contrast_rows) != 580 or len(expected) != 580:
        raise AssertionError("method contrasts must contain 580 fixed-grid rows")
    key_fields = (
        "level",
        "method_minus",
        "critic_scope",
        "environment",
        "tau",
        "seed",
    )

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[field]) for field in key_fields)

    actual_lookup = {key(row): row for row in contrast_rows}
    expected_lookup = {key(row): row for row in expected}
    if len(actual_lookup) != 580 or set(actual_lookup) != set(expected_lookup):
        raise AssertionError("method contrast key set is incomplete or duplicated")
    for row_key, expected_row in expected_lookup.items():
        actual = actual_lookup[row_key]
        if actual["method_reference"] != "td3":
            raise AssertionError(f"method contrast reference mismatch: {row_key}")
        if actual["seed_interpretation"] != seed_text:
            raise AssertionError(f"method contrast seed semantics mismatch: {row_key}")
        if actual["generalization_scope"] != scope_text:
            raise AssertionError(f"method contrast scope mismatch: {row_key}")
        for field in metric_sources:
            _close(float(actual[field]), float(expected_row[field]), f"{row_key} {field}")


def _verify_residual_aggregates(
    residual_rows: Sequence[Mapping[str, str]],
    aggregate_rows: Sequence[Mapping[str, str]],
) -> None:
    groups: dict[tuple[str, int, float, str, str], list[Mapping[str, str]]] = {}
    for row in residual_rows:
        key = (
            row["method"],
            int(row["hop"]),
            float(row["tau"]),
            row["environment"],
            row["stability"],
        )
        groups.setdefault(key, []).append(row)
    actual = {
        (
            row["method"],
            int(row["hop"]),
            float(row["tau"]),
            row["environment"],
            row["stability"],
        ): row
        for row in aggregate_rows
    }
    if len(actual) != len(aggregate_rows) or set(actual) != set(groups):
        raise AssertionError("residual aggregate key set is incomplete or duplicated")
    source_fields = {
        "r_before_mean": "r_before",
        "slack_before_mean": "slack_before",
        "r_after_mean": "r_after",
        "slack_after_mean": "slack_after",
    }
    for key, selected in groups.items():
        row = actual[key]
        if int(row["n_seed_cells"]) != len(selected):
            raise AssertionError(f"residual aggregate count mismatch: {key}")
        for output, source in source_fields.items():
            expected = float(np.mean([float(item[source]) for item in selected]))
            _close(float(row[output]), expected, f"{key} {output}")


def _verify_geometry_correlations(
    geometry_rows: Sequence[Mapping[str, str]],
    value_rows: Sequence[Mapping[str, str]],
    correlation_rows: Sequence[Mapping[str, str]],
) -> None:
    own = {
        row["key"]: float(row["delta_y_rms"])
        for row in value_rows
        if row["critic_scope"] == OWN_SCOPE
    }
    actual = {(row["method"], row["stability"]): row for row in correlation_rows}
    expected_keys = {
        (method, stability)
        for method in METHOD_HOPS
        for stability in ("all", "stable", "collapsed")
    }
    if len(correlation_rows) != 9 or set(actual) != expected_keys:
        raise AssertionError("geometry correlations must contain nine unique rows")
    for key, row in actual.items():
        method, stability = key
        selected = [
            item
            for item in geometry_rows
            if item["method"] == method
            and (stability == "all" or item["stability"] == stability)
        ]
        x = np.asarray(
            [float(item["final_target_ratio"]) for item in selected], dtype=np.float64
        )
        y = np.asarray([own[item["key"]] for item in selected], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        n_cells = int(np.sum(finite))
        expected_correlation = (
            float(np.corrcoef(x[finite], y[finite])[0, 1])
            if n_cells >= 2
            and float(np.std(x[finite])) > 0.0
            and float(np.std(y[finite])) > 0.0
            else float("nan")
        )
        if int(row["n_cells"]) != n_cells:
            raise AssertionError(f"geometry correlation count mismatch: {key}")
        _close(
            float(row["pearson_final_target_ratio_vs_within_run_delta_y_rms"]),
            expected_correlation,
            f"geometry correlation {key}",
        )
        if row["external_stability_definition"] != "normalized return <20 is collapsed":
            raise AssertionError(f"geometry correlation label mismatch: {key}")


def _verify_inventory_and_protocol(
    inventory: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    expected_counts = {
        "checkpoints": 270,
        "value_rows": 540,
        "geometry_rows": 270,
        "residual_rows": 450,
        "raw_npz": 270,
    }
    expected_formulas = {
        "direct_td_target_rms": (
            "gamma * sqrt(mean((min(Q1-,Q2-)(s_next,clip(mu1-(s_next)+epsilon))"
            " - min(Q1-,Q2-)(s_next,clip(a_next+epsilon)))**2))"
        ),
        "direct_td_target_rms_no_noise": (
            "gamma * sqrt(mean((min(Q1-,Q2-)(s_next,clip(mu1-(s_next)))"
            " - min(Q1-,Q2-)(s_next,clip(a_next)))**2))"
        ),
        "target_geometry": "sqrt(mean((mu1-(s_next)-a_next)**2))",
        "final_geometry": "sqrt(mean((muK(s_next)-a_next)**2))",
        "C_k": "mean(abs(Q1(s,mu_(k-1)(s)))) + 1e-6",
        "q_weight_k": "2*(T/K)/C_k",
        "residual": (
            "[-q_weight_k*mean(Q1(s,mu_k(s)))"
            "+mean((mu_k(s)-mu_(k-1)(s))**2)]"
            "-[-q_weight_k*mean(Q1(s,mu_(k-1)(s)))]"
        ),
        "feasible_comparator_slack": "max(0,residual)",
    }
    if protocol.get("expected_counts") != expected_counts:
        raise AssertionError("frozen protocol expected counts changed")
    if protocol.get("formulas") != expected_formulas:
        raise AssertionError("frozen protocol formulas changed")
    expected_config_contract = {
        "required_fixed": REQUIRED_CONFIG,
        "required_per_cell": ["env", "seed", "tau", "mpi_steps"],
        "optional_if_present": OPTIONAL_CONFIG_IF_PRESENT,
    }
    if protocol.get("checkpoint_config_contract") != expected_config_contract:
        raise AssertionError("frozen checkpoint config contract changed")
    grid = protocol.get("grid", {})
    if (
        tuple(grid.get("environments", ())) != ENVIRONMENTS
        or tuple(float(value) for value in grid.get("taus", ())) != TAUS
        or tuple(int(value) for value in grid.get("seeds", ())) != SEEDS
        or tuple(grid.get("methods", ())) != tuple(METHOD_HOPS)
        or int(grid.get("checkpoint_step", -1)) != 1_000_000
    ):
        raise AssertionError("frozen protocol grid changed")
    if protocol.get("value_scopes") != [OWN_SCOPE, COMMON_SCOPE]:
        raise AssertionError("frozen protocol critic scopes changed")
    residual_contract = protocol.get("residual", {})
    if (
        residual_contract.get("hops") != "k>=2 only"
        or residual_contract.get("critic_scope") != RESIDUAL_SCOPE
        or residual_contract.get("intervention")
        != "posthoc_final_checkpoint_one_adam_step"
        or _bool(residual_contract.get("historical_training_step_recovered", True))
        or _bool(residual_contract.get("fresh_optimizer_allowed", True))
    ):
        raise AssertionError("frozen residual intervention contract changed")
    provenance = inventory.get("provenance", {})
    provenance_hash = hashlib.sha256(_canonical_json_bytes(provenance)).hexdigest()
    if provenance_hash != inventory.get("provenance_bundle_sha256"):
        raise AssertionError("inventory provenance bundle hash mismatch")
    if provenance_hash != protocol.get("provenance_bundle_sha256"):
        raise AssertionError("protocol provenance bundle hash mismatch")
    if provenance.get("historical_training_identity_proven") is not False:
        raise AssertionError("inventory overclaims historical training identity")
    expected_sources = {
        "train_td3bc",
        "runner",
        "verifier",
        "_lab_import",
        "_ckpt_compat",
        "dump_target_policy_exposure",
        "d4rl_data",
    }
    sources = provenance.get("sources", {})
    if set(sources) != expected_sources:
        raise AssertionError("provenance source set is incomplete")
    for name, source in sources.items():
        digest = str(source.get("sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise AssertionError(f"invalid provenance digest: {name}")
        if not source.get("path"):
            raise AssertionError(f"missing provenance path: {name}")
    _verify_verifier_source(provenance)
    versions = provenance.get("environment_versions", {})
    if set(versions) != {"python", "jax", "flax", "optax", "numpy"} or any(
        not str(value) for value in versions.values()
    ):
        raise AssertionError("environment version provenance is incomplete")
    for git_name in ("diagnostic_git", "imported_training_git"):
        git_state = provenance.get(git_name, {})
        if (
            not git_state.get("root")
            or not git_state.get("revision")
            or not isinstance(git_state.get("dirty"), bool)
            or len(str(git_state.get("status_sha256", ""))) != 64
        ):
            raise AssertionError(f"Git provenance is incomplete: {git_name}")
    entries = inventory["entries"]
    records = {str(row["key"]): row for row in entries}
    if len(entries) != 270 or len(records) != 270 or set(records) != _expected_keys():
        raise AssertionError("inventory key set is incomplete or duplicated")
    expected_records = _expected_record_metadata()
    training_hash = sources["train_td3bc"]["sha256"]
    if inventory.get("training_source_sha256") != training_hash:
        raise AssertionError("inventory training source hash mismatch")
    if inventory.get("diagnostic_code_sha256") != sources["runner"]["sha256"]:
        raise AssertionError("inventory runner source hash mismatch")
    if protocol.get("training_source_sha256") != training_hash:
        raise AssertionError("protocol training source hash mismatch")
    if protocol.get("diagnostic_code_sha256") != sources["runner"]["sha256"]:
        raise AssertionError("protocol runner source hash mismatch")
    for key, row in records.items():
        if row.get("provenance_bundle_sha256") != provenance_hash:
            raise AssertionError(f"per-entry provenance mismatch: {key}")
        if row.get("training_source_sha256") != training_hash:
            raise AssertionError(f"per-entry training source mismatch: {key}")
        if not row.get("resolved") or not row.get("residual_supported"):
            raise AssertionError(f"unsupported inventory cell: {key}")
        for field in (
            "checkpoint_file_sha256",
            "weights_sha256",
            "online_critic_sha256",
            "target_critic_sha256",
            "target_actor_sha256",
            "config_file_sha256",
            "final_actor_sha256",
            "config_sha256",
            "dataset_sha256",
            "normalization_sha256",
            "selected_transition_sha256",
            "common_noise_sha256",
            "common_content_sha256",
            "eval_sha256",
        ):
            digest = str(row.get(field, ""))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise AssertionError(f"invalid inventory digest: {key} {field}")
        _verify_record_contract(key, row, expected_records[key])
        optimizer = row.get("optimizer_reconstruction", {})
        actors = optimizer.get("actors", ())
        if len(actors) != METHOD_HOPS[row["method"]]:
            raise AssertionError(f"optimizer actor count mismatch: {key}")
        for index, actor in enumerate(actors, start=1):
            if (
                int(actor.get("actor", -1)) != index
                or int(actor.get("actor_step", -1)) < 0
                or int(actor.get("actor_step", -1)) != int(actor.get("adam_count", -2))
                or len(str(actor.get("optimizer_state_sha256", ""))) != 64
            ):
                raise AssertionError(f"optimizer reconstruction mismatch: {key} actor {index}")
        if len(str(optimizer.get("all_optimizer_states_sha256", ""))) != 64:
            raise AssertionError(f"optimizer-state bundle hash missing: {key}")
    return records


def _verify_base_rows(
    rows: Sequence[Mapping[str, str]], records: Mapping[str, Mapping[str, Any]]
) -> None:
    exact_fields = (
        "method",
        "environment",
        "checkpoint_file_sha256",
        "weights_sha256",
        "config_sha256",
        "dataset_sha256",
        "normalization_sha256",
        "selected_transition_sha256",
        "common_noise_sha256",
    )
    for row in rows:
        record = records[row["key"]]
        for field in exact_fields:
            if str(row[field]) != str(record[field]):
                raise AssertionError(f"base-row fingerprint mismatch: {row['key']} {field}")
        if (
            int(row["seed"]) != int(record["seed"])
            or int(row["checkpoint_step"]) != int(record["checkpoint_step"])
        ):
            raise AssertionError(f"base-row integer metadata mismatch: {row['key']}")
        _close(float(row["tau"]), float(record["tau"]), f"{row['key']} tau")
        _close(
            float(row["external_score"]),
            float(record["external_score"]),
            f"{row['key']} external score",
        )


def verify_artifacts(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "MANIFEST.json"
    inventory_path = out_dir / "CHECKPOINTS.json"
    protocol_path = out_dir / "FROZEN_PROTOCOL.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if manifest["protocol"] != "p1_target_value_audit_v1":
        raise AssertionError("unexpected analysis protocol")
    _verify_manifest_contract(manifest)
    if sha256_file(inventory_path) != manifest["inventory_sha256"]:
        raise AssertionError("inventory hash differs from manifest")
    if sha256_file(protocol_path) != manifest["protocol_sha256"]:
        raise AssertionError("protocol hash differs from manifest")
    if protocol["checkpoint_manifest_sha256"] != sha256_file(inventory_path):
        raise AssertionError("protocol does not bind inventory")
    if inventory.get("n_resolved") != 270 or not inventory.get("inventory_complete"):
        raise AssertionError("inventory is not the complete 270 grid")
    records = _verify_inventory_and_protocol(inventory, protocol)
    expected_artifacts = {
        "td_target_value.csv",
        "same_next_state_geometry.csv",
        "comparator_residuals.csv",
        "method_contrasts.csv",
        "residual_aggregates.csv",
        "geometry_correlations.csv",
    }
    if set(manifest["artifact_sha256"]) != expected_artifacts:
        raise AssertionError("manifest scientific artifact set changed")
    for declared, expected_hash in manifest["artifact_sha256"].items():
        check_file_hash(_resolve_declared_path(out_dir, declared), expected_hash)
    if len(manifest["raw_sha256"]) != 270:
        raise AssertionError("manifest must bind 270 raw NPZ files")
    raw_cache: dict[str, dict[str, np.ndarray]] = {}
    for declared, expected_hash in manifest["raw_sha256"].items():
        path = _resolve_declared_path(out_dir, declared)
        check_file_hash(path, expected_hash)
        raw_cache[declared] = _load_npz(path)
    arrays_by_env = _verify_common_contracts(out_dir, inventory)
    value_rows = _read_csv(out_dir / "td_target_value.csv")
    geometry_rows = _read_csv(out_dir / "same_next_state_geometry.csv")
    residual_rows = _read_csv(out_dir / "comparator_residuals.csv")
    contrast_rows = _read_csv(out_dir / "method_contrasts.csv")
    residual_aggregate_rows = _read_csv(out_dir / "residual_aggregates.csv")
    correlation_rows = _read_csv(out_dir / "geometry_correlations.csv")
    for row in (*value_rows, *geometry_rows, *residual_rows):
        declared = str(row["raw_npz"])
        if declared not in raw_cache:
            raise AssertionError(f"CSV references undeclared raw artifact: {declared}")
        if row["raw_sha256"] != manifest["raw_sha256"][declared]:
            raise AssertionError(f"CSV raw hash mismatch: {declared}")
    _verify_base_rows((*value_rows, *geometry_rows, *residual_rows), records)
    _verify_value_and_geometry(
        inventory_records=records,
        value_rows=value_rows,
        geometry_rows=geometry_rows,
        arrays_by_env=arrays_by_env,
        raw_cache=raw_cache,
    )
    _verify_residuals(residual_rows, raw_cache, records)
    _verify_method_contrasts(value_rows, contrast_rows)
    _verify_residual_aggregates(residual_rows, residual_aggregate_rows)
    _verify_geometry_correlations(geometry_rows, value_rows, correlation_rows)
    expected_counts = {
        "checkpoints": 270,
        "value_rows": 540,
        "geometry_rows": 270,
        "residual_rows": 450,
        "raw_npz": 270,
    }
    if manifest["counts"] != expected_counts:
        raise AssertionError(f"manifest count mismatch: {manifest['counts']}")
    expected_derived_counts = {
        "method_contrast_rows": 580,
        "residual_aggregate_rows": len(residual_aggregate_rows),
        "geometry_correlation_rows": 9,
    }
    if manifest.get("derived_counts") != expected_derived_counts:
        raise AssertionError("manifest derived counts mismatch")
    raw_finite = all(
        np.all(np.isfinite(array))
        for raw in raw_cache.values()
        for array in raw.values()
    )
    derived_finite = (
        all(
            np.isfinite(float(row[field]))
            for row in value_rows
            for field in (
                "delta_y_rms",
                "delta_y_rms_no_noise",
                "smoothing_delta_y_rms_change",
                "smoothing_delta_y_rms_ratio",
            )
        )
        and all(
            np.isfinite(float(row["final_target_ratio"])) for row in geometry_rows
        )
        and all(
            np.isfinite(float(row[field]))
            for row in residual_rows
            for field in (
                "r_before",
                "r_after",
                "actor_loss_before",
                "actor_loss_after",
                "comparator_loss",
                "q_mean_before",
                "q_mean_after",
                "reference_q_mean",
                "transport_before",
                "transport_after",
            )
        )
    )
    finite = bool(raw_finite and derived_finite)
    row_flags = all(
        _bool(row["all_finite"]) for row in (*value_rows, *geometry_rows, *residual_rows)
    )
    if row_flags != finite:
        raise AssertionError("cell finite flags disagree with raw/derived values")
    if bool(manifest["finite_value_gate"]) != finite:
        raise AssertionError("finite gate disagrees with rows")
    return {
        "protocol": "p1_target_value_audit_v1",
        "pass": bool(finite),
        "artifact_integrity_pass": True,
        "artifact_only": True,
        "imports_analysis_runner": False,
        "verification_scope": (
            "artifact hashes and independently recomputed arithmetic; "
            "checkpoint networks are not reexecuted"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": sha256_file(inventory_path),
        "protocol_sha256": sha256_file(protocol_path),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "counts": expected_counts,
        "finite_value_gate": finite,
        "scientific_inclusion_gate_pass": bool(finite),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    verify_path = out_dir / "VERIFY.json"
    if verify_path.exists():
        raise FileExistsError(
            f"verification output is create-only; choose an unverified bundle: {verify_path}"
        )
    try:
        report = verify_artifacts(out_dir)
    except Exception as error:
        report = {
            "protocol": "p1_target_value_audit_v1",
            "pass": False,
            "artifact_integrity_pass": False,
            "artifact_only": True,
            "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
            "imports_analysis_runner": False,
            "verification_scope": (
                "artifact hashes and independently recomputed arithmetic; "
                "checkpoint networks are not reexecuted"
            ),
            "error": f"{type(error).__name__}: {error}",
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
    with verify_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
