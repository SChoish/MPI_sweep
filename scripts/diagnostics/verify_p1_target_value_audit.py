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
import re
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
METHOD_RESULT_DIRS = {"td3": "results_qnorm", "p3": "results_mpi3", "p4": "results/mpi4_norm"}
CHECKPOINT_STEP = 1_000_000
COLLAPSE_THRESHOLD = 20.0
PROTOCOL_VERSION = "p1_target_value_audit_v2"
MAX_ACTION = 1.0
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
ABSENT_CONFIG_FIELD = "<absent>"
CONFIG_SIGNATURE_FIELDS = (
    "mpi_steps",
    "q_scale_norm",
    "integrator",
    "mpi_two_step",
    "mpi_three_step",
    "mpi_four_step",
    "explicit_two_step",
    "explicit_three_step",
    "explicit_q_term",
    "fb",
)
ALLOWED_RAW_CONFIG_FIELDS = (
    "batch_size",
    "compilation_cache_dir",
    "data_dir",
    "discount",
    "env",
    "eval_episodes",
    "eval_freq",
    "explicit_q_term",
    "explicit_three_step",
    "explicit_two_step",
    "fb",
    "integrator",
    "lr",
    "max_timesteps",
    "method",
    "mpi_steps",
    "mpi_three_step",
    "mpi_two_step",
    "n_jitted_updates",
    "no_resume",
    "noise_clip",
    "normalize",
    "policy_freq",
    "policy_noise",
    "polyak",
    "q_scale_norm",
    "restore_path",
    "save_dir",
    "save_interval",
    "seed",
    "tau",
    "updates_per_dispatch",
)
LEGACY_CONFIG_PROFILES: dict[str, dict[str, Any]] = {
    "legacy_td3_qnorm_minimal_v0": {
        "method": "td3",
        "result_dir": "results_qnorm",
        "save_dir": "/home/ext_csv/td3_bc_jax/results_qnorm",
        "signature": {field: ABSENT_CONFIG_FIELD for field in CONFIG_SIGNATURE_FIELDS},
        "inferred_fields": {
            "mpi_steps": 1,
            "q_scale_norm": True,
            "integrator": "implicit",
        },
        "expected_inactive_optimizer_slots": 0,
    },
    "legacy_td3_qnorm_control_fields_v1": {
        "method": "td3",
        "result_dir": "results_qnorm",
        "save_dir": "/home/ext_csv/td3_bc_jax/results_qnorm",
        "signature": {
            "mpi_steps": ABSENT_CONFIG_FIELD,
            "q_scale_norm": ABSENT_CONFIG_FIELD,
            "integrator": ABSENT_CONFIG_FIELD,
            "mpi_two_step": False,
            "mpi_three_step": False,
            "mpi_four_step": ABSENT_CONFIG_FIELD,
            "explicit_two_step": False,
            "explicit_three_step": False,
            "explicit_q_term": True,
            "fb": False,
        },
        "inferred_fields": {
            "mpi_steps": 1,
            "q_scale_norm": True,
            "integrator": "implicit",
        },
        "expected_inactive_optimizer_slots": 2,
    },
    "legacy_p3_implicit_minimal_v0": {
        "method": "p3",
        "result_dir": "results_mpi3",
        "save_dir": "/home/ext_csv/td3_bc_jax/results_mpi3",
        "signature": {
            "mpi_steps": ABSENT_CONFIG_FIELD,
            "q_scale_norm": ABSENT_CONFIG_FIELD,
            "integrator": ABSENT_CONFIG_FIELD,
            "mpi_two_step": False,
            "mpi_three_step": True,
            "mpi_four_step": ABSENT_CONFIG_FIELD,
            "explicit_two_step": ABSENT_CONFIG_FIELD,
            "explicit_three_step": ABSENT_CONFIG_FIELD,
            "explicit_q_term": ABSENT_CONFIG_FIELD,
            "fb": ABSENT_CONFIG_FIELD,
        },
        "inferred_fields": {
            "mpi_steps": 3,
            "q_scale_norm": True,
            "integrator": "implicit",
        },
        "expected_inactive_optimizer_slots": 0,
    },
    "legacy_p3_implicit_explicit_flags_v1": {
        "method": "p3",
        "result_dir": "results_mpi3",
        "save_dir": "/home/ext_csv/td3_bc_jax/results_mpi3",
        "signature": {
            "mpi_steps": ABSENT_CONFIG_FIELD,
            "q_scale_norm": ABSENT_CONFIG_FIELD,
            "integrator": ABSENT_CONFIG_FIELD,
            "mpi_two_step": False,
            "mpi_three_step": True,
            "mpi_four_step": ABSENT_CONFIG_FIELD,
            "explicit_two_step": False,
            "explicit_three_step": False,
            "explicit_q_term": ABSENT_CONFIG_FIELD,
            "fb": ABSENT_CONFIG_FIELD,
        },
        "inferred_fields": {
            "mpi_steps": 3,
            "q_scale_norm": True,
            "integrator": "implicit",
        },
        "expected_inactive_optimizer_slots": 0,
    },
    "legacy_p3_implicit_qnorm_fields_v2": {
        "method": "p3",
        "result_dir": "results_mpi3",
        "save_dir": "/home/ext_csv/td3_bc_jax/results_mpi3",
        "signature": {
            "mpi_steps": ABSENT_CONFIG_FIELD,
            "q_scale_norm": True,
            "integrator": ABSENT_CONFIG_FIELD,
            "mpi_two_step": False,
            "mpi_three_step": True,
            "mpi_four_step": ABSENT_CONFIG_FIELD,
            "explicit_two_step": False,
            "explicit_three_step": False,
            "explicit_q_term": True,
            "fb": False,
        },
        "inferred_fields": {
            "mpi_steps": 3,
            "integrator": "implicit",
        },
        "expected_inactive_optimizer_slots": 0,
    },
}
CONFIG_COMPATIBILITY_CONTRACT = {
    "version": "p1_checkpoint_config_compat_v2",
    "fully_serialized_profile": "fully_serialized_config_v1",
    "allowed_raw_config_fields": list(ALLOWED_RAW_CONFIG_FIELDS),
    "unknown_raw_config_fields_rejected": True,
    "raw_config_preserved": True,
    "inference_only_when_field_absent": True,
    "historical_method_identity_proven": False,
    "historical_identity_limit": (
        "legacy path, save_dir, flags, and optimizer layout identify an eligible "
        "compatibility profile but do not prove the historical training revision"
    ),
    "legacy_profiles": LEGACY_CONFIG_PROFILES,
}
AUDIT_ADAM_HYPERPARAMETERS = {
    "b1": 0.9,
    "b2": 0.999,
    "eps": 1e-8,
    "eps_root": 0.0,
    "mu_dtype": None,
    "nesterov": False,
}
OPTIMIZER_COMPATIBILITY_CONTRACT = {
    "current_layout": "actors_tuple_with_independent_steps",
    "legacy_layout": "named_actor_states_with_adam_count_steps",
    "fresh_optimizer_allowed": False,
    "legacy_step_source": "unique_stored_adam_integer_count",
    "inactive_legacy_slot_required_count": 0,
    "schedule_derived_active_step": CHECKPOINT_STEP // REQUIRED_CONFIG["policy_freq"],
    "audit_adam_hyperparameters": AUDIT_ADAM_HYPERPARAMETERS,
    "historical_optimizer_hyperparameters_proven": False,
}
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
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _json_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise AssertionError(f"{label} is not an exact JSON integer: {value!r}")
    return value


def _csv_integer(value: Any, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None:
        raise AssertionError(f"{label} is not a canonical CSV integer: {value!r}")
    return int(value)


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (float, np.floating)) and float(actual) == expected
    return type(actual) is type(expected) and actual == expected


def _run_name(method: str, environment: str, tau: float, seed: int) -> str:
    tags = {"td3": "", "p3": "mpi3", "p4": "mpi4"}
    middle = f"_{tags[method]}" if tags[method] else ""
    return f"{environment}_tau{float(tau):g}{middle}_seed{int(seed)}"


def _expected_record_metadata(root: Path) -> dict[str, dict[str, Any]]:
    canonical_root = root.resolve()
    expected: dict[str, dict[str, Any]] = {}
    for environment in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS:
                for method, hops in METHOD_HOPS.items():
                    key = _cell_key(method, environment, tau, seed)
                    run_name = _run_name(method, environment, tau, seed)
                    result_dir = METHOD_RESULT_DIRS[method]
                    run_dir = canonical_root / result_dir / run_name
                    try:
                        run_dir.relative_to(canonical_root)
                    except ValueError as error:
                        raise AssertionError(
                            f"canonical result path escapes inventory root: {key}"
                        ) from error
                    expected[key] = {
                        "key": key,
                        "method": method,
                        "environment": environment,
                        "tau": float(tau),
                        "seed": int(seed),
                        "hops": int(hops),
                        "run_name": run_name,
                        "result_dir": result_dir,
                        "root": str(canonical_root),
                        "run_dir": str(run_dir),
                        "checkpoint_path": str(run_dir / f"params_{CHECKPOINT_STEP}.pkl"),
                        "config_path": str(run_dir / "config.json"),
                        "eval_path": str(run_dir / "eval.csv"),
                    }
    return expected


def _profile_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    return _same_value(actual, expected)


def _config_signature(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "present": field in config,
            "value": config[field] if field in config else None,
        }
        for field in CONFIG_SIGNATURE_FIELDS
    }


def _profile_matches(config: Mapping[str, Any], profile: Mapping[str, Any]) -> bool:
    for field, expected in profile["signature"].items():
        if expected == ABSENT_CONFIG_FIELD:
            if field in config:
                return False
        elif field not in config or not _profile_value_matches(config[field], expected):
            return False
    return True


def _mode_flag_errors(config: Mapping[str, Any], expected_hops: int) -> list[str]:
    errors: list[str] = []
    mpi_flags = {
        "mpi_two_step": 2,
        "mpi_three_step": 3,
        "mpi_four_step": 4,
    }
    explicit_flags = ("explicit_two_step", "explicit_three_step", "fb")
    for field in (*mpi_flags, *explicit_flags):
        if field in config and not isinstance(config[field], bool):
            errors.append(f"legacy mode flag {field} must be boolean")
    true_mpi = [
        (field, hops)
        for field, hops in mpi_flags.items()
        if config.get(field) is True
    ]
    if len(true_mpi) > 1:
        errors.append(f"multiple MPI mode flags are true: {[field for field, _ in true_mpi]}")
    elif true_mpi and true_mpi[0][1] != int(expected_hops):
        errors.append(
            f"legacy mode flag {true_mpi[0][0]} implies {true_mpi[0][1]} hops, "
            f"expected {int(expected_hops)}"
        )
    true_explicit = [field for field in explicit_flags if config.get(field) is True]
    if true_explicit:
        errors.append(f"explicit/FB mode is incompatible with implicit P1 grid: {true_explicit}")
    return errors


def _resolve_config_contract(
    config: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    raw_config = dict(config)
    errors = _mode_flag_errors(raw_config, int(expected["hops"]))
    unknown_fields = sorted(set(raw_config).difference(ALLOWED_RAW_CONFIG_FIELDS))
    if unknown_fields:
        errors.append(f"unknown raw config fields: {unknown_fields}")
    compatibility_fields = ("mpi_steps", "q_scale_norm", "integrator")
    missing_compatibility = [
        field for field in compatibility_fields if field not in raw_config
    ]
    inferred_fields: dict[str, Any] = {}
    profile_id = str(CONFIG_COMPATIBILITY_CONTRACT["fully_serialized_profile"])
    expected_inactive_slots = 0

    if missing_compatibility:
        matches = [
            (name, profile)
            for name, profile in LEGACY_CONFIG_PROFILES.items()
            if profile["method"] == expected["method"]
            and _profile_matches(raw_config, profile)
        ]
        if len(matches) != 1:
            errors.append(
                "raw config does not match exactly one closed legacy profile "
                f"for {expected['method']}: {[name for name, _ in matches]}"
            )
            profile_id = "unresolved"
        else:
            profile_id, profile = matches[0]
            inferred_fields = dict(profile["inferred_fields"])
            expected_inactive_slots = int(profile["expected_inactive_optimizer_slots"])
            if expected["result_dir"] != profile["result_dir"]:
                errors.append("record result_dir does not match the legacy profile")
            if Path(expected["run_dir"]) != (
                Path(expected["root"]) / profile["result_dir"] / expected["run_name"]
            ):
                errors.append("legacy run_dir is not the exact canonical root/method/run path")
            save_dir_text = raw_config.get("save_dir")
            if (
                not isinstance(save_dir_text, str)
                or not Path(save_dir_text).is_absolute()
                or save_dir_text != profile["save_dir"]
            ):
                errors.append("legacy config save_dir does not match the closed profile")

    effective_config = dict(raw_config)
    for field, value in inferred_fields.items():
        if field in effective_config:
            errors.append(f"legacy inference attempted to overwrite config key {field}")
        else:
            effective_config[field] = value

    locked = {
        "env": expected["environment"],
        "seed": expected["seed"],
        "tau": expected["tau"],
        "mpi_steps": expected["hops"],
        **REQUIRED_CONFIG,
    }
    for key, value in locked.items():
        if key not in effective_config:
            errors.append(f"missing effective config key {key}")
        elif not _same_value(effective_config[key], value):
            errors.append(f"effective config {key}={effective_config[key]!r} != {value!r}")
    for key, value in OPTIONAL_CONFIG_IF_PRESENT.items():
        if key in raw_config and not _same_value(raw_config[key], value):
            errors.append(f"config {key}={raw_config[key]!r} != {value!r}")

    run_suffix = str(Path(expected["result_dir"]) / expected["run_name"])
    raw_config_sha256 = hashlib.sha256(_canonical_json_bytes(raw_config)).hexdigest()
    resolution = {
        "version": CONFIG_COMPATIBILITY_CONTRACT["version"],
        "profile_id": profile_id,
        "raw_config_unchanged": True,
        "raw_config_sha256": raw_config_sha256,
        "inferred_fields": inferred_fields,
        "expected_inactive_optimizer_slots": expected_inactive_slots,
        "evidence": {
            "method": expected["method"],
            "result_dir": expected["result_dir"],
            "run_dir_suffix": run_suffix,
            "config_save_dir": raw_config.get("save_dir"),
            "signature": _config_signature(raw_config),
        },
    }
    return effective_config, resolution, errors


def _config_contract_errors(
    config: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    return _resolve_config_contract(config, expected)[2]


def _stored_action_scalar_matches(stored: Any, scale: Any, max_action: Any) -> bool:
    values = (stored, scale, max_action)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in values
    ):
        return False
    stored_value, scale_value, max_action_value = (float(value) for value in values)
    if not all(np.isfinite(value) for value in (stored_value, scale_value, max_action_value)):
        return False
    expected_float64 = scale_value * max_action_value
    expected_float32 = float(np.float32(scale_value) * np.float32(max_action_value))
    return stored_value == expected_float64 or stored_value == expected_float32


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
    identity_fields = (
        "key",
        "method",
        "environment",
        "seed",
        "hops",
        "run_name",
        "result_dir",
        "root",
    )
    for field in identity_fields:
        if not _same_value(row.get(field), expected[field]):
            raise AssertionError(f"fixed record metadata mismatch: {key} {field}")
    if not _same_value(row.get("tau"), expected["tau"]):
        raise AssertionError(f"fixed record metadata mismatch: {key} tau")
    for field in ("run_dir", "checkpoint_path", "config_path", "eval_path"):
        actual_path = row.get(field)
        if (
            not isinstance(actual_path, str)
            or not Path(actual_path).is_absolute()
            or actual_path != expected[field]
        ):
            raise AssertionError(f"canonical checkpoint path mismatch: {key} {field}")
    if _json_integer(row.get("checkpoint_step"), f"{key} checkpoint_step") != CHECKPOINT_STEP:
        raise AssertionError(f"checkpoint step mismatch: {key}")
    if _json_integer(
        row.get("external_score_step"), f"{key} external_score_step"
    ) != CHECKPOINT_STEP:
        raise AssertionError(f"external score step mismatch: {key}")

    config = row.get("config")
    if not isinstance(config, Mapping):
        raise AssertionError(f"checkpoint config is missing or invalid: {key}")
    effective_config, resolution, config_errors = _resolve_config_contract(
        config, expected
    )
    if config_errors:
        raise AssertionError(f"checkpoint config mismatch: {key}: {'; '.join(config_errors)}")
    config_hash = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    if config_hash != row.get("config_sha256"):
        raise AssertionError(f"checkpoint config hash mismatch: {key}")
    if row.get("config_schema") != resolution["profile_id"]:
        raise AssertionError(f"checkpoint config schema mismatch: {key}")
    recorded_effective = row.get("effective_config")
    if not isinstance(recorded_effective, Mapping) or dict(recorded_effective) != effective_config:
        raise AssertionError(f"effective checkpoint config mismatch: {key}")
    effective_hash = hashlib.sha256(
        _canonical_json_bytes(effective_config)
    ).hexdigest()
    if effective_hash != row.get("effective_config_sha256"):
        raise AssertionError(f"effective checkpoint config hash mismatch: {key}")
    recorded_resolution = row.get("compatibility_resolution")
    if (
        not isinstance(recorded_resolution, Mapping)
        or _canonical_json_bytes(recorded_resolution)
        != _canonical_json_bytes(resolution)
    ):
        raise AssertionError(f"checkpoint compatibility resolution mismatch: {key}")

    external_score = float(row.get("external_score", float("nan")))
    if not np.isfinite(external_score):
        raise AssertionError(f"external score is nonfinite: {key}")
    expected_collapsed = external_score < COLLAPSE_THRESHOLD
    if row.get("collapsed_lt20") is not expected_collapsed:
        raise AssertionError(f"collapse label mismatch: {key}")
    max_action_raw = row.get("max_action")
    policy_noise_raw = row.get("policy_noise")
    noise_clip_raw = row.get("noise_clip")
    if (
        isinstance(max_action_raw, bool)
        or not isinstance(max_action_raw, (int, float, np.integer, np.floating))
        or not np.isfinite(float(max_action_raw))
        or float(max_action_raw) != MAX_ACTION
    ):
        raise AssertionError(f"max_action is invalid: {key}")
    if not _stored_action_scalar_matches(
        policy_noise_raw, effective_config["policy_noise"], max_action_raw
    ):
        raise AssertionError(f"{key} policy_noise encoding mismatch")
    if not _stored_action_scalar_matches(
        noise_clip_raw, effective_config["noise_clip"], max_action_raw
    ):
        raise AssertionError(f"{key} noise_clip encoding mismatch")
    _close(
        float(row.get("discount")),
        float(effective_config["discount"]),
        f"{key} discount",
    )
    _close(
        float(row.get("learning_rate")),
        float(effective_config["lr"]),
        f"{key} learning rate",
        tolerance=1e-12,
    )
    if row.get("resolved") is not True or row.get("residual_supported") is not True:
        raise AssertionError(f"unsupported inventory cell: {key}")
    if row.get("residual_intervention") != "posthoc_final_checkpoint_one_adam_step":
        raise AssertionError(f"residual intervention mismatch: {key}")

    optimizer = row.get("optimizer_reconstruction", {})
    if optimizer.get("optimizer") != "optax.adam with frozen audit hyperparameters":
        raise AssertionError(f"optimizer identity mismatch: {key}")
    _close(
        float(optimizer.get("learning_rate")),
        float(effective_config["lr"]),
        f"{key} optimizer learning rate",
        tolerance=1e-12,
    )
    legacy_profile = (
        resolution["profile_id"]
        != CONFIG_COMPATIBILITY_CONTRACT["fully_serialized_profile"]
    )
    expected_layout = OPTIMIZER_COMPATIBILITY_CONTRACT[
        "legacy_layout" if legacy_profile else "current_layout"
    ]
    expected_step_source = (
        OPTIMIZER_COMPATIBILITY_CONTRACT["legacy_step_source"]
        if legacy_profile
        else "independently_stored_actor_step_and_adam_count"
    )
    if optimizer.get("storage_layout") != expected_layout:
        raise AssertionError(f"optimizer storage layout mismatch: {key}")
    if optimizer.get("actor_step_source") != expected_step_source:
        raise AssertionError(f"optimizer actor-step source mismatch: {key}")
    if optimizer.get("actor_step_independently_stored") is not (not legacy_profile):
        raise AssertionError(f"optimizer actor-step storage claim mismatch: {key}")

    expected_actor_step = CHECKPOINT_STEP // int(effective_config["policy_freq"])
    if _json_integer(
        optimizer.get("expected_actor_step"), f"{key} expected_actor_step"
    ) != expected_actor_step:
        raise AssertionError(f"expected optimizer step mismatch: {key}")
    actors = optimizer.get("actors", ())
    if len(actors) != int(expected["hops"]):
        raise AssertionError(f"optimizer actor count mismatch: {key}")
    for index, actor in enumerate(actors, start=1):
        if (
            _json_integer(actor.get("actor"), f"{key} actor index") != index
            or _json_integer(actor.get("actor_step"), f"{key} actor step")
            != expected_actor_step
            or _json_integer(actor.get("adam_count"), f"{key} Adam count")
            != expected_actor_step
            or not _is_sha256(actor.get("optimizer_state_sha256"))
        ):
            raise AssertionError(f"optimizer reconstruction mismatch: {key} actor {index}")

    inactive_slots = optimizer.get("inactive_legacy_slots", ())
    expected_inactive = int(resolution["expected_inactive_optimizer_slots"])
    if len(inactive_slots) != expected_inactive:
        raise AssertionError(f"inactive optimizer slot count mismatch: {key}")
    for offset, slot in enumerate(inactive_slots, start=1):
        expected_actor = int(expected["hops"]) + offset
        if (
            _json_integer(slot.get("actor"), f"{key} inactive actor index")
            != expected_actor
            or _json_integer(slot.get("adam_count"), f"{key} inactive Adam count")
            != int(OPTIMIZER_COMPATIBILITY_CONTRACT["inactive_legacy_slot_required_count"])
            or slot.get("fresh_optimizer_state_exact") is not True
            or not _is_sha256(slot.get("optimizer_state_sha256"))
        ):
            raise AssertionError(
                f"inactive optimizer reconstruction mismatch: {key} actor {expected_actor}"
            )
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
        value_rows = _json_integer(contract.get("value_rows"), f"{environment} value_rows")
        residual_rows = _json_integer(
            contract.get("residual_audit_rows"), f"{environment} residual_audit_rows"
        )
        if len(value_indices) != value_rows:
            raise AssertionError(f"value audit row count mismatch: {environment}")
        if len(residual_indices) != residual_rows:
            raise AssertionError(f"residual audit row count mismatch: {environment}")
        for prefix, expected_rows in (
            ("value", value_rows),
            ("residual", residual_rows),
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
        if (
            _csv_integer(geometry["n_rows"], f"{key} geometry n_rows") != n_rows
            or _csv_integer(geometry["action_dim"], f"{key} geometry action_dim")
            != action_dim
        ):
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
            if _csv_integer(row["n_rows"], f"{key} {scope} n_rows") != rewards.size:
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
    actual = {
        (row["key"], _csv_integer(row["hop"], f"{row['key']} residual hop"))
        for row in residual_rows
    }
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
        hop = _csv_integer(row["hop"], f"{row['key']} residual hop")
        total_hops = METHOD_HOPS[row["method"]]
        if _csv_integer(row["total_hops"], f"{row['key']} total_hops") != total_hops:
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
        expected_step = _json_integer(
            optimizer_actors[hop - 1].get("actor_step"),
            f"{row['key']} stored optimizer step",
        )
        step_before = _csv_integer(
            row["optimizer_step_before"], f"{row['key']} optimizer_step_before"
        )
        step_after = _csv_integer(
            row["optimizer_step_after"], f"{row['key']} optimizer_step_after"
        )
        if step_before != expected_step:
            raise AssertionError(f"stored optimizer step mismatch: {row['key']} k={hop}")
        if step_after != step_before + 1:
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
            _csv_integer(row["seed"], "method contrast value seed"),
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
            _csv_integer(row["hop"], f"{row['key']} aggregate source hop"),
            float(row["tau"]),
            row["environment"],
            row["stability"],
        )
        groups.setdefault(key, []).append(row)
    actual = {
        (
            row["method"],
            _csv_integer(row["hop"], "aggregate row hop"),
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
        if _csv_integer(row["n_seed_cells"], f"{key} n_seed_cells") != len(selected):
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
        if _csv_integer(row["n_cells"], f"{key} n_cells") != n_cells:
            raise AssertionError(f"geometry correlation count mismatch: {key}")
        _close(
            float(row["pearson_final_target_ratio_vs_within_run_delta_y_rms"]),
            expected_correlation,
            f"geometry correlation {key}",
        )
        if row["external_stability_definition"] != "normalized return <20 is collapsed":
            raise AssertionError(f"geometry correlation label mismatch: {key}")


def _verify_protocol_headers(
    inventory: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    if inventory.get("protocol") != PROTOCOL_VERSION:
        raise AssertionError("inventory protocol identity changed")
    if protocol.get("protocol") != PROTOCOL_VERSION:
        raise AssertionError("frozen protocol identity changed")
    if protocol.get("locked") is not True:
        raise AssertionError("frozen protocol is not locked")
    for field, expected in (
        ("atomic_inventory", True),
        ("analysis_started", False),
        ("no_substitution", True),
        ("inventory_complete", True),
    ):
        if inventory.get(field) is not expected:
            raise AssertionError(f"inventory header flag changed: {field}")
    for field in ("n_expected", "n_resolved"):
        if _json_integer(inventory.get(field), f"inventory {field}") != 270:
            raise AssertionError(f"inventory count changed: {field}")
    if (
        _json_integer(inventory.get("checkpoint_step"), "inventory checkpoint_step")
        != CHECKPOINT_STEP
    ):
        raise AssertionError("inventory checkpoint step changed")


def _verify_inventory_and_protocol(
    inventory: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    _verify_protocol_headers(inventory, protocol)
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
        "required_effective_fixed": REQUIRED_CONFIG,
        "required_effective_per_cell": ["env", "seed", "tau", "mpi_steps"],
        "optional_raw_if_present": OPTIONAL_CONFIG_IF_PRESENT,
        "max_action": MAX_ACTION,
        "compatibility": CONFIG_COMPATIBILITY_CONTRACT,
        "stored_action_scalar_encodings": [
            "exact_float64_scale_times_max_action",
            "exact_ieee754_float32_scale_times_max_action",
        ],
    }
    if protocol.get("checkpoint_config_contract") != expected_config_contract:
        raise AssertionError("frozen checkpoint config contract changed")
    if protocol.get("optimizer_state_contract") != OPTIMIZER_COMPATIBILITY_CONTRACT:
        raise AssertionError("frozen optimizer compatibility contract changed")
    grid = protocol.get("grid", {})
    if (
        tuple(grid.get("environments", ())) != ENVIRONMENTS
        or tuple(float(value) for value in grid.get("taus", ())) != TAUS
        or tuple(
            _json_integer(value, "frozen grid seed")
            for value in grid.get("seeds", ())
        )
        != SEEDS
        or tuple(grid.get("methods", ())) != tuple(METHOD_HOPS)
        or _json_integer(grid.get("checkpoint_step"), "frozen grid checkpoint_step")
        != 1_000_000
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
        or residual_contract.get("historical_training_step_recovered") is not False
        or residual_contract.get("fresh_optimizer_allowed") is not False
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
    inventory_root_text = inventory.get("root")
    if (
        not isinstance(inventory_root_text, str)
        or not Path(inventory_root_text).is_absolute()
        or str(Path(inventory_root_text).resolve()) != inventory_root_text
    ):
        raise AssertionError("inventory root is not one canonical absolute path")
    entries = inventory["entries"]
    records = {str(row["key"]): row for row in entries}
    if len(entries) != 270 or len(records) != 270 or set(records) != _expected_keys():
        raise AssertionError("inventory key set is incomplete or duplicated")
    checkpoint_paths = [str(row.get("checkpoint_path", "")) for row in entries]
    if len(set(checkpoint_paths)) != 270:
        raise AssertionError("resolved checkpoint paths are duplicated")
    expected_records = _expected_record_metadata(Path(inventory_root_text))
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
                _json_integer(actor.get("actor"), f"{key} actor index") != index
                or _json_integer(actor.get("actor_step"), f"{key} actor step") < 0
                or _json_integer(actor.get("actor_step"), f"{key} actor step")
                != _json_integer(actor.get("adam_count"), f"{key} Adam count")
                or not _is_sha256(actor.get("optimizer_state_sha256"))
            ):
                raise AssertionError(f"optimizer reconstruction mismatch: {key} actor {index}")
        if not _is_sha256(optimizer.get("all_optimizer_states_sha256")):
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
            _csv_integer(row["seed"], f"{row['key']} base seed")
            != _json_integer(record["seed"], f"{row['key']} record seed")
            or _csv_integer(
                row["checkpoint_step"], f"{row['key']} base checkpoint_step"
            )
            != _json_integer(
                record["checkpoint_step"], f"{row['key']} record checkpoint_step"
            )
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
    if manifest["protocol"] != PROTOCOL_VERSION:
        raise AssertionError("unexpected analysis protocol")
    _verify_manifest_contract(manifest)
    if sha256_file(inventory_path) != manifest["inventory_sha256"]:
        raise AssertionError("inventory hash differs from manifest")
    if sha256_file(protocol_path) != manifest["protocol_sha256"]:
        raise AssertionError("protocol hash differs from manifest")
    if protocol["checkpoint_manifest_sha256"] != sha256_file(inventory_path):
        raise AssertionError("protocol does not bind inventory")
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
        "protocol": PROTOCOL_VERSION,
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
            "protocol": PROTOCOL_VERSION,
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
