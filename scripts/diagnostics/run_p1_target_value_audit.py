#!/usr/bin/env python3
"""Strict CPU harness for the P1 target-value and comparator diagnostics.

This entrypoint is intentionally atomic.  ``run`` never analyzes a partial
grid: the exact 270 TD3/P3/P4 final checkpoints must first resolve at their
canonical paths and pass checkpoint/config/data/normalization fingerprints.
There is no glob discovery and no fallback checkpoint.

P1-C is a post-hoc one-step intervention from the *stored final* actor
parameters and Adam state on a frozen audit batch.  It cannot recover the
historical training minibatch or historical pre/post-update state.  If the
stored optimizer state cannot be reconstructed exactly with the imported
training source, preflight blocks the run instead of creating a fresh Adam
state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A CPU diagnostic must stay on CPU even inside an accelerator job shell.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4",
)

from _lab_import import REPO_ROOT, ensure_train_import_path  # noqa: E402

TRAIN_ROOT = ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import flax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

import d4rl_data as _data_module  # noqa: E402
import dump_target_policy_exposure as _exposure_module  # noqa: E402
import train_td3bc as _train_module  # noqa: E402
import _ckpt_compat as _compat_module  # noqa: E402
import _lab_import as _lab_module  # noqa: E402
from _ckpt_compat import normalize_checkpoint  # noqa: E402
from d4rl_data import DATASET_FILES, dataset_path  # noqa: E402
from dump_target_policy_exposure import (  # noqa: E402
    BootstrapTransitions,
    load_bootstrap_transitions,
)
from train_td3bc import (  # noqa: E402
    Actor,
    TwinCritic,
    load_checkpoint,
    qlearning_from_hdf5,
)


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
if set(ENVIRONMENTS) != set(DATASET_FILES):
    raise RuntimeError("imported dataset mapping differs from the locked nine-task grid")
TAUS = (4.0, 7.0, 10.0, 14.0, 20.0)
SEEDS = (0, 1)
CHECKPOINT_STEP = 1_000_000
COLLAPSE_THRESHOLD = 20.0
EXPECTED_VALUE_ROWS = 270 * 2
EXPECTED_GEOMETRY_ROWS = 270
EXPECTED_RESIDUAL_ROWS = 90 * 2 + 90 * 3
PROTOCOL_VERSION = "p1_target_value_audit_v2"
MAX_ACTION = 1.0


@dataclass(frozen=True)
class MethodSpec:
    result_dir: str
    tag: str
    hops: int
    score_key: str


METHODS: dict[str, MethodSpec] = {
    "td3": MethodSpec("results_qnorm", "", 1, "d4rl_score"),
    "p3": MethodSpec("results_mpi3", "mpi3", 3, "d4rl_pi3"),
    "p4": MethodSpec("results/mpi4_norm", "mpi4", 4, "d4rl_pi4"),
}

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
# Historical headline checkpoints predate the trainer's ``method`` field. If
# the field is present, it must identify BAR; P1's td3/p3/p4 labels remain the
# independent path-and-hop schema above.
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

VALUE_SCOPE_OWN = "within_run_target_critic"
VALUE_SCOPE_COMMON = "common_td3_target_critic"
RESIDUAL_SCOPE = "within_run_online_critic"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_cpu_backend() -> str:
    backend = str(jax.default_backend())
    if backend != "cpu":
        raise RuntimeError(
            f"P1 scientific analysis is CPU-only; resolved JAX backend is {backend!r}"
        )
    return backend


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(named_arrays: Mapping[str, Any]) -> str:
    """Stable content hash including array name, dtype, shape, and bytes."""

    digest = hashlib.sha256()
    for name in sorted(named_arrays):
        array = np.ascontiguousarray(np.asarray(named_arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_tree(tree: Any) -> str:
    """Stable hash for nested parameter trees, independent of pickle bytes."""

    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    arrays: dict[str, np.ndarray] = {}
    for index, (path, leaf) in enumerate(leaves_with_path):
        path_text = "/".join(str(component) for component in path)
        arrays[f"{index:06d}:{path_text}"] = np.asarray(leaf)
    return sha256_arrays(arrays)


def _tau_token(tau: float) -> str:
    return f"{float(tau):g}"


def run_name(method: str, environment: str, tau: float, seed: int) -> str:
    spec = METHODS[method]
    middle = f"_{spec.tag}" if spec.tag else ""
    return f"{environment}_tau{_tau_token(tau)}{middle}_seed{seed}"


def cell_key(method: str, environment: str, tau: float, seed: int) -> str:
    return f"{method}|{environment}|{_tau_token(tau)}|{seed}"


def expected_cells(root: Path, checkpoint_step: int = CHECKPOINT_STEP) -> list[dict[str, Any]]:
    canonical_root = root.resolve()
    cells: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for tau in TAUS:
            for seed in SEEDS:
                for method, spec in METHODS.items():
                    name = run_name(method, environment, tau, seed)
                    run_dir = canonical_root / spec.result_dir / name
                    cells.append(
                        {
                            "key": cell_key(method, environment, tau, seed),
                            "method": method,
                            "environment": environment,
                            "tau": float(tau),
                            "seed": int(seed),
                            "hops": int(spec.hops),
                            "run_name": name,
                            "result_dir": spec.result_dir,
                            "root": str(canonical_root),
                            "run_dir": str(run_dir),
                            "checkpoint_path": str(
                                run_dir / f"params_{int(checkpoint_step)}.pkl"
                            ),
                            "config_path": str(run_dir / "config.json"),
                            "eval_path": str(run_dir / "eval.csv"),
                        }
                    )
    if len(cells) != 270 or len({cell["key"] for cell in cells}) != 270:
        raise AssertionError("P1 expected grid must contain exactly 270 unique cells")
    return cells


def _write_json(
    path: Path, payload: Any, *, allow_identical: bool = False
) -> None:
    """Create one JSON artifact; never overwrite an existing different file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if allow_identical and path.is_file() and path.read_text(encoding="utf-8") == rendered:
            return
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def _relative_artifact(out_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(out_dir.resolve()).as_posix()


def _guard_output_lifecycle(out_dir: Path, mode: str) -> None:
    completed = out_dir / "MANIFEST.json"
    if completed.exists():
        raise FileExistsError(
            f"completed scientific output already exists; choose a new --out-dir: {completed}"
        )
    scientific = (
        "td_target_value.csv",
        "same_next_state_geometry.csv",
        "comparator_residuals.csv",
        "method_contrasts.csv",
        "residual_aggregates.csv",
        "geometry_correlations.csv",
        "RUN_STATUS.json",
        "raw",
    )
    blocked = (
        "CHECKPOINTS_UNRESOLVED.json",
        "PREFLIGHT_BLOCKED.json",
    )
    if mode == "dry-run":
        protected = (
            "CHECKPOINTS.json",
            *blocked,
            "PREFLIGHT_READY.json",
            "FROZEN_PROTOCOL.json",
            *scientific,
        )
        collisions = [name for name in protected if (out_dir / name).exists()]
        if collisions:
            raise FileExistsError(
                "dry-run cannot enter a preflight/scientific output directory: "
                + ", ".join(collisions)
            )
    if mode in {"preflight", "run"}:
        collisions = [
            name for name in (*blocked, *scientific) if (out_dir / name).exists()
        ]
        if collisions:
            raise FileExistsError(
                "blocked/scientific outputs are create-only; preflight/run cannot "
                "enter a blocked or partial scientific bundle; choose a new "
                "--out-dir or "
                "remove only an explicitly abandoned bundle: " + ", ".join(collisions)
            )


def _git_state(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "root": str(root.resolve()),
            "revision": revision,
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        }
    except Exception as error:
        return {
            "root": str(root.resolve()),
            "revision": None,
            "dirty": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _provenance_bundle() -> dict[str, Any]:
    source_paths = {
        "train_td3bc": Path(_train_module.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "verifier": Path(__file__).with_name("verify_p1_target_value_audit.py").resolve(),
        "_lab_import": Path(_lab_module.__file__).resolve(),
        "_ckpt_compat": Path(_compat_module.__file__).resolve(),
        "dump_target_policy_exposure": Path(_exposure_module.__file__).resolve(),
        "d4rl_data": Path(_data_module.__file__).resolve(),
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"provenance sources missing: {missing}")
    sources = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    diagnostic_git = _git_state(REPO_ROOT)
    training_git = _git_state(TRAIN_ROOT)
    for label, state in (
        ("diagnostic", diagnostic_git),
        ("imported training", training_git),
    ):
        if not state.get("revision") or not isinstance(state.get("dirty"), bool):
            raise RuntimeError(f"cannot record {label} Git revision/dirty state")
    return {
        "sources": sources,
        "environment_versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        },
        "diagnostic_git": diagnostic_git,
        "imported_training_git": training_git,
        "historical_training_identity_proven": False,
        "historical_training_identity_limit": (
            "checkpoint configs do not embed a Git revision; current imported "
            "source and Git state are fingerprinted but do not prove the "
            "historical training checkout was identical or clean"
        ),
    }


def write_expected_grid(out_dir: Path, root: Path, checkpoint_step: int) -> dict[str, Any]:
    document = {
        "protocol": PROTOCOL_VERSION,
        "mode": "dry_run",
        "analysis_started": False,
        "no_substitution": True,
        "n_expected": 270,
        "root": str(root.resolve()),
        "checkpoint_step": int(checkpoint_step),
        "environments": list(ENVIRONMENTS),
        "taus": list(TAUS),
        "seeds": list(SEEDS),
        "methods": {key: jsonable(value.__dict__) for key, value in METHODS.items()},
        "cells": expected_cells(root, checkpoint_step),
    }
    _write_json(out_dir / "EXPECTED_GRID.json", document, allow_identical=True)
    _write_json(
        out_dir / "DRY_RUN_STATUS.json",
        {
            "state": "dry_run_only",
            "analysis_started": False,
            "run_allowed": False,
            "n_expected": 270,
            "next": "run --mode preflight with the exact checkpoint and dataset roots",
        },
        allow_identical=True,
    )
    return document


def _actors(payload: Mapping[str, Any], expected_hops: int) -> tuple[Any, ...]:
    if "actors_params" in payload:
        actors = tuple(payload["actors_params"])
    else:
        normalized = normalize_checkpoint(payload)
        keys = [
            "actor_params",
            "actor2_params",
            "actor3_params",
            "actor4_params",
            "actor5_params",
        ]
        actors = tuple(normalized[key] for key in keys[:expected_hops] if key in normalized)
    if len(actors) != expected_hops:
        raise ValueError(f"actor count {len(actors)} != expected {expected_hops}")
    return actors


def _payload_config(
    payload: Mapping[str, Any], config_path: Path
) -> tuple[dict[str, Any], str, str]:
    if "config" not in payload:
        raise ValueError("checkpoint payload has no config")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("embedded checkpoint config is not a JSON object")
    if not config_path.is_file():
        raise ValueError("companion config.json is missing")
    embedded = jsonable(payload["config"])
    companion = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(companion, Mapping):
        raise ValueError("companion config.json is not a JSON object")
    if canonical_json_bytes(embedded) != canonical_json_bytes(companion):
        raise ValueError("embedded checkpoint config differs from companion config.json")
    return embedded, sha256_json(embedded), sha256_file(config_path)


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, (float, np.floating)) and float(actual) == expected
    return type(actual) is type(expected) and actual == expected


def _profile_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    return _same_value(actual, expected)



def _config_signature(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "present": field in config,
            "value": jsonable(config[field]) if field in config else None,
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
    config: Mapping[str, Any], cell: Mapping[str, Any], checkpoint_step: int
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    raw_config = dict(config)
    errors = _mode_flag_errors(raw_config, int(cell["hops"]))
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
            if profile["method"] == cell["method"]
            and _profile_matches(raw_config, profile)
        ]
        if len(matches) != 1:
            errors.append(
                "raw config does not match exactly one closed legacy profile "
                f"for {cell['method']}: {[name for name, _ in matches]}"
            )
            profile_id = "unresolved"
        else:
            profile_id, profile = matches[0]
            inferred_fields = dict(profile["inferred_fields"])
            expected_inactive_slots = int(profile["expected_inactive_optimizer_slots"])
            if cell.get("result_dir") != profile["result_dir"]:
                errors.append(
                    f"cell result_dir={cell.get('result_dir')!r} != "
                    f"legacy profile {profile['result_dir']!r}"
                )
            root_path = Path(str(cell.get("root", "")))
            run_dir = Path(str(cell.get("run_dir", "")))
            expected_run_dir = (
                root_path / str(profile["result_dir"]) / str(cell["run_name"])
            )
            if (
                not root_path.is_absolute()
                or not run_dir.is_absolute()
                or run_dir != expected_run_dir
            ):
                errors.append("legacy run_dir is not the exact canonical root/method/run path")
            try:
                expected_run_dir.relative_to(root_path)
            except ValueError:
                errors.append("legacy run_dir escapes the canonical root")
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

    expected = {
        "env": cell["environment"],
        "seed": cell["seed"],
        "tau": cell["tau"],
        "mpi_steps": cell["hops"],
        **REQUIRED_CONFIG,
        "max_timesteps": int(checkpoint_step),
    }
    for key, value in expected.items():
        if key not in effective_config:
            errors.append(f"missing effective config key {key}")
        elif not _same_value(effective_config[key], value):
            errors.append(f"effective config {key}={effective_config[key]!r} != {value!r}")
    for key, value in OPTIONAL_CONFIG_IF_PRESENT.items():
        if key in raw_config and not _same_value(raw_config[key], value):
            errors.append(f"config {key}={raw_config[key]!r} != {value!r}")

    result_dir = str(cell.get("result_dir", METHODS[str(cell["method"])].result_dir))
    run_suffix = str(Path(result_dir) / str(cell["run_name"]))
    resolution = {
        "version": CONFIG_COMPATIBILITY_CONTRACT["version"],
        "profile_id": profile_id,
        "raw_config_unchanged": True,
        "raw_config_sha256": sha256_json(raw_config),
        "inferred_fields": jsonable(inferred_fields),
        "expected_inactive_optimizer_slots": expected_inactive_slots,
        "evidence": {
            "method": str(cell["method"]),
            "result_dir": result_dir,
            "run_dir_suffix": run_suffix,
            "config_save_dir": jsonable(raw_config.get("save_dir")),
            "signature": _config_signature(raw_config),
        },
    }
    return effective_config, resolution, errors


def _validate_config(
    config: Mapping[str, Any], cell: Mapping[str, Any], checkpoint_step: int
) -> list[str]:
    return _resolve_config_contract(config, cell, checkpoint_step)[2]


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


def _parse_csv_integer(value: Any) -> int | None:
    if not isinstance(value, str) or re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None:
        return None
    return int(value)


def _read_external_score(cell: Mapping[str, Any]) -> tuple[float, int, str]:
    path = Path(cell["eval_path"])
    if not path.is_file():
        raise ValueError("eval.csv is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    at_step = [row for row in rows if _parse_csv_integer(row.get("step")) == CHECKPOINT_STEP]
    if len(at_step) != 1:
        raise ValueError(
            f"eval.csv must have exactly one row at {CHECKPOINT_STEP}; found {len(at_step)}"
        )
    row = at_step[0]
    score_key = METHODS[str(cell["method"])].score_key
    if row.get(score_key, "") == "":
        raise ValueError(f"eval.csv lacks final score field {score_key}")
    score = float(row[score_key])
    if not np.isfinite(score):
        raise ValueError("external final score is nonfinite")
    return score, CHECKPOINT_STEP, sha256_file(path)


LEGACY_ACTOR_PARAM_KEYS = (
    "actor_params",
    "actor2_params",
    "actor3_params",
    "actor4_params",
    "actor5_params",
)
LEGACY_ACTOR_OPT_STATE_KEYS = (
    "actor_opt_state",
    "actor2_opt_state",
    "actor3_opt_state",
    "actor4_opt_state",
    "actor5_opt_state",
)


@dataclass(frozen=True)
class StoredOptimizerBundle:
    active_opt_states: tuple[Any, ...]
    active_steps: tuple[Any, ...]
    serialized_opt_states: tuple[Any, ...]
    serialized_params: tuple[Any, ...]
    storage_layout: str
    actor_step_source: str
    actor_step_independently_stored: bool
    inactive_slots: tuple[dict[str, Any], ...]


def _adam_count_from_state(state: Any, label: str) -> int:
    count_candidates = [
        int(np.asarray(leaf))
        for leaf in jax.tree_util.tree_leaves(state)
        if np.asarray(leaf).shape == ()
        and np.issubdtype(np.asarray(leaf).dtype, np.integer)
    ]
    if len(count_candidates) != 1:
        raise ValueError(
            f"{label} Adam state has {len(count_candidates)} count candidates"
        )
    return count_candidates[0]


def _contiguous_named_values(
    payload: Mapping[str, Any], keys: Sequence[str], label: str
) -> tuple[Any, ...]:
    present = [index for index, key in enumerate(keys) if key in payload]
    if not present:
        return ()
    if present != list(range(present[-1] + 1)):
        raise ValueError(f"{label} slots are not a contiguous prefix: {present}")
    return tuple(payload[key] for key in keys[: present[-1] + 1])


def _stored_actor_optimizer_bundle(
    payload: Mapping[str, Any], expected_hops: int
) -> StoredOptimizerBundle:
    modern_present = "actors_opt_states" in payload or "actors_steps" in payload
    legacy_state_present = any(key in payload for key in LEGACY_ACTOR_OPT_STATE_KEYS)
    legacy_param_present = any(key in payload for key in LEGACY_ACTOR_PARAM_KEYS)
    if modern_present:
        if legacy_state_present or legacy_param_present:
            raise ValueError("checkpoint mixes current and legacy actor optimizer layouts")
        if "actors_opt_states" not in payload or "actors_steps" not in payload:
            raise ValueError("current optimizer layout requires states and independent steps")
        opt_states = tuple(payload["actors_opt_states"])
        actor_steps = tuple(payload["actors_steps"])
        params = tuple(payload.get("actors_params", ()))
        if (
            len(opt_states) != int(expected_hops)
            or len(actor_steps) != int(expected_hops)
            or len(params) != int(expected_hops)
        ):
            raise ValueError("current optimizer/step/parameter count differs from active hops")
        return StoredOptimizerBundle(
            active_opt_states=opt_states,
            active_steps=actor_steps,
            serialized_opt_states=opt_states,
            serialized_params=params,
            storage_layout=str(OPTIMIZER_COMPATIBILITY_CONTRACT["current_layout"]),
            actor_step_source="independently_stored_actor_step_and_adam_count",
            actor_step_independently_stored=True,
            inactive_slots=(),
        )

    if "actors_params" in payload:
        raise ValueError("checkpoint mixes tuple actor params with legacy optimizer layout")
    opt_states = _contiguous_named_values(
        payload, LEGACY_ACTOR_OPT_STATE_KEYS, "legacy optimizer"
    )
    params = _contiguous_named_values(payload, LEGACY_ACTOR_PARAM_KEYS, "legacy actor")
    if len(opt_states) != len(params):
        raise ValueError("legacy optimizer and parameter slot counts differ")
    if len(opt_states) < int(expected_hops):
        raise ValueError("legacy optimizer layout has fewer states than active hops")
    active_opt_states = opt_states[: int(expected_hops)]
    active_steps = tuple(
        _adam_count_from_state(state, f"actor {index}")
        for index, state in enumerate(active_opt_states, start=1)
    )
    inactive: list[dict[str, Any]] = []
    for index, state in enumerate(opt_states[int(expected_hops) :], start=expected_hops + 1):
        adam_count = _adam_count_from_state(state, f"inactive actor {index}")
        if adam_count != int(
            OPTIMIZER_COMPATIBILITY_CONTRACT["inactive_legacy_slot_required_count"]
        ):
            raise ValueError(
                f"inactive legacy actor {index} Adam count is {adam_count}, expected 0"
            )
        inactive.append(
            {
                "actor": int(index),
                "adam_count": adam_count,
                "optimizer_state_sha256": sha256_tree(state),
            }
        )
    return StoredOptimizerBundle(
        active_opt_states=active_opt_states,
        active_steps=active_steps,
        serialized_opt_states=opt_states,
        serialized_params=params,
        storage_layout=str(OPTIMIZER_COMPATIBILITY_CONTRACT["legacy_layout"]),
        actor_step_source=str(OPTIMIZER_COMPATIBILITY_CONTRACT["legacy_step_source"]),
        actor_step_independently_stored=False,
        inactive_slots=tuple(inactive),
    )


def _audit_adam(learning_rate: float) -> optax.GradientTransformationExtraArgs:
    return optax.adam(
        learning_rate=float(learning_rate),
        **AUDIT_ADAM_HYPERPARAMETERS,
    )


def _optimizer_state_exact(
    payload: Mapping[str, Any], actors: Sequence[Any], learning_rate: float
) -> tuple[bool, str | None, dict[str, Any]]:
    try:
        bundle = _stored_actor_optimizer_bundle(payload, len(actors))
    except ValueError as error:
        return False, str(error), {}
    optimizer = _audit_adam(learning_rate)
    for index, (params, stored) in enumerate(
        zip(bundle.serialized_params, bundle.serialized_opt_states, strict=True),
        start=1,
    ):
        fresh = optimizer.init(jax.tree_util.tree_map(jnp.asarray, params))
        if jax.tree_util.tree_structure(fresh) != jax.tree_util.tree_structure(stored):
            return False, f"actor {index} optimizer tree structure mismatch", {}
        fresh_leaves = jax.tree_util.tree_leaves(fresh)
        stored_leaves = jax.tree_util.tree_leaves(stored)
        for fresh_leaf, stored_leaf in zip(fresh_leaves, stored_leaves, strict=True):
            fresh_array = np.asarray(fresh_leaf)
            stored_array = np.asarray(stored_leaf)
            if fresh_array.shape != stored_array.shape or fresh_array.dtype != stored_array.dtype:
                return False, f"actor {index} optimizer leaf shape/dtype mismatch", {}
            if index > len(actors) and not np.array_equal(stored_array, fresh_array):
                return False, f"inactive actor {index} optimizer state is not fresh-zero", {}

    details: list[dict[str, Any]] = []
    for index, (stored, actor_step) in enumerate(
        zip(bundle.active_opt_states, bundle.active_steps, strict=True),
        start=1,
    ):
        try:
            adam_count = _adam_count_from_state(stored, f"actor {index}")
        except ValueError as error:
            return False, str(error), {}
        step_array = np.asarray(actor_step)
        if step_array.shape != () or not np.issubdtype(step_array.dtype, np.integer):
            return False, f"actor {index} step is not an integer scalar", {}
        step = int(step_array)
        if step < 0 or adam_count != step:
            return False, f"actor {index} step/count mismatch: {step} != {adam_count}", {}
        details.append(
            {
                "actor": index,
                "actor_step": step,
                "adam_count": adam_count,
                "optimizer_state_sha256": sha256_tree(stored),
            }
        )
    inactive_details = [
        {**slot, "fresh_optimizer_state_exact": True}
        for slot in bundle.inactive_slots
    ]
    return (
        True,
        None,
        {
            "optimizer": "optax.adam with frozen audit hyperparameters",
            "learning_rate": float(learning_rate),
            "storage_layout": bundle.storage_layout,
            "actor_step_source": bundle.actor_step_source,
            "actor_step_independently_stored": bundle.actor_step_independently_stored,
            "actors": details,
            "inactive_legacy_slots": inactive_details,
            "all_optimizer_states_sha256": sha256_tree(bundle.serialized_opt_states),
        },
    )


def _weights_payload(payload: Mapping[str, Any], actors: Sequence[Any]) -> dict[str, Any]:
    return {
        "actors": tuple(actors),
        "critic": payload["critic_params"],
        "target_actor": payload["target_actor_params"],
        "target_critic": payload["target_critic_params"],
    }


def _stable_seed(label: str, base_seed: int) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**63)


def _normalization_hash(mean: Any, std: Any) -> str:
    return sha256_arrays(
        {
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
        }
    )


def _present_file_gate(
    cells: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for cell in cells:
        root = Path(str(cell["root"]))
        for field in fields:
            path = Path(str(cell[field]))
            if not path.is_file():
                missing.append(
                    {
                        "key": str(cell["key"]),
                        "field": field,
                        "path": str(path),
                    }
                )
                continue
            try:
                path.relative_to(root)
                resolved = path.resolve(strict=True)
            except (OSError, ValueError):
                resolved = None
            if resolved != path:
                missing.append(
                    {
                        "key": str(cell["key"]),
                        "field": field,
                        "path": str(path),
                        "reason": "path is not a canonical nonsymlink descendant",
                    }
                )
    return missing


def _unresolved_document(
    *,
    root: Path,
    cells: Sequence[Mapping[str, Any]],
    checkpoint_step: int,
    failures: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    failure_keys = {str(item.get("key", "")) for item in failures}
    entries = []
    for cell in cells:
        entries.append(
            {
                **dict(cell),
                "resolved": False,
                "validation_stage": stage,
                "failure_recorded": str(cell["key"]) in failure_keys,
            }
        )
    return {
        "protocol": PROTOCOL_VERSION,
        "atomic_inventory": True,
        "analysis_started": False,
        "no_substitution": True,
        "root": str(root.resolve()),
        "checkpoint_step": int(checkpoint_step),
        "n_expected": 270,
        "n_resolved": 0,
        "inventory_complete": False,
        "validation_stage": stage,
        "failures": list(failures),
        "entries": entries,
    }


def _write_blocked_preflight(
    out_dir: Path,
    document: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    _write_json(
        out_dir / "CHECKPOINTS_UNRESOLVED.json",
        document,
        allow_identical=True,
    )
    _write_json(
        out_dir / "PREFLIGHT_BLOCKED.json",
        {
            "state": "preflight_blocked",
            "analysis_started": False,
            "run_allowed": False,
            "reason": reason,
            "n_expected": 270,
            "n_resolved": 0,
            "unresolved_manifest": str(
                Path("CHECKPOINTS_UNRESOLVED.json")
            ),
        },
        allow_identical=True,
    )


def _inspect_checkpoint_cell(
    cell: Mapping[str, Any], checkpoint_step: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(str(cell["checkpoint_path"]))
    config_path = Path(str(cell["config_path"]))
    payload = load_checkpoint(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    required = (
        "mean",
        "std",
        "max_action",
        "policy_noise",
        "noise_clip",
        "critic_params",
        "target_actor_params",
        "target_critic_params",
    )
    missing_keys = [key for key in required if key not in payload]
    if missing_keys:
        raise ValueError(f"checkpoint missing keys: {missing_keys}")
    payload_step = payload.get("step")
    if (
        isinstance(payload_step, bool)
        or not isinstance(payload_step, (int, np.integer))
        or int(payload_step) != int(checkpoint_step)
    ):
        raise ValueError(
            f"payload step={payload_step!r} is not exact integer {int(checkpoint_step)}"
        )
    config, config_sha256, config_file_sha256 = _payload_config(
        payload, config_path
    )
    effective_config, compatibility_resolution, config_errors = (
        _resolve_config_contract(config, cell, checkpoint_step)
    )
    if config_errors:
        raise ValueError("; ".join(config_errors))
    actors = _actors(payload, int(cell["hops"]))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("checkpoint normalization arrays have invalid shapes")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("checkpoint normalization arrays are nonfinite/nonpositive")
    max_action_raw = payload["max_action"]
    policy_noise_raw = payload["policy_noise"]
    noise_clip_raw = payload["noise_clip"]
    if (
        isinstance(max_action_raw, bool)
        or not isinstance(max_action_raw, (int, float, np.integer, np.floating))
        or not np.isfinite(float(max_action_raw))
        or float(max_action_raw) != MAX_ACTION
    ):
        raise ValueError(f"max_action must be exact fixed value {MAX_ACTION}")
    if not _stored_action_scalar_matches(
        policy_noise_raw, effective_config["policy_noise"], max_action_raw
    ):
        raise ValueError(
            "stored policy_noise is not an allowed exact config scale * max_action encoding"
        )
    if not _stored_action_scalar_matches(
        noise_clip_raw, effective_config["noise_clip"], max_action_raw
    ):
        raise ValueError(
            "stored noise_clip is not an allowed exact config scale * max_action encoding"
        )
    max_action = float(max_action_raw)
    policy_noise = float(policy_noise_raw)
    noise_clip = float(noise_clip_raw)
    optimizer_ok, optimizer_error, optimizer_details = _optimizer_state_exact(
        payload, actors, float(effective_config["lr"])
    )
    if not optimizer_ok:
        raise ValueError(
            "P1-C residual unsupported: exact Adam reconstruction failed: "
            f"{optimizer_error}"
        )
    legacy_profile = (
        compatibility_resolution["profile_id"]
        != CONFIG_COMPATIBILITY_CONTRACT["fully_serialized_profile"]
    )
    expected_layout = OPTIMIZER_COMPATIBILITY_CONTRACT[
        "legacy_layout" if legacy_profile else "current_layout"
    ]
    if optimizer_details["storage_layout"] != expected_layout:
        raise ValueError(
            "config profile and stored optimizer layout disagree: "
            f"{compatibility_resolution['profile_id']} vs "
            f"{optimizer_details['storage_layout']}"
        )
    inactive_slots = optimizer_details["inactive_legacy_slots"]
    expected_inactive_slots = int(
        compatibility_resolution["expected_inactive_optimizer_slots"]
    )
    if len(inactive_slots) != expected_inactive_slots:
        raise ValueError(
            "config profile and inactive optimizer slots disagree: "
            f"{len(inactive_slots)} != {expected_inactive_slots}"
        )
    policy_freq = int(effective_config["policy_freq"])
    if int(checkpoint_step) % policy_freq:
        raise ValueError("checkpoint step is not divisible by policy_freq")
    expected_actor_step = int(checkpoint_step) // policy_freq
    actor_steps = {
        int(actor["actor_step"]) for actor in optimizer_details["actors"]
    }
    if actor_steps != {expected_actor_step}:
        raise ValueError(
            "P1-C residual unsupported: stored actor/Adam steps do not match "
            f"the fixed update schedule: {sorted(actor_steps)} != {expected_actor_step}"
        )
    optimizer_details["expected_actor_step"] = expected_actor_step
    score, score_step, eval_sha256 = _read_external_score(cell)
    record = {
        **dict(cell),
        "resolved": True,
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "weights_sha256": sha256_tree(_weights_payload(payload, actors)),
        "online_critic_sha256": sha256_tree(payload["critic_params"]),
        "target_critic_sha256": sha256_tree(payload["target_critic_params"]),
        "target_actor_sha256": sha256_tree(payload["target_actor_params"]),
        "final_actor_sha256": sha256_tree(actors[-1]),
        "config": config,
        "config_sha256": config_sha256,
        "config_file_sha256": config_file_sha256,
        "config_schema": compatibility_resolution["profile_id"],
        "effective_config": effective_config,
        "effective_config_sha256": sha256_json(effective_config),
        "compatibility_resolution": compatibility_resolution,
        "normalization_sha256": _normalization_hash(mean, std),
        "max_action": max_action,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "discount": float(effective_config["discount"]),
        "learning_rate": float(effective_config["lr"]),
        "residual_supported": True,
        "optimizer_reconstruction": optimizer_details,
        "residual_intervention": "posthoc_final_checkpoint_one_adam_step",
        "external_score": score,
        "external_score_step": score_step,
        "collapsed_lt20": bool(score < COLLAPSE_THRESHOLD),
        "eval_sha256": eval_sha256,
    }
    runtime = {
        "mean": mean,
        "std": std,
        "max_action": max_action,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
    }
    return record, runtime


def _take(transitions: BootstrapTransitions, indices: np.ndarray) -> BootstrapTransitions:
    return transitions.take(np.asarray(indices, dtype=np.int64))


def _transition_arrays(prefix: str, transitions: BootstrapTransitions) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_observations": np.asarray(transitions.observations, dtype=np.float32),
        f"{prefix}_actions": np.asarray(transitions.actions, dtype=np.float32),
        f"{prefix}_next_observations": np.asarray(
            transitions.next_observations, dtype=np.float32
        ),
        f"{prefix}_next_actions": np.asarray(transitions.next_actions, dtype=np.float32),
        f"{prefix}_rewards": np.asarray(transitions.rewards, dtype=np.float32),
        f"{prefix}_source_indices": np.asarray(
            transitions.source_indices, dtype=np.int64
        ),
    }


def _build_common_contracts(
    *,
    data_dir: Path,
    out_dir: Path,
    records: list[dict[str, Any]],
    runtimes: Mapping[str, Mapping[str, Any]],
    n_states: int,
    sample_seed: int,
    noise_seed: int,
    max_episode_steps: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    contracts: dict[str, Any] = {}
    arrays_by_env: dict[str, dict[str, np.ndarray]] = {}
    records_by_env = {
        environment: [row for row in records if row["environment"] == environment]
        for environment in ENVIRONMENTS
    }
    for environment in ENVIRONMENTS:
        env_rows = records_by_env[environment]
        normalization_hashes = {row["normalization_sha256"] for row in env_rows}
        noise_contracts = {
            (row["max_action"], row["policy_noise"], row["noise_clip"], row["discount"])
            for row in env_rows
        }
        if len(normalization_hashes) != 1:
            failures.append(
                {"key": environment, "error": "normalization differs across 30 cells"}
            )
            continue
        if len(noise_contracts) != 1:
            failures.append(
                {"key": environment, "error": "noise/discount contract differs across 30 cells"}
            )
            continue
        path = dataset_path(environment, data_dir)
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(
                {"key": environment, "error": "dataset missing", "path": str(path)}
            )
            continue
        try:
            training_data = qlearning_from_hdf5(path, max_episode_steps=max_episode_steps)
            dataset_mean = np.asarray(
                training_data["observations"].mean(axis=0), dtype=np.float32
            )
            dataset_std = np.asarray(
                training_data["observations"].std(axis=0) + 1e-3, dtype=np.float32
            )
            checkpoint_runtime = runtimes[str(env_rows[0]["key"])]
            if not np.allclose(
                checkpoint_runtime["mean"], dataset_mean, rtol=1e-6, atol=1e-6
            ) or not np.allclose(
                checkpoint_runtime["std"], dataset_std, rtol=1e-6, atol=1e-6
            ):
                raise ValueError("checkpoint normalization differs from dataset loader")
            transitions = load_bootstrap_transitions(
                path, max_episode_steps=max_episode_steps
            )
            available = int(transitions.observations.shape[0])
            if available < 2 * n_states:
                raise ValueError(
                    f"need {2 * n_states} bootstrap rows, found {available}"
                )
            rng = np.random.default_rng(_stable_seed(environment, sample_seed))
            selected = rng.choice(available, size=2 * n_states, replace=False).astype(
                np.int64
            )
            value = _take(transitions, selected[:n_states])
            residual = _take(transitions, selected[n_states:])
            action_dim = int(value.actions.shape[-1])
            noise_rng = np.random.default_rng(_stable_seed(environment, noise_seed))
            standard_noise = noise_rng.standard_normal(
                (n_states, action_dim), dtype=np.float32
            )
            max_action, policy_noise, noise_clip, discount = next(iter(noise_contracts))
            epsilon = np.clip(
                standard_noise * float(policy_noise),
                -float(noise_clip),
                float(noise_clip),
            ).astype(np.float32)
            arrays = {
                **_transition_arrays("value", value),
                **_transition_arrays("residual", residual),
                "standard_noise": standard_noise,
                "epsilon": epsilon,
            }
            selected_hash = sha256_arrays(
                {
                    key: value
                    for key, value in arrays.items()
                    if key.startswith("value_") or key.startswith("residual_")
                }
            )
            noise_hash = sha256_arrays(
                {"standard_noise": standard_noise, "epsilon": epsilon}
            )
            content_hash = sha256_arrays(arrays)
            common_path = out_dir / "common_batches" / f"{environment}.npz"
            contracts[environment] = {
                "environment": environment,
                "dataset_path": str(path.resolve()),
                "dataset_sha256": sha256_file(path),
                "dataset_normalization_sha256": _normalization_hash(
                    dataset_mean, dataset_std
                ),
                "checkpoint_normalization_sha256": next(iter(normalization_hashes)),
                "selected_transition_sha256": selected_hash,
                "common_noise_sha256": noise_hash,
                "common_content_sha256": content_hash,
                "common_npz": _relative_artifact(out_dir, common_path),
                "value_rows": int(n_states),
                "residual_audit_rows": int(n_states),
                "action_dim": action_dim,
                "sample_seed": int(_stable_seed(environment, sample_seed)),
                "noise_seed": int(_stable_seed(environment, noise_seed)),
                "max_action": float(max_action),
                "policy_noise": float(policy_noise),
                "noise_clip": float(noise_clip),
                "discount": float(discount),
                "transition_stats": transitions.stats,
            }
            arrays_by_env[environment] = arrays
        except Exception as error:
            failures.append({"key": environment, "error": str(error)})
        finally:
            if "training_data" in locals():
                del training_data
    if failures:
        return contracts, arrays_by_env, failures
    common_dir = out_dir / "common_batches"
    common_dir.mkdir(parents=True, exist_ok=True)
    for environment, arrays in arrays_by_env.items():
        path = common_dir / f"{environment}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as existing:
                existing_arrays = {
                    name: np.asarray(existing[name]) for name in existing.files
                }
            if sha256_arrays(existing_arrays) != sha256_arrays(arrays):
                raise FileExistsError(
                    f"frozen common batch exists with different content: {path}"
                )
        else:
            with path.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
    for row in records:
        contract = contracts[str(row["environment"])]
        row.update(
            {
                "dataset_sha256": contract["dataset_sha256"],
                "dataset_normalization_sha256": contract[
                    "dataset_normalization_sha256"
                ],
                "selected_transition_sha256": contract[
                    "selected_transition_sha256"
                ],
                "common_noise_sha256": contract["common_noise_sha256"],
                "common_content_sha256": contract["common_content_sha256"],
            }
        )
    return contracts, arrays_by_env, []


def preflight(
    *,
    root: Path,
    data_dir: Path,
    out_dir: Path,
    checkpoint_step: int,
    n_states: int,
    sample_seed: int,
    noise_seed: int,
    max_episode_steps: int,
) -> tuple[bool, dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    cells = expected_cells(root, checkpoint_step)
    missing = _present_file_gate(cells, ("checkpoint_path",))
    if missing:
        document = _unresolved_document(
            root=root,
            cells=cells,
            checkpoint_step=checkpoint_step,
            failures=missing,
            stage="exact_270_checkpoint_path_presence",
        )
        _write_blocked_preflight(
            out_dir, document, reason=f"{len(missing)} exact final checkpoints are missing"
        )
        return False, document, {}
    companion_missing = _present_file_gate(cells, ("config_path", "eval_path"))
    if companion_missing:
        document = _unresolved_document(
            root=root,
            cells=cells,
            checkpoint_step=checkpoint_step,
            failures=companion_missing,
            stage="companion_config_and_external_eval_presence",
        )
        _write_blocked_preflight(
            out_dir,
            document,
            reason=f"{len(companion_missing)} companion provenance/eval files are missing",
        )
        return False, document, {}

    records: list[dict[str, Any]] = []
    runtimes: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for cell in cells:
        try:
            record, runtime = _inspect_checkpoint_cell(cell, checkpoint_step)
            records.append(record)
            runtimes[str(cell["key"])] = runtime
        except Exception as error:
            failures.append({"key": cell["key"], "error": str(error)})
    if failures:
        document = _unresolved_document(
            root=root,
            cells=cells,
            checkpoint_step=checkpoint_step,
            failures=failures,
            stage="checkpoint_identity_and_optimizer",
        )
        _write_blocked_preflight(
            out_dir,
            document,
            reason=f"{len(failures)} checkpoint/config/optimizer mismatches",
        )
        return False, document, {}

    try:
        provenance = _provenance_bundle()
    except Exception as error:
        document = _unresolved_document(
            root=root,
            cells=cells,
            checkpoint_step=checkpoint_step,
            failures=[{"key": "source_provenance", "error": str(error)}],
            stage="source_and_environment_provenance",
        )
        _write_blocked_preflight(
            out_dir, document, reason="source/environment provenance is incomplete"
        )
        return False, document, {}

    contracts, arrays_by_env, data_failures = _build_common_contracts(
        data_dir=data_dir,
        out_dir=out_dir,
        records=records,
        runtimes=runtimes,
        n_states=n_states,
        sample_seed=sample_seed,
        noise_seed=noise_seed,
        max_episode_steps=max_episode_steps,
    )
    if data_failures:
        document = _unresolved_document(
            root=root,
            cells=cells,
            checkpoint_step=checkpoint_step,
            failures=data_failures,
            stage="dataset_normalization_and_common_contract",
        )
        _write_blocked_preflight(
            out_dir,
            document,
            reason=f"{len(data_failures)} dataset/common-contract mismatches",
        )
        return False, document, {}

    source_path = Path(provenance["sources"]["train_td3bc"]["path"])
    code_path = Path(provenance["sources"]["runner"]["path"])
    source_sha256 = provenance["sources"]["train_td3bc"]["sha256"]
    provenance_sha256 = sha256_json(provenance)
    for record in records:
        record["training_source_sha256"] = source_sha256
        record["provenance_bundle_sha256"] = provenance_sha256
    document = {
        "protocol": PROTOCOL_VERSION,
        "atomic_inventory": True,
        "analysis_started": False,
        "no_substitution": True,
        "root": str(root.resolve()),
        "data_dir": str(data_dir.resolve()),
        "checkpoint_step": int(checkpoint_step),
        "n_expected": 270,
        "n_resolved": len(records),
        "inventory_complete": len(records) == 270,
        "training_source_path": str(source_path),
        "training_source_sha256": source_sha256,
        "diagnostic_code_path": str(code_path),
        "diagnostic_code_sha256": sha256_file(code_path),
        "provenance_bundle_sha256": provenance_sha256,
        "provenance": provenance,
        "source_link_limit": provenance["historical_training_identity_limit"],
        "common_contracts": contracts,
        "entries": records,
    }
    if len(records) != 270:
        raise AssertionError("atomic inventory unexpectedly resolved fewer than 270 cells")
    checkpoint_manifest = out_dir / "CHECKPOINTS.json"
    _write_json(checkpoint_manifest, document, allow_identical=True)
    protocol = {
        "protocol": PROTOCOL_VERSION,
        "locked": True,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "diagnostic_code_sha256": document["diagnostic_code_sha256"],
        "training_source_sha256": document["training_source_sha256"],
        "provenance_bundle_sha256": document["provenance_bundle_sha256"],
        "checkpoint_config_contract": {
            "required_effective_fixed": jsonable(REQUIRED_CONFIG),
            "required_effective_per_cell": ["env", "seed", "tau", "mpi_steps"],
            "optional_raw_if_present": jsonable(OPTIONAL_CONFIG_IF_PRESENT),
            "max_action": MAX_ACTION,
            "compatibility": jsonable(CONFIG_COMPATIBILITY_CONTRACT),
            "stored_action_scalar_encodings": [
                "exact_float64_scale_times_max_action",
                "exact_ieee754_float32_scale_times_max_action",
            ],
        },
        "optimizer_state_contract": jsonable(OPTIMIZER_COMPATIBILITY_CONTRACT),
        "grid": {
            "environments": list(ENVIRONMENTS),
            "taus": list(TAUS),
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "checkpoint_step": int(checkpoint_step),
        },
        "common_batch": {
            "value_rows_per_environment": int(n_states),
            "residual_audit_rows_per_environment": int(n_states),
            "disjoint": True,
            "sample_seed_base": int(sample_seed),
            "transition_mask": "training-loader rows with not_done=1",
        },
        "common_noise": {
            "noise_seed_base": int(noise_seed),
            "same_clipped_epsilon_for_actor_and_recorded_next_action": True,
            "same_epsilon_across_methods_and_seeds_within_environment": True,
        },
        "formulas": {
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
        },
        "expected_counts": {
            "checkpoints": 270,
            "value_rows": 540,
            "geometry_rows": 270,
            "residual_rows": 450,
            "raw_npz": 270,
        },
        "value_scopes": [VALUE_SCOPE_OWN, VALUE_SCOPE_COMMON],
        "common_critic": "matched TD3 target critic at identical environment/tau/seed",
        "geometry_state": "same frozen next-state batch for target and final actors",
        "residual": {
            "hops": "k>=2 only",
            "critic_scope": RESIDUAL_SCOPE,
            "intervention": "posthoc_final_checkpoint_one_adam_step",
            "historical_training_step_recovered": False,
            "fresh_optimizer_allowed": False,
        },
        "collapse_label": f"external normalized return < {COLLAPSE_THRESHOLD:g}",
        "interpretation_limits": {
            "critic": "neither critic scope is an accuracy estimate",
            "causality": "movement/value/residual are post-hoc mechanism readouts",
            "residual": (
                "sampled feasible-comparator premise check; not a global "
                "optimality gap or bound on epsilon_k"
            ),
        },
    }
    protocol_path = out_dir / "FROZEN_PROTOCOL.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if canonical_json_bytes(existing) != canonical_json_bytes(protocol):
            raise RuntimeError(
                "FROZEN_PROTOCOL.json differs from current preflight; refusing overwrite"
            )
    else:
        _write_json(protocol_path, protocol)
    _write_json(
        out_dir / "PREFLIGHT_READY.json",
        {
            "state": "preflight_ready",
            "analysis_started": False,
            "run_allowed": True,
            "n_expected": 270,
            "n_resolved": 270,
            "checkpoint_manifest": "CHECKPOINTS.json",
            "frozen_protocol": "FROZEN_PROTOCOL.json",
        },
        allow_identical=True,
    )
    return True, document, arrays_by_env


def compute_target_value_metrics(
    *,
    q_actor: np.ndarray,
    q_dataset: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
) -> tuple[dict[str, float | int | bool], dict[str, np.ndarray]]:
    """Direct operational TD-target perturbation from already-minimized twins."""

    q_actor = np.asarray(q_actor, dtype=np.float64).reshape(-1)
    q_dataset = np.asarray(q_dataset, dtype=np.float64).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if q_actor.shape != q_dataset.shape or q_actor.shape != rewards.shape:
        raise ValueError("Q/reward arrays must have the same one-dimensional shape")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")
    delta_y = float(gamma) * (q_actor - q_dataset)
    target_actor = rewards + float(gamma) * q_actor
    target_dataset = rewards + float(gamma) * q_dataset
    all_finite = bool(
        np.all(np.isfinite(delta_y))
        and np.all(np.isfinite(target_actor))
        and np.all(np.isfinite(target_dataset))
    )
    summary: dict[str, float | int | bool] = {
        "n_rows": int(q_actor.size),
        "delta_y_rms": float(np.sqrt(np.mean(np.square(delta_y)))),
        "delta_y_abs_mean": float(np.mean(np.abs(delta_y))),
        "target_actor_rms": float(np.sqrt(np.mean(np.square(target_actor)))),
        "target_dataset_rms": float(np.sqrt(np.mean(np.square(target_dataset)))),
        "target_actor_abs_mean": float(np.mean(np.abs(target_actor))),
        "target_dataset_abs_mean": float(np.mean(np.abs(target_dataset))),
        "q_actor_abs_mean": float(np.mean(np.abs(q_actor))),
        "q_dataset_abs_mean": float(np.mean(np.abs(q_dataset))),
        "all_finite": all_finite,
    }
    return summary, {
        "q_actor": q_actor.astype(np.float32),
        "q_dataset": q_dataset.astype(np.float32),
        "delta_y": delta_y.astype(np.float32),
        "target_actor": target_actor.astype(np.float32),
        "target_dataset": target_dataset.astype(np.float32),
    }


def compute_same_next_state_geometry(
    *,
    target_actions: np.ndarray,
    final_actions: np.ndarray,
    dataset_actions: np.ndarray,
    smoothed_target_actions: np.ndarray,
    smoothed_dataset_actions: np.ndarray,
    max_action: float,
) -> dict[str, float | int | bool]:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            target_actions,
            final_actions,
            dataset_actions,
            smoothed_target_actions,
            smoothed_dataset_actions,
        )
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("geometry action arrays must have identical shapes")
    if arrays[0].ndim != 2:
        raise ValueError("geometry actions must be [state, action_dim]")
    target, final, dataset, smooth_target, smooth_dataset = arrays

    def rms(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(left - right))))

    target_rms = rms(target, dataset)
    final_rms = rms(final, dataset)
    ratio = float(final_rms / target_rms) if target_rms > 0.0 else float("nan")
    return {
        "n_rows": int(target.shape[0]),
        "action_dim": int(target.shape[1]),
        "target_to_next_data_rms": target_rms,
        "final_to_next_data_rms": final_rms,
        "final_to_target_rms": rms(final, target),
        "smoothed_target_to_next_data_rms": rms(smooth_target, smooth_dataset),
        "final_target_ratio": ratio,
        "target_saturation_fraction": float(
            np.mean(np.abs(target) >= float(max_action))
        ),
        "final_saturation_fraction": float(
            np.mean(np.abs(final) >= float(max_action))
        ),
        "smoothed_target_saturation_fraction": float(
            np.mean(np.abs(smooth_target) >= float(max_action))
        ),
        "smoothed_dataset_saturation_fraction": float(
            np.mean(np.abs(smooth_dataset) >= float(max_action))
        ),
        "all_finite": bool(
            all(np.all(np.isfinite(value)) for value in arrays)
            and np.isfinite(ratio)
        ),
    }


def comparator_residual_terms(
    *,
    q_current: np.ndarray,
    q_reference: np.ndarray,
    current_actions: np.ndarray,
    reference_actions: np.ndarray,
    q_weight: float,
) -> dict[str, float]:
    """Evaluate L(mu_k|ref)-L(ref|ref) for one fixed critic/reference/C_k."""

    q_current = np.asarray(q_current, dtype=np.float64).reshape(-1)
    q_reference = np.asarray(q_reference, dtype=np.float64).reshape(-1)
    current_actions = np.asarray(current_actions, dtype=np.float64)
    reference_actions = np.asarray(reference_actions, dtype=np.float64)
    if q_current.shape != q_reference.shape:
        raise ValueError("current/reference Q arrays differ in shape")
    if current_actions.shape != reference_actions.shape:
        raise ValueError("current/reference action arrays differ in shape")
    if current_actions.ndim != 2 or current_actions.shape[0] != q_current.size:
        raise ValueError("action arrays must align with Q rows")
    transport = float(np.mean(np.square(current_actions - reference_actions)))
    current_loss = -float(q_weight) * float(np.mean(q_current)) + transport
    comparator_loss = -float(q_weight) * float(np.mean(q_reference))
    residual = current_loss - comparator_loss
    return {
        "q_mean": float(np.mean(q_current)),
        "reference_q_mean": float(np.mean(q_reference)),
        "transport_mean_per_coordinate": transport,
        "actor_loss": current_loss,
        "comparator_loss": comparator_loss,
        "residual": residual,
        "feasible_comparator_slack": max(0.0, residual),
    }


def _actor_actions(model: Actor, params: Any, states: np.ndarray) -> np.ndarray:
    return np.asarray(jax.jit(model.apply)(params, states), dtype=np.float32)


def _twin_min_q(
    critic: TwinCritic, params: Any, states: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    def apply(p: Any, s: jax.Array, a: jax.Array) -> jax.Array:
        q1, q2 = critic.apply(p, s, a)
        return jnp.squeeze(jnp.minimum(q1, q2), axis=-1)

    return np.asarray(jax.jit(apply)(params, states, actions), dtype=np.float32)


def _q1(
    critic: TwinCritic, params: Any, states: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    def apply(p: Any, s: jax.Array, a: jax.Array) -> jax.Array:
        q, _ = critic.apply(p, s, a)
        return jnp.squeeze(q, axis=-1)

    return np.asarray(jax.jit(apply)(params, states, actions), dtype=np.float32)


def _posthoc_residual_interventions(
    *,
    payload: Mapping[str, Any],
    actors: Sequence[Any],
    actor_model: Actor,
    critic: TwinCritic,
    states: np.ndarray,
    tau: float,
    learning_rate: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """One stored-Adam intervention per k>=2, independently from final state."""

    total_hops = len(actors)
    if total_hops < 2:
        empty = np.empty((0, states.shape[0]), dtype=np.float32)
        return [], {
            "residual_hops": np.empty((0,), dtype=np.int16),
            "residual_q_weight": np.empty((0,), dtype=np.float64),
            "residual_C_k": np.empty((0,), dtype=np.float64),
            "residual_q_reference": empty,
            "residual_q_before": empty,
            "residual_q_after": empty,
            "residual_transport_before": empty,
            "residual_transport_after": empty,
        }
    optimizer_bundle = _stored_actor_optimizer_bundle(payload, total_hops)
    opt_states = optimizer_bundle.active_opt_states
    actor_steps = optimizer_bundle.active_steps
    critic_params = payload["critic_params"]
    optimizer = _audit_adam(learning_rate)
    tau_step = float(tau) / float(total_hops)
    rows: list[dict[str, Any]] = []
    raw: dict[str, list[Any]] = {
        "residual_hops": [],
        "residual_q_weight": [],
        "residual_C_k": [],
        "residual_q_reference": [],
        "residual_q_before": [],
        "residual_q_after": [],
        "residual_transport_before": [],
        "residual_transport_after": [],
    }
    for hop in range(2, total_hops + 1):
        actor_index = hop - 1
        reference = _actor_actions(actor_model, actors[actor_index - 1], states)
        before_actions = _actor_actions(actor_model, actors[actor_index], states)
        q_reference = _q1(critic, critic_params, states, reference)
        C_k = float(np.mean(np.abs(q_reference))) + 1e-6
        q_weight = 2.0 * tau_step / C_k

        def loss_fn(params: Any) -> jax.Array:
            actions = actor_model.apply(params, states)
            q1, _ = critic.apply(critic_params, states, actions)
            return -q_weight * jnp.mean(q1) + jnp.mean(
                jnp.square(actions - reference)
            )

        loss_before, grads = jax.value_and_grad(loss_fn)(actors[actor_index])
        state = TrainState.create(
            apply_fn=actor_model.apply,
            params=actors[actor_index],
            tx=optimizer,
        ).replace(
            opt_state=opt_states[actor_index],
            step=actor_steps[actor_index],
        )
        updated = state.apply_gradients(grads=grads)
        after_actions = _actor_actions(actor_model, updated.params, states)
        q_before = _q1(critic, critic_params, states, before_actions)
        q_after = _q1(critic, critic_params, states, after_actions)
        before = comparator_residual_terms(
            q_current=q_before,
            q_reference=q_reference,
            current_actions=before_actions,
            reference_actions=reference,
            q_weight=q_weight,
        )
        after = comparator_residual_terms(
            q_current=q_after,
            q_reference=q_reference,
            current_actions=after_actions,
            reference_actions=reference,
            q_weight=q_weight,
        )
        finite = bool(
            np.isfinite(float(loss_before))
            and all(np.isfinite(value) for value in before.values())
            and all(np.isfinite(value) for value in after.values())
        )
        rows.append(
            {
                "hop": hop,
                "total_hops": total_hops,
                "tau_step": tau_step,
                "C_k": C_k,
                "q_weight": q_weight,
                "optimizer_step_before": int(np.asarray(actor_steps[actor_index])),
                "optimizer_step_after": int(np.asarray(updated.step)),
                "r_before": before["residual"],
                "slack_before": before["feasible_comparator_slack"],
                "r_after": after["residual"],
                "slack_after": after["feasible_comparator_slack"],
                "actor_loss_before": before["actor_loss"],
                "actor_loss_after": after["actor_loss"],
                "comparator_loss": before["comparator_loss"],
                "q_mean_before": before["q_mean"],
                "q_mean_after": after["q_mean"],
                "reference_q_mean": before["reference_q_mean"],
                "transport_before": before["transport_mean_per_coordinate"],
                "transport_after": after["transport_mean_per_coordinate"],
                "all_finite": finite,
            }
        )
        raw["residual_hops"].append(hop)
        raw["residual_q_weight"].append(q_weight)
        raw["residual_C_k"].append(C_k)
        raw["residual_q_reference"].append(q_reference)
        raw["residual_q_before"].append(q_before)
        raw["residual_q_after"].append(q_after)
        raw["residual_transport_before"].append(
            np.mean(np.square(before_actions - reference), axis=-1)
        )
        raw["residual_transport_after"].append(
            np.mean(np.square(after_actions - reference), axis=-1)
        )
    packed = {
        "residual_hops": np.asarray(raw["residual_hops"], dtype=np.int16),
        "residual_q_weight": np.asarray(raw["residual_q_weight"], dtype=np.float64),
        "residual_C_k": np.asarray(raw["residual_C_k"], dtype=np.float64),
        "residual_q_reference": np.asarray(
            raw["residual_q_reference"], dtype=np.float32
        ),
        "residual_q_before": np.asarray(raw["residual_q_before"], dtype=np.float32),
        "residual_q_after": np.asarray(raw["residual_q_after"], dtype=np.float32),
        "residual_transport_before": np.asarray(
            raw["residual_transport_before"], dtype=np.float32
        ),
        "residual_transport_after": np.asarray(
            raw["residual_transport_after"], dtype=np.float32
        ),
    }
    return rows, packed


def _base_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": record["key"],
        "method": record["method"],
        "environment": record["environment"],
        "tau": float(record["tau"]),
        "seed": int(record["seed"]),
        "checkpoint_step": int(record["checkpoint_step"]),
        "checkpoint_file_sha256": record["checkpoint_file_sha256"],
        "weights_sha256": record["weights_sha256"],
        "config_sha256": record["config_sha256"],
        "dataset_sha256": record["dataset_sha256"],
        "normalization_sha256": record["normalization_sha256"],
        "selected_transition_sha256": record["selected_transition_sha256"],
        "common_noise_sha256": record["common_noise_sha256"],
        "external_score": float(record["external_score"]),
        "stability": "collapsed" if record["collapsed_lt20"] else "stable",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty required CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _analyze_cell(
    *,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    common_target_critic_params: Any,
    common_critic_sha256: str,
    arrays: Mapping[str, np.ndarray],
    raw_path: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(Path(str(record["checkpoint_path"]))) != record["checkpoint_file_sha256"]:
        raise RuntimeError(f"checkpoint changed after preflight: {record['key']}")
    actors = _actors(payload, int(record["hops"]))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    value_next_states = (
        np.asarray(arrays["value_next_observations"], dtype=np.float32) - mean
    ) / std
    residual_states = (
        np.asarray(arrays["residual_observations"], dtype=np.float32) - mean
    ) / std
    dataset_next_actions = np.asarray(arrays["value_next_actions"], dtype=np.float32)
    epsilon = np.asarray(arrays["epsilon"], dtype=np.float32)
    rewards = np.asarray(arrays["value_rewards"], dtype=np.float32)
    max_action = float(payload["max_action"])
    gamma = float(record["discount"])
    actor_model = Actor(
        action_dim=int(dataset_next_actions.shape[-1]), max_action=max_action
    )
    critic = TwinCritic()
    target_actions = _actor_actions(
        actor_model, payload["target_actor_params"], value_next_states
    )
    final_actions = _actor_actions(actor_model, actors[-1], value_next_states)
    smoothed_target = np.clip(
        target_actions + epsilon, -max_action, max_action
    ).astype(np.float32)
    smoothed_dataset = np.clip(
        dataset_next_actions + epsilon, -max_action, max_action
    ).astype(np.float32)
    deterministic_target = np.clip(
        target_actions, -max_action, max_action
    ).astype(np.float32)
    deterministic_dataset = np.clip(
        dataset_next_actions, -max_action, max_action
    ).astype(np.float32)
    own_q_actor = _twin_min_q(
        critic, payload["target_critic_params"], value_next_states, smoothed_target
    )
    own_q_dataset = _twin_min_q(
        critic, payload["target_critic_params"], value_next_states, smoothed_dataset
    )
    common_q_actor = _twin_min_q(
        critic, common_target_critic_params, value_next_states, smoothed_target
    )
    common_q_dataset = _twin_min_q(
        critic, common_target_critic_params, value_next_states, smoothed_dataset
    )
    own_q_actor_no_noise = _twin_min_q(
        critic, payload["target_critic_params"], value_next_states, deterministic_target
    )
    own_q_dataset_no_noise = _twin_min_q(
        critic, payload["target_critic_params"], value_next_states, deterministic_dataset
    )
    common_q_actor_no_noise = _twin_min_q(
        critic, common_target_critic_params, value_next_states, deterministic_target
    )
    common_q_dataset_no_noise = _twin_min_q(
        critic, common_target_critic_params, value_next_states, deterministic_dataset
    )
    base = _base_row(record)
    value_rows: list[dict[str, Any]] = []
    raw_value: dict[str, np.ndarray] = {}
    for scope, q_actor, q_dataset, q_actor_no_noise, q_dataset_no_noise, critic_hash in (
        (
            VALUE_SCOPE_OWN,
            own_q_actor,
            own_q_dataset,
            own_q_actor_no_noise,
            own_q_dataset_no_noise,
            sha256_tree(payload["target_critic_params"]),
        ),
        (
            VALUE_SCOPE_COMMON,
            common_q_actor,
            common_q_dataset,
            common_q_actor_no_noise,
            common_q_dataset_no_noise,
            common_critic_sha256,
        ),
    ):
        summary, raw = compute_target_value_metrics(
            q_actor=q_actor, q_dataset=q_dataset, rewards=rewards, gamma=gamma
        )
        no_noise_summary, no_noise_raw = compute_target_value_metrics(
            q_actor=q_actor_no_noise,
            q_dataset=q_dataset_no_noise,
            rewards=rewards,
            gamma=gamma,
        )
        no_noise_rms = float(no_noise_summary["delta_y_rms"])
        smoothing_change = float(summary["delta_y_rms"]) - no_noise_rms
        smoothing_ratio = (
            float(summary["delta_y_rms"]) / no_noise_rms
            if no_noise_rms > 0.0
            else float("nan")
        )
        combined_finite = bool(
            summary["all_finite"]
            and no_noise_summary["all_finite"]
            and np.isfinite(smoothing_change)
            and np.isfinite(smoothing_ratio)
        )
        value_rows.append(
            {
                **base,
                "critic_scope": scope,
                "critic_params_sha256": critic_hash,
                "gamma": gamma,
                "action_clip_target_fraction": float(
                    np.mean(np.abs(target_actions + epsilon) > max_action)
                ),
                "action_clip_dataset_fraction": float(
                    np.mean(np.abs(dataset_next_actions + epsilon) > max_action)
                ),
                **summary,
                "action_clip_target_fraction_no_noise": float(
                    np.mean(np.abs(target_actions) > max_action)
                ),
                "action_clip_dataset_fraction_no_noise": float(
                    np.mean(np.abs(dataset_next_actions) > max_action)
                ),
                "delta_y_rms_no_noise": no_noise_rms,
                "delta_y_abs_mean_no_noise": float(
                    no_noise_summary["delta_y_abs_mean"]
                ),
                "smoothing_delta_y_rms_change": smoothing_change,
                "smoothing_delta_y_rms_ratio": smoothing_ratio,
                "no_noise_all_finite": bool(no_noise_summary["all_finite"]),
                "all_finite": combined_finite,
            }
        )
        raw_value[f"{scope}_q_actor"] = raw["q_actor"]
        raw_value[f"{scope}_q_dataset"] = raw["q_dataset"]
        raw_value[f"{scope}_delta_y"] = raw["delta_y"]
        raw_value[f"{scope}_target_actor"] = raw["target_actor"]
        raw_value[f"{scope}_target_dataset"] = raw["target_dataset"]
        raw_value[f"{scope}_q_actor_no_noise"] = no_noise_raw["q_actor"]
        raw_value[f"{scope}_q_dataset_no_noise"] = no_noise_raw["q_dataset"]
        raw_value[f"{scope}_delta_y_no_noise"] = no_noise_raw["delta_y"]
        raw_value[f"{scope}_target_actor_no_noise"] = no_noise_raw["target_actor"]
        raw_value[f"{scope}_target_dataset_no_noise"] = no_noise_raw["target_dataset"]
    geometry = {
        **base,
        **compute_same_next_state_geometry(
            target_actions=target_actions,
            final_actions=final_actions,
            dataset_actions=dataset_next_actions,
            smoothed_target_actions=smoothed_target,
            smoothed_dataset_actions=smoothed_dataset,
            max_action=max_action,
        ),
    }
    residual_rows, residual_raw = _posthoc_residual_interventions(
        payload=payload,
        actors=actors,
        actor_model=actor_model,
        critic=critic,
        states=residual_states,
        tau=float(record["tau"]),
        learning_rate=float(record["learning_rate"]),
    )
    residual_rows = [
        {
            **base,
            "critic_scope": RESIDUAL_SCOPE,
            "critic_params_sha256": sha256_tree(payload["critic_params"]),
            "intervention": "posthoc_final_checkpoint_one_adam_step",
            "historical_training_step_recovered": False,
            **row,
        }
        for row in residual_rows
    ]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            gamma=np.asarray(gamma, dtype=np.float64),
            rewards=rewards,
            epsilon=epsilon,
            target_actions=target_actions,
            final_actions=final_actions,
            dataset_next_actions=dataset_next_actions,
            smoothed_target_actions=smoothed_target,
            smoothed_dataset_actions=smoothed_dataset,
            deterministic_target_actions=deterministic_target,
            deterministic_dataset_actions=deterministic_dataset,
            **raw_value,
            **residual_raw,
        )
    raw_sha256 = sha256_file(raw_path)
    raw_reference = _relative_artifact(out_dir, raw_path)
    for row in value_rows:
        row["raw_npz"] = raw_reference
        row["raw_sha256"] = raw_sha256
    geometry["raw_npz"] = raw_reference
    geometry["raw_sha256"] = raw_sha256
    for row in residual_rows:
        row["raw_npz"] = raw_reference
        row["raw_sha256"] = raw_sha256
    return value_rows, geometry, residual_rows


def _method_contrasts(value_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["method"]),
            str(row["environment"]),
            float(row["tau"]),
            int(row["seed"]),
            str(row["critic_scope"]),
        ): row
        for row in value_rows
    }
    metric_fields = (
        "delta_y_rms_difference",
        "delta_y_rms_no_noise_difference",
        "smoothing_delta_y_rms_change_difference",
    )
    seed_interpretation = (
        "seeds are marginal run replicates within fixed task-budget cells"
    )
    generalization_scope = "fixed nine-task, five-budget grid only"

    def _means(selected: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        if not selected:
            raise AssertionError("contrast aggregation selected no rows")
        return {
            field: float(np.mean([float(row[field]) for row in selected]))
            for field in metric_fields
        }

    def _labels() -> dict[str, str]:
        return {
            "seed_interpretation": seed_interpretation,
            "generalization_scope": generalization_scope,
        }

    rows: list[dict[str, Any]] = []
    for method in ("p3", "p4"):
        for scope in (VALUE_SCOPE_OWN, VALUE_SCOPE_COMMON):
            cell_differences: list[dict[str, Any]] = []
            for environment in ENVIRONMENTS:
                for tau in TAUS:
                    for seed in SEEDS:
                        method_row = lookup[(method, environment, tau, seed, scope)]
                        td3_row = lookup[("td3", environment, tau, seed, scope)]
                        cell_differences.append(
                            {
                                "level": "cell",
                                "method_minus": method,
                                "method_reference": "td3",
                                "critic_scope": scope,
                                "environment": environment,
                                "tau": tau,
                                "seed": seed,
                                "delta_y_rms_difference": float(
                                    method_row["delta_y_rms"]
                                )
                                - float(td3_row["delta_y_rms"]),
                                "delta_y_rms_no_noise_difference": float(
                                    method_row["delta_y_rms_no_noise"]
                                )
                                - float(td3_row["delta_y_rms_no_noise"]),
                                "smoothing_delta_y_rms_change_difference": float(
                                    method_row["smoothing_delta_y_rms_change"]
                                )
                                - float(td3_row["smoothing_delta_y_rms_change"]),
                                **_labels(),
                            }
                        )
            rows.extend(cell_differences)
            task_budget: list[dict[str, Any]] = []
            for environment in ENVIRONMENTS:
                for tau in TAUS:
                    selected = [
                        row
                        for row in cell_differences
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
                            **_means(selected),
                            **_labels(),
                        }
                    )
            rows.extend(task_budget)
            task_means: list[dict[str, Any]] = []
            for environment in ENVIRONMENTS:
                selected = [
                    row
                    for row in cell_differences
                    if row["environment"] == environment
                ]
                task_means.append(
                    {
                        "level": "task_marginal_mean",
                        "method_minus": method,
                        "method_reference": "td3",
                        "critic_scope": scope,
                        "environment": environment,
                        "tau": "",
                        "seed": "",
                        **_means(selected),
                        **_labels(),
                    }
                )
            rows.extend(task_means)
            rows.append(
                {
                    "level": "task_equal_overall",
                    "method_minus": method,
                    "method_reference": "td3",
                    "critic_scope": scope,
                    "environment": "",
                    "tau": "",
                    "seed": "",
                    **_means(task_means),
                    **_labels(),
                }
            )
    return rows


def _residual_aggregates(
    residual_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in residual_rows:
        key = (
            row["method"],
            int(row["hop"]),
            float(row["tau"]),
            row["environment"],
            row["stability"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (method, hop, tau, environment, stability), rows in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "hop": hop,
                "tau": tau,
                "environment": environment,
                "stability": stability,
                "n_seed_cells": len(rows),
                "r_before_mean": float(np.mean([float(row["r_before"]) for row in rows])),
                "slack_before_mean": float(
                    np.mean([float(row["slack_before"]) for row in rows])
                ),
                "r_after_mean": float(np.mean([float(row["r_after"]) for row in rows])),
                "slack_after_mean": float(
                    np.mean([float(row["slack_after"]) for row in rows])
                ),
            }
        )
    return output


def _geometry_correlations(
    geometry_rows: Sequence[Mapping[str, Any]],
    value_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    own = {
        str(row["key"]): float(row["delta_y_rms"])
        for row in value_rows
        if row["critic_scope"] == VALUE_SCOPE_OWN
    }
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for stability in ("all", "stable", "collapsed"):
            selected = [
                row
                for row in geometry_rows
                if row["method"] == method
                and (stability == "all" or row["stability"] == stability)
            ]
            x = np.asarray(
                [float(row["final_target_ratio"]) for row in selected],
                dtype=np.float64,
            )
            y = np.asarray([own[str(row["key"])] for row in selected], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            correlation = (
                float(np.corrcoef(x[finite], y[finite])[0, 1])
                if int(np.sum(finite)) >= 2
                and float(np.std(x[finite])) > 0
                and float(np.std(y[finite])) > 0
                else float("nan")
            )
            output.append(
                {
                    "method": method,
                    "stability": stability,
                    "n_cells": int(np.sum(finite)),
                    "pearson_final_target_ratio_vs_within_run_delta_y_rms": correlation,
                    "external_stability_definition": "normalized return <20 is collapsed",
                }
            )
    return output


def run_analysis(
    *,
    out_dir: Path,
    inventory: Mapping[str, Any],
    arrays_by_env: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    backend = _require_cpu_backend()
    records = {
        str(row["key"]): row for row in inventory["entries"]
    }
    expected = {
        cell_key(method, environment, tau, seed)
        for environment in ENVIRONMENTS
        for tau in TAUS
        for seed in SEEDS
        for method in METHODS
    }
    if set(records) != expected or len(records) != 270:
        raise RuntimeError("analysis received a noncanonical inventory")
    value_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for environment in ENVIRONMENTS:
        arrays = arrays_by_env[environment]
        if sha256_arrays(arrays) != inventory["common_contracts"][environment][
            "common_content_sha256"
        ]:
            raise RuntimeError(f"common batch/noise changed: {environment}")
        for tau in TAUS:
            for seed in SEEDS:
                group_records = {
                    method: records[cell_key(method, environment, tau, seed)]
                    for method in METHODS
                }
                payloads = {
                    method: load_checkpoint(Path(str(record["checkpoint_path"])))
                    for method, record in group_records.items()
                }
                common_params = payloads["td3"]["target_critic_params"]
                common_hash = sha256_tree(common_params)
                td3_own_hash = sha256_tree(payloads["td3"]["target_critic_params"])
                if common_hash != td3_own_hash:
                    raise AssertionError("TD3 own/common critic identity failure")
                for method in METHODS:
                    record = group_records[method]
                    raw_path = (
                        out_dir
                        / "raw"
                        / environment
                        / f"{run_name(method, environment, tau, seed)}.npz"
                    )
                    cell_values, geometry, cell_residuals = _analyze_cell(
                        record=record,
                        payload=payloads[method],
                        common_target_critic_params=common_params,
                        common_critic_sha256=common_hash,
                        arrays=arrays,
                        raw_path=raw_path,
                        out_dir=out_dir,
                    )
                    value_rows.extend(cell_values)
                    geometry_rows.append(geometry)
                    residual_rows.extend(cell_residuals)
                    raw_hashes[_relative_artifact(out_dir, raw_path)] = sha256_file(
                        raw_path
                    )
    if len(value_rows) != EXPECTED_VALUE_ROWS:
        raise AssertionError(f"value rows={len(value_rows)} != {EXPECTED_VALUE_ROWS}")
    if len(geometry_rows) != EXPECTED_GEOMETRY_ROWS:
        raise AssertionError(
            f"geometry rows={len(geometry_rows)} != {EXPECTED_GEOMETRY_ROWS}"
        )
    if len(residual_rows) != EXPECTED_RESIDUAL_ROWS:
        raise AssertionError(
            f"residual rows={len(residual_rows)} != {EXPECTED_RESIDUAL_ROWS}"
        )
    residual_keys = {
        (row["key"], int(row["hop"])) for row in residual_rows
    }
    if len(residual_keys) != EXPECTED_RESIDUAL_ROWS or any(
        int(row["hop"]) < 2 for row in residual_rows
    ):
        raise AssertionError("residual keys are duplicated or include k=1")
    value_path = out_dir / "td_target_value.csv"
    geometry_path = out_dir / "same_next_state_geometry.csv"
    residual_path = out_dir / "comparator_residuals.csv"
    contrast_path = out_dir / "method_contrasts.csv"
    residual_aggregate_path = out_dir / "residual_aggregates.csv"
    correlation_path = out_dir / "geometry_correlations.csv"
    contrast_rows = _method_contrasts(value_rows)
    residual_aggregate_rows = _residual_aggregates(residual_rows)
    correlation_rows = _geometry_correlations(geometry_rows, value_rows)
    _write_csv(value_path, value_rows)
    _write_csv(geometry_path, geometry_rows)
    _write_csv(residual_path, residual_rows)
    _write_csv(contrast_path, contrast_rows)
    _write_csv(residual_aggregate_path, residual_aggregate_rows)
    _write_csv(correlation_path, correlation_rows)
    artifact_paths = (
        value_path,
        geometry_path,
        residual_path,
        contrast_path,
        residual_aggregate_path,
        correlation_path,
    )
    finite_gate = bool(
        all(bool(row["all_finite"]) for row in value_rows)
        and all(bool(row["all_finite"]) for row in geometry_rows)
        and all(bool(row["all_finite"]) for row in residual_rows)
    )
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "status": "analysis_complete",
        "cpu_only": True,
        "jax_backend": backend,
        "operational_readout": "stored learned target critic; not critic accuracy",
        "audit_batch": "fixed audit batch; no holdout or generalization claim",
        "residual_intervention": "posthoc_final_checkpoint_one_adam_step",
        "historical_training_step_recovered": False,
        "inventory_sha256": sha256_file(out_dir / "CHECKPOINTS.json"),
        "protocol_sha256": sha256_file(out_dir / "FROZEN_PROTOCOL.json"),
        "counts": {
            "checkpoints": 270,
            "value_rows": len(value_rows),
            "geometry_rows": len(geometry_rows),
            "residual_rows": len(residual_rows),
            "raw_npz": len(raw_hashes),
        },
        "derived_counts": {
            "method_contrast_rows": len(contrast_rows),
            "residual_aggregate_rows": len(residual_aggregate_rows),
            "geometry_correlation_rows": len(correlation_rows),
        },
        "artifact_sha256": {
            _relative_artifact(out_dir, path): sha256_file(path)
            for path in artifact_paths
        },
        "raw_sha256": raw_hashes,
        "finite_value_gate": finite_gate,
        "scientific_inclusion_gate": (
            "requires independent VERIFY.json pass in addition to this finite flag"
        ),
        "verification_scope": (
            "artifact hashes and independently recomputed arithmetic; the "
            "artifact-only verifier does not reexecute checkpoint networks"
        ),
        "interpretation_limits": {
            "critic_accuracy": False,
            "causal_return_effect": False,
            "global_optimality_gap": False,
            "epsilon_k_bound": False,
            "broader_task_generalization": False,
            "seed_role": (
                "marginal run replicates within the fixed task-budget grid"
            ),
        },
        "written_at": now_iso(),
    }
    _write_json(out_dir / "MANIFEST.json", manifest)
    _write_json(
        out_dir / "RUN_STATUS.json",
        {
            "state": "analysis_complete_unverified",
            "analysis_started": True,
            "run_allowed": True,
            "independent_verifier_required": True,
            "finite_value_gate": finite_gate,
            "written_at": now_iso(),
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("dry-run", "preflight", "run"), default="dry-run"
    )
    parser.add_argument("--root", default=os.environ.get("RESULTS_ROOT", str(REPO_ROOT)))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
    parser.add_argument(
        "--out-dir",
        required=True,
        help="New or preflight-only bundle directory; use a path outside the repository.",
    )
    parser.add_argument("--checkpoint-step", type=int, default=CHECKPOINT_STEP)
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260902)
    parser.add_argument("--noise-seed", type=int, default=20260903)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_step != CHECKPOINT_STEP:
        raise ValueError("P1 is locked to params_1000000.pkl; checkpoint-step is not flexible")
    if args.n_states < 1:
        raise ValueError("n-states must be positive")
    if args.max_episode_steps < 1:
        raise ValueError("max-episode-steps must be positive")
    root = Path(args.root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    try:
        out_dir.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("--out-dir must be outside the repository worktree")
    out_dir.mkdir(parents=True, exist_ok=True)
    _guard_output_lifecycle(out_dir, args.mode)
    if args.mode == "dry-run":
        document = write_expected_grid(out_dir, root, args.checkpoint_step)
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "n_expected": document["n_expected"],
                    "analysis_started": False,
                    "artifact": str((out_dir / "EXPECTED_GRID.json").resolve()),
                },
                indent=2,
            )
        )
        return 0
    ready, inventory, arrays_by_env = preflight(
        root=root,
        data_dir=data_dir,
        out_dir=out_dir,
        checkpoint_step=args.checkpoint_step,
        n_states=args.n_states,
        sample_seed=args.sample_seed,
        noise_seed=args.noise_seed,
        max_episode_steps=args.max_episode_steps,
    )
    if not ready:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "ready": False,
                    "analysis_started": False,
                    "stage": inventory["validation_stage"],
                    "failures": len(inventory["failures"]),
                },
                indent=2,
            )
        )
        return 2
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "mode": "preflight",
                    "ready": True,
                    "n_resolved": inventory["n_resolved"],
                    "analysis_started": False,
                },
                indent=2,
            )
        )
        return 0
    manifest = run_analysis(
        out_dir=out_dir, inventory=inventory, arrays_by_env=arrays_by_env
    )
    print(json.dumps({"mode": "run", **manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
