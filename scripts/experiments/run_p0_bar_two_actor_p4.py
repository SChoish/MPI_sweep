#!/usr/bin/env python3
"""Freeze, plan, and explicitly launch the decisive BAR-P4 control experiment.

With no subcommand this program only prints the fixed grid.  ``launch`` also
remains read-only unless ``--execute`` is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "p0_bar_p4_vs_two_actor_p4"
ENVIRONMENTS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
DATASET_FILES = {
    "halfcheetah-medium-v2": "halfcheetah_medium-v2.hdf5",
    "halfcheetah-medium-replay-v2": "halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah-expert-v2": "halfcheetah_expert-v2.hdf5",
    "hopper-medium-v2": "hopper_medium-v2.hdf5",
    "hopper-medium-replay-v2": "hopper_medium_replay-v2.hdf5",
    "hopper-expert-v2": "hopper_expert-v2.hdf5",
    "walker2d-medium-v2": "walker2d_medium-v2.hdf5",
    "walker2d-medium-replay-v2": "walker2d_medium_replay-v2.hdf5",
    "walker2d-expert-v2": "walker2d_expert-v2.hdf5",
}
T_VALUES = (4, 7, 10, 14, 20)
SEEDS = (0, 1)
CONDITIONS = ("bar_p4", "two_actor_p4")
THRESHOLDS = (0, 10, 20, 30, 40)
SCIENTIFIC_CONFIG_KEYS = (
    "method",
    "env",
    "seed",
    "tau",
    "mpi_steps",
    "integrator",
    "max_timesteps",
    "eval_freq",
    "eval_episodes",
    "batch_size",
    "discount",
    "polyak",
    "policy_noise",
    "noise_clip",
    "policy_freq",
    "lr",
    "normalize",
    "q_scale_norm",
    "n_jitted_updates",
    "updates_per_dispatch",
)
SOURCE_FILES = (
    "train_td3bc.py",
    "launch_mpi_sweep.py",
    "d4rl_data.py",
    "tau_grids.py",
    "requirements.txt",
    "scripts/experiments/run_p0_bar_two_actor_p4.py",
    "scripts/diagnostics/analyze_p0_bar_two_actor_p4.py",
    "scripts/diagnostics/verify_p0_bar_two_actor_p4.py",
)
WEIGHT_KEYS = (
    "actors_params",
    "critic_params",
    "target_actor_params",
    "target_critic_params",
)
CHECKPOINT_KEYS = (
    "step",
    "rng",
    "mean",
    "std",
    "config",
    "max_action",
    "policy_noise",
    "noise_clip",
    "actors_params",
    "actors_steps",
    "critic_params",
    "critic_step",
    "target_actor_params",
    "target_critic_params",
    "actors_opt_states",
    "critic_opt_state",
)


@dataclass(frozen=True)
class LaunchJob:
    run: dict[str, Any]
    command: list[str]
    resume_checkpoint: Path | None


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def normalization_sha256(mean: Any, std: Any) -> str:
    return json_sha256({"mean": array_sha256(mean), "std": array_sha256(std)})


def hash_tree(value: Any) -> str:
    """Hash a nested checkpoint tree with type and length framing."""
    digest = hashlib.sha256()

    def update(node: Any) -> None:
        if isinstance(node, Mapping):
            digest.update(b"M")
            digest.update(len(node).to_bytes(8, "big"))
            for key in sorted(node, key=lambda item: str(item)):
                key_bytes = canonical_json([type(key).__name__, str(key)])
                digest.update(len(key_bytes).to_bytes(8, "big"))
                digest.update(key_bytes)
                update(node[key])
        elif isinstance(node, (tuple, list)):
            digest.update(b"S")
            digest.update(len(node).to_bytes(8, "big"))
            for item in node:
                update(item)
        elif hasattr(node, "shape") and hasattr(node, "dtype"):
            digest.update(b"A")
            digest.update(bytes.fromhex(array_sha256(node)))
        elif isinstance(node, (str, int, float, bool)) or node is None:
            encoded = canonical_json(node)
            digest.update(b"P")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        else:
            raise TypeError(
                f"unsupported checkpoint value for hashing: {type(node)!r}"
            )

    update(value)
    return digest.hexdigest()


def require_finite_tree(value: Any, label: str) -> None:
    """Reject nonnumeric or nonfinite leaves in a trusted checkpoint tree."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}[{index}]")
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"checkpoint {label} has nonnumeric dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"checkpoint {label} contains a nonfinite value")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(value):
            raise ValueError(f"checkpoint {label} contains a nonfinite value")
    else:
        raise ValueError(
            f"checkpoint {label} has unsupported leaf type {type(value)!r}"
        )


def scalar_integer(value: Any, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"checkpoint {label} must be one integer scalar")
    return int(array)


def scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in SCIENTIFIC_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"config is missing scientific keys: {missing}")
    return {key: config[key] for key in SCIENTIFIC_CONFIG_KEYS}


def run_key(condition: str, environment: str, tau: int, seed: int) -> str:
    return f"{condition}|{environment}|T={tau}|seed={seed}"


def resolved_config(condition: str, environment: str, tau: int, seed: int) -> dict:
    return {
        "method": "bar" if condition == "bar_p4" else "mcep",
        "env": environment,
        "seed": seed,
        "tau": float(tau),
        "mpi_steps": 4,
        "integrator": "implicit",
        "max_timesteps": 1_000_000,
        "eval_freq": 1_000_000,
        "eval_episodes": 10,
        "batch_size": 256,
        "discount": 0.99,
        "polyak": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "lr": 0.0003,
        "normalize": True,
        "q_scale_norm": True,
        "n_jitted_updates": 8,
        "updates_per_dispatch": 64,
    }


def build_runs(output_root: Path) -> list[dict[str, Any]]:
    output_root = output_root.resolve()
    runs: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    method_tag = "mpi4" if condition == "bar_p4" else "mcep4"
                    tag = f"{environment}_tau{tau}_{method_tag}_seed{seed}"
                    config = resolved_config(condition, environment, tau, seed)
                    condition_root = output_root / condition
                    runs.append(
                        {
                            "key": run_key(condition, environment, tau, seed),
                            "condition": condition,
                            "environment": environment,
                            "T": tau,
                            "seed": seed,
                            "tag": tag,
                            "output_root": str(condition_root),
                            "output_dir": str(condition_root / tag),
                            "score_column": (
                                "d4rl_pi4" if condition == "bar_p4" else "d4rl_eval"
                            ),
                            "resolved_config": config,
                            "config_sha256": json_sha256(config),
                        }
                    )
    validate_run_grid(runs)
    return runs


def validate_run_grid(runs: list[dict[str, Any]]) -> None:
    expected = {
        run_key(condition, environment, tau, seed)
        for condition in CONDITIONS
        for environment in ENVIRONMENTS
        for tau in T_VALUES
        for seed in SEEDS
    }
    actual = [run["key"] for run in runs]
    if len(actual) != 180 or len(set(actual)) != 180 or set(actual) != expected:
        raise ValueError("P0 run grid must contain exactly 180 unique expected keys")
    condition_roots = {}
    for condition in CONDITIONS:
        condition_runs = [run for run in runs if run["condition"] == condition]
        if len(condition_runs) != 90:
            raise ValueError(f"condition {condition} must contain exactly 90 runs")
        roots = {run["output_root"] for run in condition_runs}
        if len(roots) != 1:
            raise ValueError(f"condition {condition} must have one output root")
        condition_root = Path(roots.pop())
        if not condition_root.is_absolute() or condition_root.name != condition:
            raise ValueError(f"condition {condition} output root contract drift")
        condition_roots[condition] = condition_root
    if len({root.parent for root in condition_roots.values()}) != 1:
        raise ValueError("both conditions must share one external result parent")
    for run in runs:
        condition = run["condition"]
        environment = run["environment"]
        if type(run["T"]) is not int or type(run["seed"]) is not int:
            raise ValueError(f"run T/seed types drift: {run['key']}")
        tau = run["T"]
        seed = run["seed"]
        if (
            condition not in CONDITIONS
            or environment not in ENVIRONMENTS
            or tau not in T_VALUES
            or seed not in SEEDS
        ):
            raise ValueError(f"run fields fall outside the frozen grid: {run['key']}")
        expected_tag = (
            f"{environment}_tau{tau}_"
            f"{'mpi4' if condition == 'bar_p4' else 'mcep4'}_seed{seed}"
        )
        expected_score = "d4rl_pi4" if condition == "bar_p4" else "d4rl_eval"
        config = resolved_config(condition, environment, tau, seed)
        if run["key"] != run_key(condition, environment, tau, seed):
            raise ValueError(f"run key/fields disagree: {run['key']}")
        if run["tag"] != expected_tag or run["score_column"] != expected_score:
            raise ValueError(f"run tag/score contract drift: {run['key']}")
        if Path(run["output_dir"]) != Path(run["output_root"]) / expected_tag:
            raise ValueError(f"run output path contract drift: {run['key']}")
        if run["resolved_config"] != config or run["config_sha256"] != json_sha256(config):
            raise ValueError(f"run scientific config contract drift: {run['key']}")


def _normalization_record(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        observations = np.asarray(handle["observations"], dtype=np.float32)
        terminals = np.asarray(handle["terminals"], dtype=np.float32)
        timeouts = (
            np.asarray(handle["timeouts"], dtype=np.float32)
            if "timeouts" in handle
            else None
        )
    kept: list[int] = []
    episode_step = 0
    for index in range(observations.shape[0] - 1):
        done = bool(terminals[index])
        final_timestep = (
            bool(timeouts[index]) if timeouts is not None else episode_step == 999
        )
        if final_timestep:
            episode_step = 0
            continue
        episode_step = 0 if done else episode_step + 1
        kept.append(index)
    training_observations = observations[np.asarray(kept, dtype=np.int64)]
    mean = training_observations.mean(axis=0)
    std = training_observations.std(axis=0) + np.float32(1e-3)
    return {
        "algorithm": "train_td3bc.qlearning_from_hdf5 kept-observation mean/std; eps=1e-3",
        "training_rows": int(training_observations.shape[0]),
        "state_dimension": int(training_observations.shape[1]),
        "mean_sha256": array_sha256(mean),
        "std_sha256": array_sha256(std),
        "statistics_sha256": normalization_sha256(mean, std),
    }


def dataset_records(data_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for environment in ENVIRONMENTS:
        path = (data_dir / DATASET_FILES[environment]).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing frozen dataset for {environment}: {path}")
        records[environment] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "normalization": _normalization_record(path),
        }
    return records


def dependency_snapshot() -> dict[str, Any]:
    python_executable = Path(sys.executable).resolve()
    packages = sorted(
        {
            f"{dist.metadata.get('Name', dist.name).lower()}=={dist.version}"
            for dist in importlib.metadata.distributions()
        }
    )
    return {
        "python_executable": str(python_executable),
        "python_executable_sha256": file_sha256(python_executable),
        "python_version": platform.python_version(),
        "resolved_packages": packages,
        "resolved_packages_sha256": json_sha256(packages),
        "requirements_sha256": file_sha256(ROOT / "requirements.txt"),
    }


def hardware_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("GPU freeze requires a working nvidia-smi") from error
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        gpus.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "memory_mib": fields[4],
            }
        )
    if not gpus:
        raise RuntimeError("GPU freeze resolved zero devices")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "gpus": gpus,
    }


def source_identity_snapshot() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hashes = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing source contract file: {path}")
        hashes[relative] = file_sha256(path)
    return {"git_revision": revision, "files": hashes}


def source_snapshot() -> dict[str, Any]:
    identity = source_identity_snapshot()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("freeze requires a completely clean committed checkout")
    return {**identity, "tracked_worktree_clean": True}


def build_command(manifest: Mapping[str, Any], run: Mapping[str, Any]) -> list[str]:
    config = run["resolved_config"]
    execution = manifest["execution"]
    command = [
        execution["python"],
        "-u",
        str(ROOT / "train_td3bc.py"),
        "--env", str(config["env"]),
        "--tau", f"{float(config['tau']):g}",
        "--mpi-steps", str(config["mpi_steps"]),
        "--method", str(config["method"]),
        "--integrator", str(config["integrator"]),
        "--seed", str(config["seed"]),
        "--max-timesteps", str(config["max_timesteps"]),
        "--eval-freq", str(config["eval_freq"]),
        "--eval-episodes", str(config["eval_episodes"]),
        "--batch-size", str(config["batch_size"]),
        "--discount", str(config["discount"]),
        "--polyak", str(config["polyak"]),
        "--policy-noise", str(config["policy_noise"]),
        "--noise-clip", str(config["noise_clip"]),
        "--policy-freq", str(config["policy_freq"]),
        "--lr", str(config["lr"]),
        "--n-jitted-updates", str(config["n_jitted_updates"]),
        "--updates-per-dispatch", str(config["updates_per_dispatch"]),
        "--compilation-cache-dir", execution["compilation_cache_dir"],
        "--save-interval", str(execution["save_interval"]),
        "--data-dir", manifest["dataset_root"],
        "--save-dir", run["output_root"],
        "--normalize",
        "--q-scale-norm",
        "--no-resume",
    ]
    return command


def validate_manifest_commands(manifest: Mapping[str, Any]) -> None:
    """Require every frozen launch command to match its locked run fields."""
    execution = manifest["execution"]
    selected_gpus = list(execution["gpus"])
    available_gpus = {
        str(gpu["index"]) for gpu in manifest["hardware"].get("gpus", ())
    }
    if (
        not selected_gpus
        or len(selected_gpus) != len(set(selected_gpus))
        or not set(selected_gpus).issubset(available_gpus)
        or int(execution["slots_per_gpu"]) < 1
    ):
        raise ValueError("manifest execution GPU contract drift")
    if Path(execution["python"]).resolve() != Path(
        manifest["dependencies"]["python_executable"]
    ).resolve():
        raise ValueError("manifest execution Python differs from dependency snapshot")
    for run in manifest["runs"]:
        if run.get("command") != build_command(manifest, run):
            raise ValueError(f"frozen launch command drift: {run['key']}")


def build_manifest(
    *,
    data_dir: Path,
    output_root: Path,
    python: Path,
    gpus: list[str],
    slots_per_gpu: int,
    compilation_cache_dir: Path,
    datasets: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    hardware: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if slots_per_gpu < 1 or not gpus:
        raise ValueError("freeze requires GPUs and positive slots-per-gpu")
    if len(set(gpus)) != len(gpus):
        raise ValueError("freeze GPU selection must not contain duplicates")
    available_gpus = {str(gpu["index"]) for gpu in hardware.get("gpus", ())}
    if not set(gpus).issubset(available_gpus):
        raise ValueError(
            f"freeze requested GPUs {gpus}, but inventory exposes "
            f"{sorted(available_gpus)}"
        )
    resolved_output_root = output_root.resolve()
    if resolved_output_root == ROOT or ROOT in resolved_output_root.parents:
        raise ValueError("freeze requires an output root outside the source checkout")
    runs = build_runs(output_root)
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen_before_launch",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {
            "environments": list(ENVIRONMENTS),
            "T": list(T_VALUES),
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "runs_per_condition": 90,
            "total_runs": 180,
            "schedule": "cell-paired interleaving of BAR-P4 and two-actor-P4",
        },
        "procedure_contract": {
            "bar_p4": (
                "four persistent sequentially re-centered actors; only actor1 "
                "supplies Bellman targets; actor4 is deployed"
            ),
            "two_actor_p4": (
                "two independently initialized dataset-anchored actors; target "
                "coefficient T/4 supplies Bellman targets; deployment coefficient "
                "T is excluded from Bellman targets"
            ),
            "interpretation": (
                "full-procedure comparison, not a re-centering-only causal "
                "isolation; two-actor policy-separation control (legacy mcep "
                "token; not published reproduction)"
            ),
        },
        "dataset_root": str(data_dir.resolve()),
        "datasets": dict(datasets),
        "dependencies": dict(dependencies),
        "hardware": dict(hardware),
        "source": dict(source),
        "execution": {
            "python": str(python.resolve()),
            "gpus": gpus,
            "slots_per_gpu": slots_per_gpu,
            "compilation_cache_dir": str(compilation_cache_dir.resolve()),
            "save_interval": 250_000,
            "launch_requires_explicit_execute": True,
        },
        "evaluation_contract": {
            "critic_updates": 1_000_000,
            "final_evaluation_episodes": 10,
            "evaluation_frequency": 1_000_000,
            "target_score_column": "d4rl_score",
            "bar_deployment_score_column": "d4rl_pi4",
            "two_actor_deployment_score_column": "d4rl_eval",
        },
        "output_contract": {
            "config": "<output_dir>/config.json",
            "evaluation": "<output_dir>/eval.csv",
            "final_checkpoint": "<output_dir>/params_1000000.pkl",
            "bar_eval_columns": [
                "step",
                "return",
                "d4rl_score",
                "critic_loss",
                "actor_loss",
                "return_pi4",
                "d4rl_pi4",
                "final_actor_loss",
            ],
            "two_actor_eval_columns": [
                "step",
                "return",
                "d4rl_score",
                "critic_loss",
                "actor_loss",
                "return_eval",
                "d4rl_eval",
                "final_actor_loss",
            ],
            "checkpoint_keys": list(CHECKPOINT_KEYS),
        },
        "resume_contract": {
            "automatic_resume": False,
            "required_registry": "create-only resume registry tied to this frozen manifest",
            "required_checkpoint_hashes": [
                "file_sha256",
                "scientific_config_sha256",
                "weights_sha256",
                "normalization_sha256",
            ],
        },
        "analysis_contract": {
            "primary_contrast": "BAR-P4 deployment minus two-actor-P4 deployment",
            "primary_estimator": "mean of nine fixed task means (balanced 90-cell grid)",
            "paired_median_and_wins": True,
            "bootstrap": {
                "resampling_unit": "fixed task variant",
                "draws": 100_000,
                "seed": 20260902,
                "interval": "percentile 95%; numpy.quantile linear",
            },
            "inference_scope": (
                "fixed 9-task x 5-T x 2-seed grid across three shared dynamics "
                "families; not broader offline-RL generalization"
            ),
            "collapse_thresholds": list(THRESHOLDS),
            "collapse_units": ["raw_run", "two_seed_mean"],
            "collapse_rule": "score strictly below threshold",
            "tie_tolerance": 1e-12,
            "author_defined_minimum_worthwhile_difference": 3.0,
            "decision_gate": {
                "four_actor_procedure_supporting": "interval lower > 0 and point >= +3",
                "two_actor_procedure_supporting": "interval upper < 0 and point <= -3",
                "author_band_comparable": "full interval inside [-3,+3]",
                "unresolved": "all other outcomes",
            },
            "decision_precedence": [
                "four_actor_procedure_supporting",
                "two_actor_procedure_supporting",
                "author_band_comparable",
                "unresolved",
            ],
        },
        "runs": runs,
    }
    for run in manifest["runs"]:
        run["command"] = build_command(manifest, run)
    validate_manifest_commands(manifest)
    return manifest


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(
            f"refusing to replace frozen artifact or sidecar: {path}, {sidecar}"
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")
    return digest


def load_hashed_json(path: Path) -> tuple[dict[str, Any], str]:
    digest = file_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing SHA-256 sidecar: {sidecar}")
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    if recorded != digest:
        raise ValueError(f"SHA-256 sidecar mismatch for {path}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def verify_manifest_software(manifest: Mapping[str, Any]) -> None:
    """Verify the frozen run, source identity, and resolved Python environment."""
    validate_run_grid(list(manifest["runs"]))
    validate_manifest_commands(manifest)
    expected_source = manifest["source"]
    if expected_source.get("tracked_worktree_clean") is not True:
        raise RuntimeError("manifest was not frozen from a clean source checkout")
    source_identity = source_identity_snapshot()
    if (
        source_identity["git_revision"] != expected_source.get("git_revision")
        or source_identity["files"] != expected_source.get("files")
    ):
        raise RuntimeError("source revision or file hashes differ from frozen manifest")
    if dependency_snapshot() != manifest["dependencies"]:
        raise RuntimeError("resolved dependency snapshot differs from frozen manifest")


def verify_manifest_environment(manifest: Mapping[str, Any]) -> None:
    verify_manifest_software(manifest)
    if source_snapshot() != manifest["source"]:
        raise RuntimeError("launch requires the clean frozen source checkout")
    if hardware_snapshot() != manifest["hardware"]:
        raise RuntimeError("hardware inventory differs from frozen manifest")
    for environment, record in manifest["datasets"].items():
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"dataset hash drift for {environment}")
        if _normalization_record(path) != record["normalization"]:
            raise RuntimeError(f"normalization hash drift for {environment}")


def checkpoint_record(
    path: Path,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Hash and validate one trusted local checkpoint against its frozen run."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    expected_keys = set(manifest["output_contract"]["checkpoint_keys"])
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        unexpected = sorted(actual_keys.difference(expected_keys))
        raise ValueError(
            f"checkpoint key schema mismatch missing={missing} "
            f"unexpected={unexpected}: {path}"
        )
    if type(payload["step"]) is not int:
        raise ValueError(f"checkpoint step must be a Python integer: {path}")
    step = payload["step"]
    filename_step = path.stem.removeprefix("params_")
    if not filename_step.isdigit() or int(filename_step) != step:
        raise ValueError(f"checkpoint filename/payload step mismatch: {path}")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"checkpoint step {step} != {expected_step}: {path}")
    config = scientific_config(payload["config"])
    config_digest = json_sha256(config)
    if config_digest != run["config_sha256"]:
        raise ValueError(f"checkpoint config hash does not match frozen run: {path}")
    expected_actor_count = 4 if run["condition"] == "bar_p4" else 2
    if len(payload["actors_params"]) != expected_actor_count:
        raise ValueError(
            f"checkpoint actor count does not match frozen procedure: {path}"
        )
    if (
        len(payload["actors_steps"]) != expected_actor_count
        or len(payload["actors_opt_states"]) != expected_actor_count
    ):
        raise ValueError(
            f"checkpoint actor state count does not match frozen procedure: {path}"
        )
    critic_step = scalar_integer(payload["critic_step"], "critic_step")
    if critic_step != step:
        raise ValueError(f"checkpoint critic_step does not equal step: {path}")
    expected_actor_step = step // int(config["policy_freq"])
    actor_steps = [
        scalar_integer(actor_step, f"actors_steps[{index}]")
        for index, actor_step in enumerate(payload["actors_steps"])
    ]
    if actor_steps != [expected_actor_step] * expected_actor_count:
        raise ValueError(f"checkpoint actor steps do not match policy schedule: {path}")
    for key in (*WEIGHT_KEYS, "actors_opt_states", "critic_opt_state"):
        require_finite_tree(payload[key], key)
    for key in ("max_action", "policy_noise", "noise_clip"):
        value = float(payload[key])
        if not np.isfinite(value) or (key == "max_action" and value <= 0.0):
            raise ValueError(f"checkpoint {key} is not a valid finite scalar: {path}")
    weights = {key: payload[key] for key in WEIGHT_KEYS}
    weights_digest = hash_tree(weights)
    norm_digest = normalization_sha256(payload["mean"], payload["std"])
    expected_norm = manifest["datasets"][run["environment"]]["normalization"][
        "statistics_sha256"
    ]
    if norm_digest != expected_norm:
        raise ValueError(f"checkpoint normalization hash mismatch: {path}")
    return {
        "run_key": run["key"],
        "checkpoint": str(path.resolve()),
        "step": step,
        "file_sha256": file_sha256(path),
        "scientific_config_sha256": config_digest,
        "weights_sha256": weights_digest,
        "normalization_sha256": norm_digest,
    }


def inspect_checkpoint(path: Path, run: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict:
    record = checkpoint_record(path, run, manifest)
    if record["step"] <= 0 or record["step"] >= 1_000_000:
        raise ValueError(f"resume checkpoint step must be in (0,1000000): {path}")
    return record


def _final_eval_complete(
    run: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    out_dir = Path(run["output_dir"])
    config_path = out_dir / "config.json"
    eval_path = out_dir / "eval.csv"
    checkpoint = out_dir / "params_1000000.pkl"
    if not (config_path.is_file() and eval_path.is_file() and checkpoint.is_file()):
        return False
    checkpoint_record(checkpoint, run, manifest, expected_step=1_000_000)
    config = scientific_config(json.loads(config_path.read_text(encoding="utf-8")))
    if json_sha256(config) != run["config_sha256"]:
        raise ValueError(f"completed-run config drift: {run['key']}")
    schema_name = (
        "bar_eval_columns"
        if run["condition"] == "bar_p4"
        else "two_actor_eval_columns"
    )
    with eval_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != manifest["output_contract"][schema_name]:
            raise ValueError(f"completed-run eval schema drift: {run['key']}")
        rows = list(reader)
    final = [row for row in rows if int(float(row["step"])) == 1_000_000]
    if len(final) != 1 or run["score_column"] not in final[0]:
        raise ValueError(f"invalid final evaluation output: {run['key']}")
    score = float(final[0][run["score_column"]])
    if not np.isfinite(score):
        raise ValueError(f"non-finite final score: {run['key']}")
    return True


def _latest_partial_checkpoint(run: Mapping[str, Any]) -> Path | None:
    out_dir = Path(run["output_dir"])
    candidates = []
    for path in out_dir.glob("params_*.pkl") if out_dir.is_dir() else []:
        suffix = path.stem.removeprefix("params_")
        if suffix.isdigit() and 0 < int(suffix) < 1_000_000:
            candidates.append((int(suffix), path))
    return max(candidates)[1] if candidates else None


def prepare_jobs(
    manifest: Mapping[str, Any], resume_registry: Mapping[str, Any] | None
) -> tuple[list[LaunchJob], int]:
    registry_entries = [] if resume_registry is None else resume_registry["entries"]
    entry_keys = [entry["run_key"] for entry in registry_entries]
    expected_keys = {run["key"] for run in manifest["runs"]}
    if len(entry_keys) != len(set(entry_keys)):
        raise ValueError("resume registry contains duplicate run keys")
    if not set(entry_keys).issubset(expected_keys):
        raise ValueError("resume registry contains a run outside the frozen grid")
    entries = {entry["run_key"]: entry for entry in registry_entries}
    jobs = []
    completed = 0
    for run in manifest["runs"]:
        if _final_eval_complete(run, manifest):
            completed += 1
            continue
        checkpoint = _latest_partial_checkpoint(run)
        command = list(run["command"])
        if checkpoint is not None:
            if run["key"] not in entries:
                raise RuntimeError(f"unsealed checkpoint blocks resume: {checkpoint}")
            actual = inspect_checkpoint(checkpoint, run, manifest)
            if actual != entries[run["key"]]:
                raise RuntimeError(f"sealed checkpoint hashes changed: {checkpoint}")
            command.extend(["--restore-path", str(checkpoint.resolve())])
        elif Path(run["output_dir"]).exists() and any(Path(run["output_dir"]).iterdir()):
            raise RuntimeError(
                f"nonempty run directory has no sealed checkpoint: "
                f"{run['output_dir']}"
            )
        jobs.append(LaunchJob(run=run, command=command, resume_checkpoint=checkpoint))
    return jobs, completed


def _worker_environment(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return environment


def execute_jobs(manifest: Mapping[str, Any], jobs: list[LaunchJob]) -> int:
    execution = manifest["execution"]
    gpus = execution["gpus"]
    concurrency = len(gpus) * int(execution["slots_per_gpu"])
    log_dir = Path(manifest["runs"][0]["output_root"]).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, LaunchJob, object, int]] = {}
    next_job = 0
    failures = 0
    try:
        while next_job < len(jobs) or running:
            used_slots = {metadata[3] for metadata in running.values()}
            free_slots = [slot for slot in range(concurrency) if slot not in used_slots]
            while next_job < len(jobs) and free_slots:
                slot = free_slots.pop(0)
                job = jobs[next_job]
                next_job += 1
                gpu = gpus[slot // int(execution["slots_per_gpu"])]
                log_path = log_dir / f"{job.run['key'].replace('|', '__')}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                print(
                    f"\n[p0] start {job.run['key']} "
                    f"at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                    file=log_handle,
                    flush=True,
                )
                process = subprocess.Popen(
                    job.command,
                    cwd=ROOT,
                    env=_worker_environment(gpu),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running[process.pid] = (process, job, log_handle, slot)
                print(f"[launch] gpu={gpu} pid={process.pid} {job.run['key']}", flush=True)
            if running:
                time.sleep(1.0)
            for pid in [pid for pid, item in running.items() if item[0].poll() is not None]:
                process, job, log_handle, _slot = running.pop(pid)
                log_handle.close()
                if process.returncode:
                    failures += 1
                    print(f"[fail] rc={process.returncode} {job.run['key']}", flush=True)
                else:
                    print(f"[ok] {job.run['key']}", flush=True)
    except (KeyboardInterrupt, SystemExit):
        for process, _job, _handle, _slot in running.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process, _job, handle, _slot in running.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            handle.close()
        raise
    return 1 if failures else 0


def command_plan(args: argparse.Namespace) -> int:
    runs = build_runs(Path(args.output_root))
    print(json.dumps({
        "experiment": EXPERIMENT,
        "launches": False,
        "conditions": {
            condition: sum(run["condition"] == condition for run in runs)
            for condition in CONDITIONS
        },
        "total_runs": len(runs),
        "environments": list(ENVIRONMENTS),
        "T": list(T_VALUES),
        "seeds": list(SEEDS),
    }, indent=2))
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    freeze_python = Path(args.python).resolve()
    current_python = Path(sys.executable).resolve()
    if freeze_python != current_python:
        raise ValueError(
            "--python must be the interpreter running freeze so its resolved "
            "dependency snapshot is the launched environment"
        )
    datasets = dataset_records(Path(args.data_dir))
    manifest = build_manifest(
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_root),
        python=Path(args.python),
        gpus=args.gpus.replace(",", " ").split(),
        slots_per_gpu=args.slots_per_gpu,
        compilation_cache_dir=Path(args.compilation_cache_dir),
        datasets=datasets,
        dependencies=dependency_snapshot(),
        hardware=hardware_snapshot(),
        source=source_snapshot(),
    )
    digest = _write_create_only(Path(args.manifest), manifest)
    print(f"[frozen] {args.manifest} sha256={digest} runs=180")
    return 0


def command_seal_resume(args: argparse.Namespace) -> int:
    manifest, manifest_digest = load_hashed_json(Path(args.manifest))
    validate_run_grid(manifest["runs"])
    by_dir = {str(Path(run["output_dir"]).resolve()): run for run in manifest["runs"]}
    entries = []
    for checkpoint_text in args.checkpoint:
        checkpoint = Path(checkpoint_text).resolve()
        run = by_dir.get(str(checkpoint.parent))
        if run is None:
            raise ValueError(f"checkpoint is outside the frozen run grid: {checkpoint}")
        entries.append(inspect_checkpoint(checkpoint, run, manifest))
    if len({entry["run_key"] for entry in entries}) != len(entries):
        raise ValueError("resume registry contains duplicate run keys")
    registry = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "frozen_manifest_sha256": manifest_digest,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": sorted(entries, key=lambda entry: entry["run_key"]),
    }
    digest = _write_create_only(Path(args.output), registry)
    print(f"[sealed] {args.output} sha256={digest} checkpoints={len(entries)}")
    return 0


def command_launch(args: argparse.Namespace) -> int:
    manifest, manifest_digest = load_hashed_json(Path(args.manifest))
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("wrong experiment manifest")
    verify_manifest_environment(manifest)
    registry = None
    if args.resume_registry:
        registry, _ = load_hashed_json(Path(args.resume_registry))
        if registry.get("experiment") != EXPERIMENT:
            raise ValueError("resume registry targets a different experiment")
        if registry.get("frozen_manifest_sha256") != manifest_digest:
            raise ValueError("resume registry targets a different frozen manifest")
    jobs, completed = prepare_jobs(manifest, registry)
    print(f"[p0] completed={completed} pending={len(jobs)} execute={args.execute}")
    if not args.execute:
        for job in jobs:
            prefix = "resume" if job.resume_checkpoint else "fresh"
            print(f"[{prefix}] {shlex.join(job.command)}")
        return 0
    return execute_jobs(manifest, jobs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    plan = subparsers.add_parser("plan", help="print the fixed grid; never launch")
    plan.add_argument("--output-root", default="results/p0_bar_two_actor_p4")

    freeze = subparsers.add_parser("freeze", help="create the immutable launch manifest")
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--data-dir", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--python", default=sys.executable)
    freeze.add_argument("--gpus", default="0")
    freeze.add_argument("--slots-per-gpu", type=int, default=1)
    freeze.add_argument("--compilation-cache-dir", required=True)

    seal = subparsers.add_parser("seal-resume", help="seal exact trusted partial checkpoints")
    seal.add_argument("--manifest", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--checkpoint", action="append", required=True)

    launch = subparsers.add_parser("launch", help="verify locks and plan; --execute starts jobs")
    launch.add_argument("--manifest", required=True)
    launch.add_argument("--resume-registry")
    launch.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command is None:
        return parser.parse_args(["plan"])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {
        "plan": command_plan,
        "freeze": command_freeze,
        "seal-resume": command_seal_resume,
        "launch": command_launch,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
