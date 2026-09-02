#!/usr/bin/env python3
"""Independently verify P0 raw outputs and analysis artifacts.

This artifact-only verifier deliberately shares no code with the analysis
program.  It rebuilds the exact grid, hashes, paired statistics, task
bootstrap, collapse tables, and author-defined decision gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

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
T_VALUES = (4, 7, 10, 14, 20)
SEEDS = (0, 1)
CONDITIONS = ("bar_p4", "two_actor_p4")
THRESHOLDS = (0, 10, 20, 30, 40)
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
SCIENTIFIC_KEYS = (
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
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260902
AUTHOR_MWD = 3.0
TIE_TOLERANCE = 1e-12
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
BAR_EVAL_COLUMNS = (
    "step",
    "return",
    "d4rl_score",
    "critic_loss",
    "actor_loss",
    "return_pi4",
    "d4rl_pi4",
    "final_actor_loss",
)
TWO_ACTOR_EVAL_COLUMNS = (
    "step",
    "return",
    "d4rl_score",
    "critic_loss",
    "actor_loss",
    "return_eval",
    "d4rl_eval",
    "final_actor_loss",
)
DECISION_GATE = {
    "four_actor_procedure_supporting": "interval lower > 0 and point >= +3",
    "two_actor_procedure_supporting": "interval upper < 0 and point <= -3",
    "author_band_comparable": "full interval inside [-3,+3]",
    "unresolved": "all other outcomes",
}
DECISION_PRECEDENCE = (
    "four_actor_procedure_supporting",
    "two_actor_procedure_supporting",
    "author_band_comparable",
    "unresolved",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_snapshot() -> dict[str, Any]:
    python_executable = Path(sys.executable).resolve()
    packages = sorted(
        {
            f"{dist.metadata.get('Name', dist.name).lower()}=={dist.version}"
            for dist in importlib.metadata.distributions()
        }
    )
    return {
        "python_executable": str(python_executable),
        "python_executable_sha256": _file_hash(python_executable),
        "python_version": platform.python_version(),
        "resolved_packages": packages,
        "resolved_packages_sha256": _json_hash(packages),
        "requirements_sha256": _file_hash(ROOT / "requirements.txt"),
    }


def _source_identity() -> dict[str, Any]:
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
            raise FileNotFoundError(f"missing frozen source file: {path}")
        hashes[relative] = _file_hash(path)
    return {"git_revision": revision, "files": hashes}


def _verify_frozen_software(manifest: Mapping[str, Any]) -> None:
    expected_source = manifest.get("source", {})
    if expected_source.get("tracked_worktree_clean") is not True:
        raise ValueError("manifest was not frozen from a clean source checkout")
    actual_source = _source_identity()
    if (
        actual_source["git_revision"] != expected_source.get("git_revision")
        or actual_source["files"] != expected_source.get("files")
    ):
        raise ValueError("current verifier source differs from frozen manifest")
    if _dependency_snapshot() != manifest.get("dependencies"):
        raise ValueError("current verifier dependencies differ from frozen manifest")


def _array_hash(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(_canonical(list(value.shape)))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _normalization_hash(mean: Any, std: Any) -> str:
    return _json_hash({"mean": _array_hash(mean), "std": _array_hash(std)})


def _hash_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def update(node: Any) -> None:
        if isinstance(node, Mapping):
            digest.update(b"M")
            digest.update(len(node).to_bytes(8, "big"))
            for key in sorted(node, key=lambda item: str(item)):
                key_bytes = _canonical([type(key).__name__, str(key)])
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
            digest.update(bytes.fromhex(_array_hash(node)))
        elif isinstance(node, (str, int, float, bool)) or node is None:
            encoded = _canonical(node)
            digest.update(b"P")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        else:
            raise TypeError(
                f"unsupported checkpoint value for hashing: {type(node)!r}"
            )

    update(value)
    return digest.hexdigest()


def _load_hashed(path: Path) -> tuple[dict[str, Any], str]:
    digest = _file_hash(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("frozen manifest SHA-256 sidecar mismatch")
    return json.loads(path.read_text(encoding="utf-8")), digest


def _key(condition: str, environment: str, tau: int, seed: int) -> str:
    return f"{condition}|{environment}|T={tau}|seed={seed}"


def _expected_config(
    condition: str, environment: str, tau: int, seed: int
) -> dict[str, Any]:
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


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")
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


def _scalar_integer(value: Any, label: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"checkpoint {label} must be one integer scalar")
    return int(array)


def _project_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if any(key not in config for key in SCIENTIFIC_KEYS):
        raise ValueError("result config is missing a scientific key")
    return {key: config[key] for key in SCIENTIFIC_KEYS}


def _transition(rows: list[dict[str, float]], threshold: float) -> dict[str, int]:
    counts = {
        "neither_collapsed": 0,
        "bar_only_collapsed": 0,
        "two_actor_only_collapsed": 0,
        "both_collapsed": 0,
    }
    for row in rows:
        bar = row["bar_score"] < threshold
        two = row["two_actor_score"] < threshold
        name = (
            "both_collapsed" if bar and two else
            "bar_only_collapsed" if bar else
            "two_actor_only_collapsed" if two else
            "neither_collapsed"
        )
        counts[name] += 1
    counts["bar_collapsed"] = counts["bar_only_collapsed"] + counts["both_collapsed"]
    counts["two_actor_collapsed"] = (
        counts["two_actor_only_collapsed"] + counts["both_collapsed"]
    )
    counts["pairs"] = len(rows)
    return counts


def _outcome(point: float, low: float, high: float) -> str:
    if low > 0.0 and point >= AUTHOR_MWD:
        return "four_actor_procedure_supporting"
    if high < 0.0 and point <= -AUTHOR_MWD:
        return "two_actor_procedure_supporting"
    if low >= -AUTHOR_MWD and high <= AUTHOR_MWD:
        return "author_band_comparable"
    return "unresolved"


def _read_raw(
    manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str, dict[str, dict[str, Any]]]:
    expected_keys = {
        _key(condition, environment, tau, seed)
        for condition in CONDITIONS
        for environment in ENVIRONMENTS
        for tau in T_VALUES
        for seed in SEEDS
    }
    runs = manifest.get("runs", [])
    actual_keys = [run.get("key") for run in runs]
    if (
        len(runs) != 180
        or len(set(actual_keys)) != 180
        or set(actual_keys) != expected_keys
    ):
        raise ValueError("frozen manifest does not contain the exact 180-key grid")
    condition_roots = {}
    for condition in CONDITIONS:
        condition_runs = [run for run in runs if run["condition"] == condition]
        if len(condition_runs) != 90:
            raise ValueError("frozen manifest must contain 90 runs per condition")
        roots = {run["output_root"] for run in condition_runs}
        if len(roots) != 1:
            raise ValueError("frozen manifest condition output-root drift")
        condition_root = Path(roots.pop())
        if not condition_root.is_absolute() or condition_root.name != condition:
            raise ValueError("frozen manifest condition output-root contract drift")
        condition_roots[condition] = condition_root
    if len({root.parent for root in condition_roots.values()}) != 1:
        raise ValueError("frozen manifest conditions do not share a result parent")
    output_contract = manifest.get("output_contract", {})
    if tuple(output_contract.get("checkpoint_keys", ())) != CHECKPOINT_KEYS:
        raise ValueError("frozen manifest checkpoint key schema is not the locked schema")
    if (
        tuple(output_contract.get("bar_eval_columns", ())) != BAR_EVAL_COLUMNS
        or tuple(output_contract.get("two_actor_eval_columns", ()))
        != TWO_ACTOR_EVAL_COLUMNS
    ):
        raise ValueError("frozen manifest deployment eval-column contract drift")
    evaluation_contract = manifest.get("evaluation_contract", {})
    expected_evaluation = {
        "critic_updates": 1_000_000,
        "final_evaluation_episodes": 10,
        "evaluation_frequency": 1_000_000,
        "target_score_column": "d4rl_score",
        "bar_deployment_score_column": "d4rl_pi4",
        "two_actor_deployment_score_column": "d4rl_eval",
    }
    if evaluation_contract != expected_evaluation:
        raise ValueError("frozen manifest evaluation contract drift")
    grid = manifest.get("grid", {})
    if (
        grid.get("environments") != list(ENVIRONMENTS)
        or grid.get("T") != list(T_VALUES)
        or grid.get("seeds") != list(SEEDS)
        or grid.get("conditions") != list(CONDITIONS)
        or grid.get("runs_per_condition") != 90
        or grid.get("total_runs") != 180
    ):
        raise ValueError("frozen manifest grid order/count drift")
    expected_procedure = {
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
            "isolation; two-actor policy-separation control (legacy mcep token; "
            "not published reproduction)"
        ),
    }
    if manifest.get("procedure_contract") != expected_procedure:
        raise ValueError("frozen manifest full-procedure contract drift")
    analysis_contract = manifest.get("analysis_contract", {})
    expected_bootstrap = {
        "resampling_unit": "fixed task variant",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "interval": "percentile 95%; numpy.quantile linear",
    }
    if (
        analysis_contract.get("primary_contrast")
        != "BAR-P4 deployment minus two-actor-P4 deployment"
        or analysis_contract.get("primary_estimator")
        != "mean of nine fixed task means (balanced 90-cell grid)"
        or analysis_contract.get("paired_median_and_wins") is not True
        or analysis_contract.get("bootstrap") != expected_bootstrap
        or analysis_contract.get("inference_scope")
        != (
            "fixed 9-task x 5-T x 2-seed grid across three shared dynamics "
            "families; not broader offline-RL generalization"
        )
        or analysis_contract.get("collapse_thresholds") != list(THRESHOLDS)
        or analysis_contract.get("collapse_units")
        != ["raw_run", "two_seed_mean"]
        or analysis_contract.get("collapse_rule")
        != "score strictly below threshold"
        or analysis_contract.get("author_defined_minimum_worthwhile_difference")
        != AUTHOR_MWD
        or analysis_contract.get("tie_tolerance") != TIE_TOLERANCE
        or analysis_contract.get("decision_gate") != DECISION_GATE
        or tuple(analysis_contract.get("decision_precedence", ()))
        != DECISION_PRECEDENCE
    ):
        raise ValueError("frozen manifest analysis contract drift")

    scores = {}
    artifact_hashes = {}
    for run in runs:
        condition = run["condition"]
        environment = run["environment"]
        if type(run["T"]) is not int or type(run["seed"]) is not int:
            raise ValueError(f"manifest run T/seed types drift: {run['key']}")
        tau = run["T"]
        seed = run["seed"]
        if (
            condition not in CONDITIONS
            or environment not in ENVIRONMENTS
            or tau not in T_VALUES
            or seed not in SEEDS
        ):
            raise ValueError(f"manifest run fields leave frozen grid: {run['key']}")
        expected_tag = (
            f"{environment}_tau{tau}_"
            f"{'mpi4' if condition == 'bar_p4' else 'mcep4'}_seed{seed}"
        )
        expected_score = "d4rl_pi4" if condition == "bar_p4" else "d4rl_eval"
        if run["key"] != _key(condition, environment, tau, seed):
            raise ValueError(f"manifest run key/fields mismatch: {run['key']}")
        if run["tag"] != expected_tag or run["score_column"] != expected_score:
            raise ValueError(f"manifest run tag/score mismatch: {run['key']}")
        if Path(run["output_dir"]) != Path(run["output_root"]) / expected_tag:
            raise ValueError(f"manifest run output path mismatch: {run['key']}")
        config_from_manifest = _project_config(run["resolved_config"])
        if config_from_manifest != _expected_config(condition, environment, tau, seed):
            raise ValueError(f"manifest scientific config values drift: {run['key']}")
        if _json_hash(config_from_manifest) != run["config_sha256"]:
            raise ValueError(f"tampered manifest config hash: {run['key']}")
        out_dir = Path(run["output_dir"])
        config_path = out_dir / "config.json"
        eval_path = out_dir / "eval.csv"
        checkpoint_path = out_dir / "params_1000000.pkl"
        if not all(path.is_file() for path in (config_path, eval_path, checkpoint_path)):
            raise FileNotFoundError(f"missing raw artifact: {run['key']}")
        actual_config = _project_config(json.loads(config_path.read_text(encoding="utf-8")))
        if _json_hash(actual_config) != run["config_sha256"]:
            raise ValueError(f"raw config does not match frozen hash: {run['key']}")
        schema_name = (
            "bar_eval_columns" if run["condition"] == "bar_p4"
            else "two_actor_eval_columns"
        )
        with eval_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != manifest["output_contract"][schema_name]:
                raise ValueError(f"raw eval schema mismatch: {run['key']}")
            eval_rows = list(reader)
        final = [row for row in eval_rows if int(float(row["step"])) == 1_000_000]
        if len(final) != 1:
            raise ValueError(f"raw eval must have one final row: {run['key']}")
        score = float(final[0][run["score_column"]])
        if not math.isfinite(score):
            raise ValueError(f"non-finite raw score: {run['key']}")
        cell = (run["environment"], int(run["T"]), int(run["seed"]), run["condition"])
        if cell in scores:
            raise ValueError(f"duplicate raw cell: {cell}")
        scores[cell] = score
        # The manifest limits pickle loading to these trusted local result paths.
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        actual_checkpoint_keys = set(payload)
        if actual_checkpoint_keys != set(CHECKPOINT_KEYS):
            missing = sorted(set(CHECKPOINT_KEYS).difference(actual_checkpoint_keys))
            unexpected = sorted(actual_checkpoint_keys.difference(CHECKPOINT_KEYS))
            raise ValueError(
                f"final checkpoint key schema mismatch missing={missing} "
                f"unexpected={unexpected}: {run['key']}"
            )
        if type(payload["step"]) is not int:
            raise ValueError(
                f"final checkpoint step must be an integer: {run['key']}"
            )
        if payload["step"] != 1_000_000:
            raise ValueError(f"final checkpoint step mismatch: {run['key']}")
        checkpoint_config = _project_config(payload["config"])
        checkpoint_config_hash = _json_hash(checkpoint_config)
        if checkpoint_config_hash != run["config_sha256"]:
            raise ValueError(f"final checkpoint config mismatch: {run['key']}")
        expected_actor_count = 4 if run["condition"] == "bar_p4" else 2
        if len(payload["actors_params"]) != expected_actor_count:
            raise ValueError(f"final checkpoint actor count mismatch: {run['key']}")
        if (
            len(payload["actors_steps"]) != expected_actor_count
            or len(payload["actors_opt_states"]) != expected_actor_count
        ):
            raise ValueError(
                f"final checkpoint actor state count mismatch: {run['key']}"
            )
        critic_step = _scalar_integer(payload["critic_step"], "critic_step")
        if critic_step != payload["step"]:
            raise ValueError(f"final checkpoint critic_step mismatch: {run['key']}")
        expected_actor_step = payload["step"] // int(
            checkpoint_config["policy_freq"]
        )
        actor_steps = [
            _scalar_integer(actor_step, f"actors_steps[{index}]")
            for index, actor_step in enumerate(payload["actors_steps"])
        ]
        if actor_steps != [expected_actor_step] * expected_actor_count:
            raise ValueError(f"final checkpoint actor-step mismatch: {run['key']}")
        for key in (*WEIGHT_KEYS, "actors_opt_states", "critic_opt_state"):
            _require_finite_tree(payload[key], key)
        for key in ("max_action", "policy_noise", "noise_clip"):
            value = float(payload[key])
            if not np.isfinite(value) or (key == "max_action" and value <= 0.0):
                raise ValueError(
                    f"final checkpoint {key} is not a valid finite scalar: "
                    f"{run['key']}"
                )
        norm_hash = _normalization_hash(payload["mean"], payload["std"])
        expected_norm = manifest["datasets"][run["environment"]]["normalization"][
            "statistics_sha256"
        ]
        if norm_hash != expected_norm:
            raise ValueError(f"final checkpoint normalization mismatch: {run['key']}")
        checkpoint_record = {
            "run_key": run["key"],
            "checkpoint": str(checkpoint_path.resolve()),
            "step": 1_000_000,
            "file_sha256": _file_hash(checkpoint_path),
            "scientific_config_sha256": checkpoint_config_hash,
            "weights_sha256": _hash_tree(
                {key: payload[key] for key in WEIGHT_KEYS}
            ),
            "normalization_sha256": norm_hash,
        }
        artifact_hashes[run["key"]] = {
            "config": _file_hash(config_path),
            "eval": _file_hash(eval_path),
            "checkpoint": checkpoint_record,
        }

    pairs = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            for seed in SEEDS:
                bar = scores[(environment, tau, seed, "bar_p4")]
                two = scores[(environment, tau, seed, "two_actor_p4")]
                pairs.append({
                    "environment": environment,
                    "T": tau,
                    "seed": seed,
                    "bar_score": bar,
                    "two_actor_score": two,
                    "delta_bar_minus_two_actor": bar - two,
                })
    if len(pairs) != 90:
        raise AssertionError("verifier failed to rebuild 90 pairs")
    checkpoint_records = {
        key: hashes["checkpoint"] for key, hashes in artifact_hashes.items()
    }
    return pairs, _json_hash(artifact_hashes), checkpoint_records


def _expected_summary(
    pairs: list[dict[str, Any]], manifest_digest: str, raw_hash: str
) -> dict[str, Any]:
    by_task = defaultdict(list)
    deltas = []
    for row in pairs:
        delta = float(row["delta_bar_minus_two_actor"])
        deltas.append(delta)
        by_task[row["environment"]].append(delta)
    if any(len(by_task[task]) != 10 for task in ENVIRONMENTS):
        raise ValueError("verifier did not resolve ten cells per task")
    task_means = {
        task: float(np.mean(by_task[task], dtype=np.float64)) for task in ENVIRONMENTS
    }
    point = float(np.mean(list(task_means.values()), dtype=np.float64))
    values = np.asarray([task_means[task] for task in ENVIRONMENTS], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 9, size=(BOOTSTRAP_DRAWS, 9))
    boot = values[indices].mean(axis=1)
    low, high = [float(value) for value in np.quantile(boot, [0.025, 0.975])]
    bar_wins = sum(delta > TIE_TOLERANCE for delta in deltas)
    two_wins = sum(delta < -TIE_TOLERANCE for delta in deltas)

    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["environment"], int(row["T"]))].append(row)
    seed_means = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            rows = grouped[(environment, tau)]
            if len(rows) != 2:
                raise ValueError("verifier seed mean does not have two rows")
            seed_means.append({
                "bar_score": float(np.mean([row["bar_score"] for row in rows])),
                "two_actor_score": float(np.mean([row["two_actor_score"] for row in rows])),
            })
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "resampling_unit": "nine fixed task variants",
        "inference_scope": (
            "fixed 9-task x 5-T x 2-seed grid; task variants share three "
            "dynamics families; not broader offline-RL generalization"
        ),
        "n_paired_runs": 90,
        "primary": {
            "contrast": "BAR-P4 deployment minus two-actor-P4 deployment",
            "task_equal_mean": point,
            "paired_median": float(np.median(np.asarray(deltas, dtype=np.float64))),
            "wins_ties": {
                "bar_wins": bar_wins,
                "two_actor_wins": two_wins,
                "ties": 90 - bar_wins - two_wins,
                "tie_tolerance": TIE_TOLERANCE,
            },
            "task_means": task_means,
            "task_bootstrap_95_interval": [low, high],
            "task_bootstrap_draws": BOOTSTRAP_DRAWS,
            "task_bootstrap_seed": BOOTSTRAP_SEED,
            "author_defined_minimum_worthwhile_difference": AUTHOR_MWD,
            "outcome_label": _outcome(point, low, high),
            "outcome_scope": "full procedure; the continuous contrast is primary",
        },
        "collapse_transitions": {
            "raw_run": {
                str(threshold): _transition(pairs, threshold)
                for threshold in THRESHOLDS
            },
            "two_seed_mean": {
                str(threshold): _transition(seed_means, threshold)
                for threshold in THRESHOLDS
            },
        },
        "collapse_role": "secondary; cannot override the primary outcome label",
        "frozen_manifest_sha256": manifest_digest,
        "raw_input_artifact_tree_sha256": raw_hash,
    }


def _assert_equal(actual: Any, expected: Any, path: str = "summary") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{path} keys differ from independent recomputation")
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} list shape differs")
        for index, value in enumerate(expected):
            _assert_equal(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{path} differs: {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValueError(f"{path} differs: {actual!r} != {expected!r}")


def _verify_paired_csv(path: Path, expected: list[dict[str, Any]]) -> None:
    fields = [
        "environment", "T", "seed", "bar_score", "two_actor_score",
        "delta_bar_minus_two_actor",
    ]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError("paired_scores.csv schema mismatch")
        rows = list(reader)
    if len(rows) != 90:
        raise ValueError("paired_scores.csv must have exactly 90 rows")
    for index, (row, wanted) in enumerate(zip(rows, expected, strict=True)):
        if (
            row["environment"] != wanted["environment"]
            or int(row["T"]) != wanted["T"]
            or int(row["seed"]) != wanted["seed"]
        ):
            raise ValueError(f"paired_scores.csv key mismatch at row {index}")
        for field in ("bar_score", "two_actor_score", "delta_bar_minus_two_actor"):
            if not math.isclose(float(row[field]), float(wanted[field]), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"paired_scores.csv {field} mismatch at row {index}")


def verify(manifest_path: Path, analysis_dir: Path) -> dict[str, Any]:
    manifest, manifest_digest = _load_hashed(manifest_path)
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("wrong P0 experiment manifest")
    _verify_frozen_software(manifest)
    pairs, raw_hash, checkpoint_records = _read_raw(manifest)
    analysis_manifest_path = analysis_dir / "ANALYSIS_MANIFEST.json"
    paired_path = analysis_dir / "paired_scores.csv"
    summary_path = analysis_dir / "SUMMARY.json"
    checkpoint_path = analysis_dir / "FINAL_CHECKPOINTS.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    if (
        analysis_manifest.get("schema_version") != 1
        or analysis_manifest.get("experiment") != EXPERIMENT
    ):
        raise ValueError("wrong analysis artifact schema or experiment")
    if analysis_manifest.get("frozen_manifest_sha256") != manifest_digest:
        raise ValueError("analysis artifacts reference a different frozen manifest")
    if analysis_manifest.get("raw_input_artifact_tree_sha256") != raw_hash:
        raise ValueError("raw result artifacts changed after analysis")
    checkpoint_tree_hash = _json_hash(checkpoint_records)
    if analysis_manifest.get("final_checkpoint_tree_sha256") != checkpoint_tree_hash:
        raise ValueError("final checkpoint fingerprint tree mismatch")
    expected_file_hashes = {
        "paired_scores.csv": _file_hash(paired_path),
        "SUMMARY.json": _file_hash(summary_path),
        "FINAL_CHECKPOINTS.json": _file_hash(checkpoint_path),
    }
    if analysis_manifest.get("files") != expected_file_hashes:
        raise ValueError("analysis artifact file hash mismatch")
    recorded_checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    _assert_equal(recorded_checkpoints, checkpoint_records, "final_checkpoints")
    _verify_paired_csv(paired_path, pairs)
    actual_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = _expected_summary(pairs, manifest_digest, raw_hash)
    _assert_equal(actual_summary, expected_summary)
    verifier_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "pass",
        "run_artifacts_verified": 180,
        "paired_cells_verified": 90,
        "frozen_manifest": str(manifest_path.resolve()),
        "frozen_manifest_sha256": manifest_digest,
        "analysis_dir": str(analysis_dir.resolve()),
        "analysis_manifest_sha256": _file_hash(analysis_manifest_path),
        "raw_input_artifact_tree_sha256": raw_hash,
        "final_checkpoint_tree_sha256": checkpoint_tree_hash,
        "verifier_source": str(verifier_path),
        "verifier_source_sha256": _file_hash(verifier_path),
        "outcome_label": expected_summary["primary"]["outcome_label"],
        "outcome_scope": expected_summary["primary"]["outcome_scope"],
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(
            f"refusing to replace verification receipt or sidecar: {path}, {sidecar}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")
    return digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument(
        "--receipt",
        help="create-only receipt path; defaults to <analysis-dir>/VERIFY.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    receipt_path = (
        Path(args.receipt) if args.receipt else analysis_dir / "VERIFY.json"
    )
    receipt = {
        **verify(Path(args.manifest), analysis_dir),
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    digest = _write_receipt(receipt_path, receipt)
    print(
        "PASS P0 BAR-P4/two-actor-P4 artifacts: "
        f"180 runs, 90 pairs, locked analysis; receipt={receipt_path} "
        f"sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
