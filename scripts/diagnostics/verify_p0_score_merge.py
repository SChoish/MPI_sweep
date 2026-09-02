#!/usr/bin/env python3
"""Read-only independent verifier for the compact cross-host P0 score merge.

An exit status of zero certifies compact arithmetic and integrity only.  The
verified archive is deliberately reported as scientifically NOT_ADMISSIBLE
under the frozen P0 contract because the two shards used different resolved
dependency stacks and five retained tail runs have nonfinite optimizer state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = (
    ROOT / "sweep_results" / "diagnostics" / "p0_bar_p4_vs_two_actor_p4"
)
DEFAULT_TAIL = (
    ROOT
    / "sweep_results"
    / "diagnostics"
    / "hosts"
    / "ext_csh"
    / "p0_manifest_tail90"
)

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
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260902
AUTHOR_MWD = 3.0
TIE_TOLERANCE = 1e-12
REASON_CODES = {
    "multiple_frozen_manifests",
    "resolved_dependency_snapshot_mismatch",
    "tail_raw_artifacts_unavailable",
    "five_nonfinite_optimizer_states",
}
DEPENDENCY_NAMES = ("jax", "jaxlib", "flax", "optax", "numpy", "gym")
OPTIMIZER_DIVERGED_KEYS = {
    "bar_p4|walker2d-expert-v2|T=14|seed=1",
    "two_actor_p4|walker2d-expert-v2|T=14|seed=1",
    "bar_p4|walker2d-expert-v2|T=20|seed=0",
    "two_actor_p4|walker2d-expert-v2|T=20|seed=0",
    "bar_p4|walker2d-expert-v2|T=20|seed=1",
}

HEAD_FIELDS = (
    "key",
    "condition",
    "environment",
    "T",
    "seed",
    "tag",
    "score_column",
    "deployment_d4rl",
    "target_d4rl",
    "return_deploy_col",
    "host_run_dir",
)
SOURCE_FIELDS = (
    "key",
    "condition",
    "environment",
    "T",
    "seed",
    "tag",
    "host_run_dir",
    "config_path",
    "config_bytes",
    "config_sha256",
    "eval_path",
    "eval_bytes",
    "eval_sha256",
    "checkpoint_path",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "score_column",
    "deployment_d4rl",
)
TAIL_FIELDS = (
    "manifest_index",
    "run_key",
    "condition",
    "environment",
    "T",
    "seed",
    "score_column",
    "score",
    "config_sha256",
    "eval_sha256",
    "checkpoint_sha256",
    "weights_sha256",
    "normalization_sha256",
)
MERGED_FIELDS = (
    "manifest_index",
    "key",
    "condition",
    "environment",
    "T",
    "seed",
    "score_column",
    "deployment_d4rl",
    "host_shard",
    "score_source",
    "config_sha256",
)
PAIRED_FIELDS = (
    "environment",
    "T",
    "seed",
    "bar_score",
    "two_actor_score",
    "delta_bar_minus_two_actor",
)
MERGE_HASHED_FILES = {"SUMMARY.json", "merged_final_scores.csv", "paired_scores.csv"}
TAIL_HASHED_FILES = {"CHECKPOINTS.json", "scores.csv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise AssertionError(f"CSV schema mismatch: {path}")
        rows = list(reader)
    return rows


def _close(actual: float, expected: float, *, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError(f"{label}: {actual} != {expected}")


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AssertionError(f"invalid SHA-256 field: {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AssertionError(f"invalid SHA-256 field: {label}") from exc
    return value


def _expected_runs() -> list[tuple[str, str, int, int, str]]:
    runs = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    key = f"{condition}|{environment}|T={tau}|seed={seed}"
                    runs.append((condition, environment, tau, seed, key))
    return runs


def _verify_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise AssertionError("frozen P0 identity mismatch")
    grid = manifest.get("grid")
    expected_grid = {
        "T": list(T_VALUES),
        "conditions": list(CONDITIONS),
        "environments": list(ENVIRONMENTS),
        "runs_per_condition": 90,
        "schedule": "cell-paired interleaving of BAR-P4 and two-actor-P4",
        "seeds": list(SEEDS),
        "total_runs": 180,
    }
    if grid != expected_grid:
        raise AssertionError("frozen P0 grid mismatch")
    expected = _expected_runs()
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != len(expected):
        raise AssertionError("frozen P0 run count mismatch")
    for index, (run, (condition, environment, tau, seed, key)) in enumerate(
        zip(runs, expected, strict=True), start=1
    ):
        if (
            run.get("key") != key
            or run.get("condition") != condition
            or run.get("environment") != environment
            or int(run.get("T", -1)) != tau
            or int(run.get("seed", -1)) != seed
            or run.get("score_column")
            != ("d4rl_pi4" if condition == "bar_p4" else "d4rl_eval")
        ):
            raise AssertionError(f"frozen P0 run-grid drift at index {index}")
        if canonical_sha256(run.get("resolved_config")) != run.get("config_sha256"):
            raise AssertionError(f"resolved-config hash mismatch: {key}")

    contract = manifest.get("analysis_contract", {})
    bootstrap = contract.get("bootstrap", {})
    if (
        contract.get("primary_contrast")
        != "BAR-P4 deployment minus two-actor-P4 deployment"
        or contract.get("primary_estimator")
        != "mean of nine fixed task means (balanced 90-cell grid)"
        or bootstrap
        != {
            "draws": BOOTSTRAP_DRAWS,
            "interval": "percentile 95%; numpy.quantile linear",
            "resampling_unit": "fixed task variant",
            "seed": BOOTSTRAP_SEED,
        }
        or contract.get("collapse_thresholds") != list(THRESHOLDS)
        or contract.get("collapse_units") != ["raw_run", "two_seed_mean"]
        or contract.get("collapse_rule") != "score strictly below threshold"
        or contract.get("tie_tolerance") != TIE_TOLERANCE
        or contract.get("author_defined_minimum_worthwhile_difference")
        != AUTHOR_MWD
    ):
        raise AssertionError("frozen P0 analysis contract drift")


def _package_versions(dependencies: Mapping[str, Any]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for record in dependencies.get("resolved_packages", []):
        if not isinstance(record, str) or "==" not in record:
            continue
        name, version = record.split("==", 1)
        normalized = name.lower().replace("_", "-")
        if normalized in DEPENDENCY_NAMES:
            previous = versions.get(normalized)
            if previous is not None and previous != version:
                raise AssertionError(f"ambiguous dependency version: {normalized}")
            versions[normalized] = version
    if set(versions) != set(DEPENDENCY_NAMES):
        raise AssertionError("required dependency versions are incomplete")
    return versions


def _dataset_identity(manifest: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    result = {}
    datasets = manifest.get("datasets", {})
    if set(datasets) != set(ENVIRONMENTS):
        raise AssertionError("frozen dataset grid mismatch")
    for environment in ENVIRONMENTS:
        record = datasets[environment]
        normalization = record.get("normalization", {})
        result[environment] = (
            record.get("bytes"),
            record.get("sha256"),
            normalization.get("statistics_sha256"),
            normalization.get("mean_sha256"),
            normalization.get("std_sha256"),
            normalization.get("training_rows"),
        )
    return result


def _transition_counts(rows: Iterable[Mapping[str, float]], threshold: float) -> dict[str, int]:
    counts = {
        "neither_collapsed": 0,
        "bar_only_collapsed": 0,
        "two_actor_only_collapsed": 0,
        "both_collapsed": 0,
    }
    total = 0
    for row in rows:
        total += 1
        bar = float(row["bar_score"]) < threshold
        control = float(row["two_actor_score"]) < threshold
        label = (
            "both_collapsed"
            if bar and control
            else "bar_only_collapsed"
            if bar
            else "two_actor_only_collapsed"
            if control
            else "neither_collapsed"
        )
        counts[label] += 1
    counts["bar_collapsed"] = counts["bar_only_collapsed"] + counts["both_collapsed"]
    counts["two_actor_collapsed"] = (
        counts["two_actor_only_collapsed"] + counts["both_collapsed"]
    )
    counts["pairs"] = total
    return counts


def _recompute_summary(pairs: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    by_task: dict[str, list[float]] = defaultdict(list)
    deltas = []
    for row in pairs:
        delta = float(row["delta_bar_minus_two_actor"])
        deltas.append(delta)
        by_task[str(row["environment"])].append(delta)
    if set(by_task) != set(ENVIRONMENTS) or any(
        len(by_task[environment]) != 10 for environment in ENVIRONMENTS
    ):
        raise AssertionError("paired rows do not form nine complete task blocks")
    task_means = {
        environment: float(np.mean(by_task[environment], dtype=np.float64))
        for environment in ENVIRONMENTS
    }
    values = np.asarray([task_means[environment] for environment in ENVIRONMENTS])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = (float(value) for value in np.quantile(estimates, [0.025, 0.975]))
    point = float(np.mean(values, dtype=np.float64))
    if low > 0.0 and point >= AUTHOR_MWD:
        outcome = "four_actor_procedure_supporting"
    elif high < 0.0 and point <= -AUTHOR_MWD:
        outcome = "two_actor_procedure_supporting"
    elif low >= -AUTHOR_MWD and high <= AUTHOR_MWD:
        outcome = "author_band_comparable"
    else:
        outcome = "unresolved"

    grouped: dict[tuple[str, int], list[Mapping[str, float]]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["environment"]), int(row["T"]))].append(row)
    seed_means = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            selected = grouped[(environment, tau)]
            if len(selected) != 2:
                raise AssertionError("two-seed collapse grid is incomplete")
            seed_means.append(
                {
                    "bar_score": float(np.mean([row["bar_score"] for row in selected])),
                    "two_actor_score": float(
                        np.mean([row["two_actor_score"] for row in selected])
                    ),
                }
            )
    return {
        "task_means": task_means,
        "task_equal_mean": point,
        "task_bootstrap_95_interval": [low, high],
        "paired_median": float(np.median(np.asarray(deltas, dtype=np.float64))),
        "wins_ties": {
            "bar_wins": sum(value > TIE_TOLERANCE for value in deltas),
            "two_actor_wins": sum(value < -TIE_TOLERANCE for value in deltas),
            "ties": sum(abs(value) <= TIE_TOLERANCE for value in deltas),
            "tie_tolerance": TIE_TOLERANCE,
        },
        "outcome_label": outcome,
        "collapse_transitions": {
            "raw_run": {
                str(threshold): _transition_counts(pairs, threshold)
                for threshold in THRESHOLDS
            },
            "two_seed_mean": {
                str(threshold): _transition_counts(seed_means, threshold)
                for threshold in THRESHOLDS
            },
        },
    }


def verify_archive(archive_dir: Path, tail_dir: Path | None = None) -> dict[str, Any]:
    archive = archive_dir.expanduser().resolve()
    tail = (
        tail_dir.expanduser().resolve()
        if tail_dir is not None
        else archive.parent / "hosts" / "ext_csh" / "p0_manifest_tail90"
    )
    merge = _read_json(archive / "MERGE_MANIFEST.json")
    summary = _read_json(archive / "SUMMARY.json")
    status = _read_json(archive / "STATUS.json")
    head_frozen = _read_json(archive / "FROZEN_MANIFEST.json")
    tail_frozen = _read_json(tail / "FROZEN_MANIFEST.json")
    tail_manifest = _read_json(tail / "MANIFEST.json")
    tail_receipt = _read_json(tail / "VERIFY.json")
    tail_checkpoints = _read_json(tail / "CHECKPOINTS.json")
    _verify_frozen_manifest(head_frozen)
    _verify_frozen_manifest(tail_frozen)

    if merge.get("schema_version") != 1 or merge.get("experiment") != EXPERIMENT:
        raise AssertionError("P0 merge identity mismatch")
    merge_files = merge.get("files", {})
    if set(merge_files) != MERGE_HASHED_FILES:
        raise AssertionError("P0 compact hashed-file inventory mismatch")
    for relative, digest in merge_files.items():
        _sha(digest, label=f"merge file {relative}")
        if sha256_file(archive / relative) != digest:
            raise AssertionError(f"P0 compact file hash mismatch: {relative}")

    head_hash = sha256_file(archive / "FROZEN_MANIFEST.json")
    checksum_text = (archive / "FROZEN_MANIFEST.json.sha256").read_text().split()
    if checksum_text != [head_hash, "FROZEN_MANIFEST.json"]:
        raise AssertionError("head frozen-manifest sidecar mismatch")
    if (
        merge.get("head", {}).get("frozen_manifest_sha256") != head_hash
        or merge.get("head", {}).get("frozen_manifest") != "FROZEN_MANIFEST.json"
        or merge.get("head", {}).get("scores_sha256")
        != sha256_file(archive / "first90_final_scores.csv")
        or merge.get("head", {}).get("scores") != "first90_final_scores.csv"
        or merge.get("head", {}).get("manifest_index_range_one_based") != [1, 90]
        or merge.get("head", {}).get("host") != "ext_csv"
    ):
        raise AssertionError("P0 head-shard binding mismatch")

    tail_hash = sha256_file(tail / "FROZEN_MANIFEST.json")
    tail_manifest_hash = sha256_file(tail / "MANIFEST.json")
    tail_score_hash = sha256_file(tail / "scores.csv")
    tail_meta = merge.get("tail", {})
    if (
        tail_meta.get("frozen_manifest_sha256") != tail_hash
        or tail_meta.get("host_shard_manifest_sha256") != tail_manifest_hash
        or tail_meta.get("scores_sha256") != tail_score_hash
        or tail_meta.get("manifest_index_range_one_based") != [91, 180]
        or tail_meta.get("host") != "ext_csh"
        or tail_meta.get("host_shard_manifest")
        != "hosts/ext_csh/p0_manifest_tail90/MANIFEST.json"
        or tail_meta.get("scores") != "hosts/ext_csh/p0_manifest_tail90/scores.csv"
    ):
        raise AssertionError("P0 tail-shard binding mismatch")
    tail_files = tail_manifest.get("files", {})
    if set(tail_files) != TAIL_HASHED_FILES:
        raise AssertionError("P0 tail hashed-file inventory mismatch")
    for relative, digest in tail_files.items():
        _sha(digest, label=f"tail file {relative}")
        if sha256_file(tail / relative) != digest:
            raise AssertionError(f"P0 tail file hash mismatch: {relative}")
    if (
        tail_manifest.get("artifact") != "p0_manifest_tail90_host_shard"
        or tail_manifest.get("status") != "complete"
        or tail_manifest.get("runs") != 90
        or tail_manifest.get("conditions") != {"bar_p4": 45, "two_actor_p4": 45}
        or tail_manifest.get("manifest_index_range_one_based") != [91, 180]
        or tail_manifest.get("frozen_manifest_sha256") != tail_hash
        or tail_manifest.get("host") != "ext_csh"
        or tail_manifest.get("source_revision")
        != head_frozen.get("source", {}).get("git_revision")
        or tail_manifest.get("source_revision")
        != tail_frozen.get("source", {}).get("git_revision")
        or tail_receipt.get("pass") is not True
        or tail_receipt.get("runs") != 90
        or tail_receipt.get("optimizer_diverged_runs") != 5
        or tail_receipt.get("manifest_sha256") != tail_manifest_hash
        or tail_meta.get("verify") != tail_receipt
    ):
        raise AssertionError("P0 tail verification receipt mismatch")

    expected = _expected_runs()
    head_runs = head_frozen["runs"]
    tail_runs = tail_frozen["runs"]
    if [run["key"] for run in head_runs] != [run["key"] for run in tail_runs]:
        raise AssertionError("head/tail frozen run keys differ")
    if any(
        head_run["resolved_config"] != tail_run["resolved_config"]
        or head_run["config_sha256"] != tail_run["config_sha256"]
        for head_run, tail_run in zip(head_runs, tail_runs, strict=True)
    ):
        raise AssertionError("head/tail scientific configs differ")
    if any(path.suffix == ".pkl" for root in (archive, tail) for path in root.rglob("*")):
        raise AssertionError("compact P0 archive unexpectedly contains raw checkpoints")

    head_rows = _read_csv(archive / "first90_final_scores.csv", HEAD_FIELDS)
    source_rows = _read_csv(archive / "SOURCE_MANIFEST.csv", SOURCE_FIELDS)
    tail_rows = _read_csv(tail / "scores.csv", TAIL_FIELDS)
    merged_rows = _read_csv(archive / "merged_final_scores.csv", MERGED_FIELDS)
    paired_rows = _read_csv(archive / "paired_scores.csv", PAIRED_FIELDS)
    if not (
        len(head_rows) == len(source_rows) == len(tail_rows) == 90
        and len(merged_rows) == 180
        and len(paired_rows) == 90
    ):
        raise AssertionError("P0 compact row counts are incomplete")

    for offset, (head_row, source_row) in enumerate(
        zip(head_rows, source_rows, strict=True)
    ):
        condition, environment, tau, seed, key = expected[offset]
        if (
            head_row["key"] != key
            or source_row["key"] != key
            or head_row["condition"] != condition
            or source_row["condition"] != condition
            or head_row["environment"] != environment
            or source_row["environment"] != environment
            or int(head_row["T"]) != tau
            or int(source_row["T"]) != tau
            or int(head_row["seed"]) != seed
            or int(source_row["seed"]) != seed
            or head_row["score_column"] != head_runs[offset]["score_column"]
            or source_row["score_column"] != head_runs[offset]["score_column"]
            or head_row["tag"] != head_runs[offset]["tag"]
            or source_row["tag"] != head_runs[offset]["tag"]
            or not head_row["host_run_dir"].startswith("/home/ext_csv/")
            or source_row["host_run_dir"] != head_row["host_run_dir"]
            or not source_row["config_path"].startswith(head_row["host_run_dir"] + "/")
            or not source_row["eval_path"].startswith(head_row["host_run_dir"] + "/")
            or not source_row["checkpoint_path"].startswith(
                head_row["host_run_dir"] + "/"
            )
        ):
            raise AssertionError(f"P0 head row mismatch: {key}")
        _close(
            float(head_row["deployment_d4rl"]),
            float(source_row["deployment_d4rl"]),
            label=f"head score {key}",
        )
        if not math.isfinite(float(head_row["target_d4rl"])):
            raise AssertionError(f"nonfinite head target score: {key}")
        for field in ("config_sha256", "eval_sha256", "checkpoint_sha256"):
            _sha(source_row[field], label=f"source {field} {key}")
        for field in ("config_bytes", "eval_bytes", "checkpoint_bytes"):
            if int(source_row[field]) <= 0:
                raise AssertionError(f"nonpositive source size: {key}/{field}")

    tail_keys = {key for *_, key in expected[90:]}
    if set(tail_checkpoints) != tail_keys:
        raise AssertionError("tail checkpoint-key set is not exactly indices 91--180")
    for offset, row in enumerate(tail_rows, start=90):
        condition, environment, tau, seed, key = expected[offset]
        checkpoint = tail_checkpoints[key]
        if (
            int(row["manifest_index"]) != offset + 1
            or row["run_key"] != key
            or row["condition"] != condition
            or row["environment"] != environment
            or int(row["T"]) != tau
            or int(row["seed"]) != seed
            or row["score_column"] != tail_runs[offset]["score_column"]
            or checkpoint.get("run_key") != key
            or checkpoint.get("step") != 1_000_000
            or row["checkpoint_sha256"] != checkpoint.get("file_sha256")
            or row["weights_sha256"] != checkpoint.get("weights_sha256")
            or row["normalization_sha256"]
            != checkpoint.get("normalization_sha256")
            or not str(checkpoint.get("checkpoint", "")).startswith("/home/ext_csh/")
            or not math.isfinite(float(row["score"]))
        ):
            raise AssertionError(f"P0 tail row mismatch: {key}")
        for field in (
            "config_sha256",
            "eval_sha256",
            "checkpoint_sha256",
            "weights_sha256",
            "normalization_sha256",
        ):
            _sha(row[field], label=f"tail {field} {key}")

    scores: dict[tuple[str, str, int, int], float] = {}
    for offset, row in enumerate(merged_rows):
        condition, environment, tau, seed, key = expected[offset]
        source = head_rows[offset] if offset < 90 else tail_rows[offset - 90]
        source_score = source["deployment_d4rl"] if offset < 90 else source["score"]
        expected_host = "ext_csv" if offset < 90 else "ext_csh"
        expected_source = (
            "first90_final_scores.csv"
            if offset < 90
            else "hosts/ext_csh/p0_manifest_tail90/scores.csv"
        )
        if (
            int(row["manifest_index"]) != offset + 1
            or row["key"] != key
            or row["condition"] != condition
            or row["environment"] != environment
            or int(row["T"]) != tau
            or int(row["seed"]) != seed
            or row["score_column"] != head_runs[offset]["score_column"]
            or row["host_shard"] != expected_host
            or row["score_source"] != expected_source
            or row["config_sha256"] != head_runs[offset]["config_sha256"]
        ):
            raise AssertionError(f"P0 merged row mismatch: {key}")
        score = float(row["deployment_d4rl"])
        _close(score, float(source_score), label=f"merged score {key}")
        cell = (condition, environment, tau, seed)
        if cell in scores:
            raise AssertionError(f"duplicate merged P0 cell: {cell}")
        scores[cell] = score
    if len(scores) != 180:
        raise AssertionError("merged P0 union is not exactly 180 keys")

    rebuilt_pairs: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            for seed in SEEDS:
                bar = scores[("bar_p4", environment, tau, seed)]
                control = scores[("two_actor_p4", environment, tau, seed)]
                rebuilt_pairs.append(
                    {
                        "environment": environment,
                        "T": tau,
                        "seed": seed,
                        "bar_score": bar,
                        "two_actor_score": control,
                        "delta_bar_minus_two_actor": bar - control,
                    }
                )
    for row, rebuilt in zip(paired_rows, rebuilt_pairs, strict=True):
        if (
            row["environment"] != rebuilt["environment"]
            or int(row["T"]) != rebuilt["T"]
            or int(row["seed"]) != rebuilt["seed"]
        ):
            raise AssertionError("P0 paired row grid/order mismatch")
        for field in ("bar_score", "two_actor_score", "delta_bar_minus_two_actor"):
            _close(float(row[field]), rebuilt[field], label=f"paired {field}")
    # first90_paired_scores.csv has a historical schema; validate it separately.
    with (archive / "first90_paired_scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        expected_head_fields = (
            "environment",
            "T",
            "seed",
            "bar_p4_deployment_d4rl",
            "two_actor_p4_deployment_d4rl",
            "delta_bar_minus_control",
        )
        if tuple(reader.fieldnames or ()) != expected_head_fields:
            raise AssertionError("head-only paired CSV schema mismatch")
        historical_head_pairs = list(reader)
    if len(historical_head_pairs) != 45:
        raise AssertionError("head-only paired CSV count mismatch")
    rebuilt_head = {
        (row["environment"], int(row["T"]), int(row["seed"])): row
        for row in rebuilt_pairs[:45]
    }
    historical_head = {
        (row["environment"], int(row["T"]), int(row["seed"])): row
        for row in historical_head_pairs
    }
    if set(historical_head) != set(rebuilt_head):
        raise AssertionError("head-only paired cell set mismatch")
    for cell, row in historical_head.items():
        rebuilt = rebuilt_head[cell]
        _close(float(row["bar_p4_deployment_d4rl"]), rebuilt["bar_score"], label="head bar")
        _close(
            float(row["two_actor_p4_deployment_d4rl"]),
            rebuilt["two_actor_score"],
            label="head control",
        )
        _close(
            float(row["delta_bar_minus_control"]),
            rebuilt["delta_bar_minus_two_actor"],
            label="head delta",
        )

    recomputed = _recompute_summary(rebuilt_pairs)
    primary = summary.get("primary", {})
    for environment, expected_mean in recomputed["task_means"].items():
        _close(
            float(primary.get("task_means", {}).get(environment)),
            expected_mean,
            label=f"task mean {environment}",
        )
    for field in ("task_equal_mean", "paired_median"):
        _close(float(primary.get(field)), recomputed[field], label=field)
    for actual, expected_value, label in zip(
        primary.get("task_bootstrap_95_interval", []),
        recomputed["task_bootstrap_95_interval"],
        ("bootstrap low", "bootstrap high"),
        strict=True,
    ):
        _close(float(actual), expected_value, label=label)
    if (
        primary.get("task_bootstrap_draws") != BOOTSTRAP_DRAWS
        or primary.get("task_bootstrap_seed") != BOOTSTRAP_SEED
        or primary.get("author_defined_minimum_worthwhile_difference") != AUTHOR_MWD
        or primary.get("wins_ties") != recomputed["wins_ties"]
        or primary.get("outcome_label") != recomputed["outcome_label"]
        or summary.get("collapse_transitions")
        != recomputed["collapse_transitions"]
        or summary.get("n_paired_runs") != 90
    ):
        raise AssertionError("P0 summary arithmetic or decision label mismatch")

    diverged_sets = (
        set(tail_meta.get("optimizer_diverged_runs", [])),
        set(tail_manifest.get("optimizer_diverged_runs", [])),
        set(summary.get("merge", {}).get("optimizer_diverged_tail_runs", [])),
    )
    if any(value != OPTIMIZER_DIVERGED_KEYS for value in diverged_sets):
        raise AssertionError("the five retained optimizer-diverged keys changed")
    if not OPTIMIZER_DIVERGED_KEYS.issubset(tail_keys):
        raise AssertionError("optimizer-diverged keys are not all in the tail shard")

    head_dependencies = head_frozen.get("dependencies", {})
    tail_dependencies = tail_frozen.get("dependencies", {})
    head_versions = _package_versions(head_dependencies)
    tail_versions = _package_versions(tail_dependencies)
    computed_mismatches = {
        "python": [
            head_dependencies.get("python_version"),
            tail_dependencies.get("python_version"),
        ],
        **{
            name: [head_versions[name], tail_versions[name]]
            for name in DEPENDENCY_NAMES
            if head_versions[name] != tail_versions[name]
        },
    }
    identity = merge.get("scientific_identity", {})
    actual_identity = {
        "dataset_normalization_equal": _dataset_identity(head_frozen)
        == _dataset_identity(tail_frozen),
        "requirements_sha256_equal": head_dependencies.get("requirements_sha256")
        == tail_dependencies.get("requirements_sha256"),
        "resolved_dependency_snapshot_equal": head_dependencies.get(
            "resolved_packages_sha256"
        )
        == tail_dependencies.get("resolved_packages_sha256"),
        "resolved_config_and_config_sha256_equal": True,
        "run_keys_equal": True,
        "source_file_hashes_equal": head_frozen.get("source", {}).get("files")
        == tail_frozen.get("source", {}).get("files"),
        "host_paths_python_hardware_differ": (
            head_dependencies.get("python_executable")
            != tail_dependencies.get("python_executable")
            and head_frozen.get("hardware") != tail_frozen.get("hardware")
            and head_frozen.get("dataset_root") != tail_frozen.get("dataset_root")
        ),
    }
    for field, actual in actual_identity.items():
        if identity.get(field) is not actual:
            raise AssertionError(f"P0 scientific-identity flag mismatch: {field}")
    if identity.get("resolved_dependency_mismatches") != computed_mismatches:
        raise AssertionError("P0 resolved dependency mismatch table is inaccurate")
    if computed_mismatches.get("python") == [None, None] or len(computed_mismatches) < 2:
        raise AssertionError("P0 dependency mismatch unexpectedly disappeared")

    summary_merge = summary.get("merge", {})
    if (
        merge.get("merge_kind") != "cross_host_score_level"
        or merge.get("official_ckpt_bound_analyze_verify") is not False
        or merge.get("compact_integrity_pass") is not True
        or merge.get("scientific_admissible_under_frozen_p0_contract") is not False
        or merge.get("manuscript_primary_promotion_allowed") is not False
        or set(merge.get("reason_codes", [])) != REASON_CODES
        or summary_merge.get("official_ckpt_bound_analyze_verify") is not False
        or summary_merge.get("compact_integrity_pass") is not True
        or summary_merge.get("scientific_admissible_under_frozen_p0_contract")
        is not False
        or summary_merge.get("manuscript_primary_promotion_allowed") is not False
        or set(summary_merge.get("reason_codes", [])) != REASON_CODES
        or status.get("compact_integrity_pass") is not True
        or status.get("scientific_admissible_under_frozen_p0_contract") is not False
        or status.get("manuscript_primary_promotion_allowed") is not False
        or status.get("completed_finals") != 180
        or status.get("completed_by_condition") != {"bar_p4": 90, "two_actor_p4": 90}
        or status.get("paired_cells_in_archive") != 90
        or status.get("frozen_manifest_sha256") != head_hash
        or summary.get("head_frozen_manifest_sha256") != head_hash
        or summary.get("tail_frozen_manifest_sha256") != tail_hash
        or status.get("decision_label") != recomputed["outcome_label"]
        or recomputed["outcome_label"] != "unresolved"
    ):
        raise AssertionError("P0 integrity/admissibility semantics drift")
    _close(
        float(status.get("outcome", {}).get("task_equal_mean")),
        recomputed["task_equal_mean"],
        label="status task-equal mean",
    )
    status_interval = status.get("outcome", {}).get("task_bootstrap_95_interval", [])
    if len(status_interval) != 2:
        raise AssertionError("P0 status interval missing")
    for actual, expected_value in zip(
        status_interval, recomputed["task_bootstrap_95_interval"], strict=True
    ):
        _close(float(actual), expected_value, label="status bootstrap interval")
    if status.get("outcome", {}).get("outcome_label") != "unresolved":
        raise AssertionError("P0 status outcome mismatch")

    return {
        "experiment": EXPERIMENT,
        "compact_integrity": "PASS",
        "scientific_admissibility": "NOT_ADMISSIBLE",
        "manuscript_primary_promotion_allowed": False,
        "runs_verified": 180,
        "pairs_verified": 90,
        "optimizer_diverged_runs_retained": 5,
        "reason_codes": sorted(REASON_CODES),
        "outcome_label": recomputed["outcome_label"],
        "task_equal_mean": recomputed["task_equal_mean"],
        "task_bootstrap_95_interval": recomputed["task_bootstrap_95_interval"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--tail-dir", type=Path, default=DEFAULT_TAIL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_archive(args.archive_dir, args.tail_dir)
    except Exception as exc:
        failure = {
            "compact_integrity": "FAIL",
            "scientific_admissibility": "NOT_ADMISSIBLE",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
