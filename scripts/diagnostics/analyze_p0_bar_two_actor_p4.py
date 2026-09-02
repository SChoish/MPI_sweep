#!/usr/bin/env python3
"""Analyze the frozen BAR-P4 versus two-actor-P4 experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.run_p0_bar_two_actor_p4 import (  # noqa: E402
    ENVIRONMENTS,
    EXPERIMENT,
    SEEDS,
    T_VALUES,
    THRESHOLDS,
    checkpoint_record,
    file_sha256,
    json_sha256,
    load_hashed_json,
    scientific_config,
    validate_run_grid,
    verify_manifest_software,
)

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260902
AUTHOR_MWD = 3.0
TIE_TOLERANCE = 1e-12
DECISION_GATE = {
    "four_actor_procedure_supporting": "interval lower > 0 and point >= +3",
    "two_actor_procedure_supporting": "interval upper < 0 and point <= -3",
    "author_band_comparable": "full interval inside [-3,+3]",
    "unresolved": "all other outcomes",
}
OUTCOME_LABELS = (
    "four_actor_procedure_supporting",
    "two_actor_procedure_supporting",
    "author_band_comparable",
    "unresolved",
)


def _validate_analysis_contract(manifest: Mapping[str, Any]) -> None:
    contract = manifest.get("analysis_contract", {})
    bootstrap = contract.get("bootstrap", {})
    if (
        contract.get("primary_contrast")
        != "BAR-P4 deployment minus two-actor-P4 deployment"
        or contract.get("primary_estimator")
        != "mean of nine fixed task means (balanced 90-cell grid)"
        or bootstrap.get("resampling_unit") != "fixed task variant"
        or contract.get("paired_median_and_wins") is not True
        or bootstrap.get("draws") != BOOTSTRAP_DRAWS
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("interval") != "percentile 95%; numpy.quantile linear"
        or contract.get("inference_scope")
        != (
            "fixed 9-task x 5-T x 2-seed grid across three shared dynamics "
            "families; not broader offline-RL generalization"
        )
        or contract.get("collapse_thresholds") != list(THRESHOLDS)
        or contract.get("collapse_units") != ["raw_run", "two_seed_mean"]
        or contract.get("collapse_rule") != "score strictly below threshold"
        or contract.get("tie_tolerance") != TIE_TOLERANCE
        or contract.get("author_defined_minimum_worthwhile_difference")
        != AUTHOR_MWD
        or contract.get("decision_gate") != DECISION_GATE
        or tuple(contract.get("decision_precedence", ())) != OUTCOME_LABELS
    ):
        raise ValueError("frozen manifest analysis contract drift")


def task_bootstrap_interval(
    task_means: Mapping[str, float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if set(task_means) != set(ENVIRONMENTS):
        raise ValueError("bootstrap requires exactly the nine fixed task means")
    values = np.asarray([task_means[task] for task in ENVIRONMENTS], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def decision_label(point: float, low: float, high: float) -> str:
    """Apply the frozen author-defined +/-3 gate in its declared precedence."""
    if low > 0.0 and point >= AUTHOR_MWD:
        return "four_actor_procedure_supporting"
    if high < 0.0 and point <= -AUTHOR_MWD:
        return "two_actor_procedure_supporting"
    if low >= -AUTHOR_MWD and high <= AUTHOR_MWD:
        return "author_band_comparable"
    return "unresolved"


def transition_counts(pairs: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    counts = {
        "neither_collapsed": 0,
        "bar_only_collapsed": 0,
        "two_actor_only_collapsed": 0,
        "both_collapsed": 0,
    }
    for row in pairs:
        bar = float(row["bar_score"]) < threshold
        control = float(row["two_actor_score"]) < threshold
        key = (
            "both_collapsed"
            if bar and control
            else "bar_only_collapsed"
            if bar
            else "two_actor_only_collapsed"
            if control
            else "neither_collapsed"
        )
        counts[key] += 1
    counts["bar_collapsed"] = counts["bar_only_collapsed"] + counts["both_collapsed"]
    counts["two_actor_collapsed"] = (
        counts["two_actor_only_collapsed"] + counts["both_collapsed"]
    )
    counts["pairs"] = len(pairs)
    return counts


def _read_score(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[float, dict]:
    out_dir = Path(run["output_dir"])
    config_path = out_dir / "config.json"
    eval_path = out_dir / "eval.csv"
    checkpoint_path = out_dir / "params_1000000.pkl"
    for path in (config_path, eval_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing P0 artifact for {run['key']}: {path}")
    config = scientific_config(json.loads(config_path.read_text(encoding="utf-8")))
    if json_sha256(config) != run["config_sha256"]:
        raise ValueError(f"scientific config hash mismatch: {run['key']}")
    expected_columns = manifest["output_contract"][
        "bar_eval_columns" if run["condition"] == "bar_p4" else "two_actor_eval_columns"
    ]
    with eval_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"eval schema mismatch: {run['key']}")
        rows = list(reader)
    final_rows = [row for row in rows if int(float(row["step"])) == 1_000_000]
    if len(final_rows) != 1:
        raise ValueError(f"expected one final eval row: {run['key']}")
    score = float(final_rows[0][run["score_column"]])
    if not math.isfinite(score):
        raise ValueError(f"non-finite deployment score: {run['key']}")
    hashes = {
        "config": file_sha256(config_path),
        "eval": file_sha256(eval_path),
        "checkpoint": checkpoint_record(
            checkpoint_path, run, manifest, expected_step=1_000_000
        ),
    }
    return score, hashes


def collect_pairs(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    validate_run_grid(list(manifest["runs"]))
    verify_manifest_software(manifest)
    _validate_analysis_contract(manifest)
    scores: dict[tuple[str, int, int, str], float] = {}
    artifact_hashes = {}
    for run in manifest["runs"]:
        expected_hash = json_sha256(scientific_config(run["resolved_config"]))
        if expected_hash != run["config_sha256"]:
            raise ValueError(f"manifest run config hash mismatch: {run['key']}")
        score, hashes = _read_score(run, manifest)
        cell = (run["environment"], int(run["T"]), int(run["seed"]), run["condition"])
        if cell in scores:
            raise ValueError(f"duplicate manifest result cell: {cell}")
        scores[cell] = score
        artifact_hashes[run["key"]] = hashes
    pairs = []
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            for seed in SEEDS:
                bar = scores[(environment, tau, seed, "bar_p4")]
                control = scores[(environment, tau, seed, "two_actor_p4")]
                pairs.append(
                    {
                        "environment": environment,
                        "T": tau,
                        "seed": seed,
                        "bar_score": bar,
                        "two_actor_score": control,
                        "delta_bar_minus_two_actor": bar - control,
                    }
                )
    if len(pairs) != 90:
        raise AssertionError("paired grid must have exactly 90 cells")
    return pairs, artifact_hashes


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pairs) != 90:
        raise ValueError("summary requires the exact 90 paired cells")
    by_task: dict[str, list[float]] = defaultdict(list)
    deltas = []
    for row in pairs:
        delta = float(row["delta_bar_minus_two_actor"])
        deltas.append(delta)
        by_task[row["environment"]].append(delta)
    task_means = {
        task: float(np.mean(by_task[task], dtype=np.float64)) for task in ENVIRONMENTS
    }
    if any(len(by_task[task]) != 10 for task in ENVIRONMENTS):
        raise ValueError("each task must contribute exactly 5 T values x 2 seeds")
    point = float(np.mean(list(task_means.values()), dtype=np.float64))
    low, high = task_bootstrap_interval(task_means)
    wins = sum(delta > TIE_TOLERANCE for delta in deltas)
    losses = sum(delta < -TIE_TOLERANCE for delta in deltas)
    ties = len(deltas) - wins - losses

    seed_mean_pairs = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["environment"], int(row["T"]))].append(row)
    for environment in ENVIRONMENTS:
        for tau in T_VALUES:
            rows = grouped[(environment, tau)]
            if len(rows) != 2:
                raise ValueError("seed-mean collapse requires exactly two seeds")
            seed_mean_pairs.append(
                {
                    "bar_score": float(np.mean([float(row["bar_score"]) for row in rows])),
                    "two_actor_score": float(
                        np.mean([float(row["two_actor_score"]) for row in rows])
                    ),
                }
            )
    collapse = {
        "raw_run": {
            str(threshold): transition_counts(pairs, threshold)
            for threshold in THRESHOLDS
        },
        "two_seed_mean": {
            str(threshold): transition_counts(seed_mean_pairs, threshold)
            for threshold in THRESHOLDS
        },
    }
    label = decision_label(point, low, high)
    if label not in OUTCOME_LABELS:
        raise AssertionError("unknown outcome label")
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
                "bar_wins": wins,
                "two_actor_wins": losses,
                "ties": ties,
                "tie_tolerance": TIE_TOLERANCE,
            },
            "task_means": task_means,
            "task_bootstrap_95_interval": [low, high],
            "task_bootstrap_draws": BOOTSTRAP_DRAWS,
            "task_bootstrap_seed": BOOTSTRAP_SEED,
            "author_defined_minimum_worthwhile_difference": AUTHOR_MWD,
            "outcome_label": label,
            "outcome_scope": "full procedure; the continuous contrast is primary",
        },
        "collapse_transitions": collapse,
        "collapse_role": "secondary; cannot override the primary outcome label",
    }


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    pairs: list[dict[str, Any]],
    artifact_hashes: Mapping[str, Any],
    frozen_manifest_sha256: str,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"analysis output directory must be absent or empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_scores.csv"
    summary_path = output_dir / "SUMMARY.json"
    analysis_path = output_dir / "ANALYSIS_MANIFEST.json"
    checkpoint_path = output_dir / "FINAL_CHECKPOINTS.json"
    for path in (paired_path, summary_path, analysis_path, checkpoint_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite analysis artifact: {path}")
    fieldnames = [
        "environment",
        "T",
        "seed",
        "bar_score",
        "two_actor_score",
        "delta_bar_minus_two_actor",
    ]
    with paired_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in pairs:
            writer.writerow(
                {
                    **row,
                    "bar_score": format(float(row["bar_score"]), ".17g"),
                    "two_actor_score": format(float(row["two_actor_score"]), ".17g"),
                    "delta_bar_minus_two_actor": format(
                        float(row["delta_bar_minus_two_actor"]), ".17g"
                    ),
                }
            )
    checkpoint_records = {
        key: hashes["checkpoint"] for key, hashes in artifact_hashes.items()
    }
    _write_json_create_only(checkpoint_path, checkpoint_records)
    summary_payload = {
        **summary,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "raw_input_artifact_tree_sha256": json_sha256(artifact_hashes),
    }
    _write_json_create_only(summary_path, summary_payload)
    analysis_manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "raw_input_artifact_tree_sha256": json_sha256(artifact_hashes),
        "final_checkpoint_tree_sha256": json_sha256(checkpoint_records),
        "files": {
            "paired_scores.csv": file_sha256(paired_path),
            "SUMMARY.json": file_sha256(summary_path),
            "FINAL_CHECKPOINTS.json": file_sha256(checkpoint_path),
        },
    }
    _write_json_create_only(analysis_path, analysis_manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, manifest_digest = load_hashed_json(Path(args.manifest))
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("wrong experiment manifest")
    pairs, artifact_hashes = collect_pairs(manifest)
    summary = summarize(pairs)
    if args.output_dir:
        _write_outputs(
            Path(args.output_dir), summary, pairs, artifact_hashes, manifest_digest
        )
        print(f"[analysis] wrote {args.output_dir}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
