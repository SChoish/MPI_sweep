#!/usr/bin/env python3
"""Verify released MPI results and the manuscript-facing numerical claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

GRID = (0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0)
METHODS = {
    "td3": ("K=1", "Imp"),
    "prox2": ("K=2", "Imp"),
    "prox3": ("K=3", "Imp"),
    "lin2": ("K=2", "Exp"),
    "lin3": ("K=3", "Exp"),
}
EXPECTED = {
    "td3": (25.9708, 35, 68),
    "prox2": (42.0213, 25, 53),
    "prox3": (52.7833, 17, 41),
    "lin2": (36.6838, 29, 58),
    "lin3": (43.9113, 24, 53),
}


def close(actual: float, expected: float, tol: float = 5e-4) -> None:
    if not math.isclose(actual, expected, abs_tol=tol):
        raise AssertionError(f"{actual} != {expected} within {tol}")


def load_sweep(root: Path, hops: str, integrator: str) -> tuple[list[str], dict[tuple[float, int, str], float]]:
    values: dict[tuple[float, int, str], float] = {}
    environments: list[str] = []
    for seed in (0, 1):
        path = root / hops / integrator / f"seed{seed}.csv"
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle)
            environments = [name for name in rows.fieldnames or [] if name != "tau"]
            for row in rows:
                tau = float(row["tau"])
                if tau not in GRID:
                    continue
                for env in environments:
                    if row[env] not in ("", "—"):
                        values[(tau, seed, env)] = float(row[env])
    expected_cells = len(GRID) * 2 * len(environments)
    if len(values) != expected_cells:
        raise AssertionError(f"{hops}/{integrator}: {len(values)} values, expected {expected_cells}")
    return environments, values


def verify_complete_grid(root: Path) -> None:
    for method, (hops, integrator) in METHODS.items():
        envs, values = load_sweep(root, hops, integrator)
        high = [score for (tau, _, _), score in values.items() if tau >= 4]
        cells = []
        for tau in GRID:
            if tau < 4:
                continue
            for env in envs:
                cells.append(statistics.mean(values[(tau, seed, env)] for seed in (0, 1)))
        high_mean = statistics.mean(high)
        cell_collapses = sum(score < 20 for score in cells)
        raw_collapses = sum(score < 20 for score in high)
        exp_mean, exp_cells, exp_raw = EXPECTED[method]
        close(high_mean, exp_mean)
        assert (cell_collapses, raw_collapses) == (exp_cells, exp_raw)
        print(f"PASS complete-grid {method}: high={high_mean:.4f}, collapse={cell_collapses}/63 and {raw_collapses}/126")


def verify_controls(diag: Path) -> None:
    rows = list(csv.DictReader((diag / "control_final_scores.csv").open(newline="")))
    summary = list(csv.DictReader((diag / "control_summary.csv").open(newline="")))
    for expected in summary:
        group = [
            float(row["d4rl_score"])
            for row in rows
            if row["experiment"] == expected["experiment"] and row["condition"] == expected["condition"]
        ]
        assert len(group) == int(expected["n_runs"])
        close(statistics.mean(group), float(expected["mean_d4rl_score"]), 1e-6)
        close(statistics.median(group), float(expected["median_d4rl_score"]), 1e-6)
        assert sum(score < 20 for score in group) == int(expected["collapse_below_20"])
    print("PASS targeted controls: run-level scores reproduce control_summary.csv")


def verify_diagnostics(diag: Path) -> None:
    target = json.loads((diag / "target_policy_exposure" / "SUMMARY.json").read_text())
    pair = target["paired_method_comparisons"]["mpi3_vs_td3"]
    assert target["completed_cells"] == 270
    assert pair["target_deterministic"]["left_smaller_cells"] == 88
    close(pair["target_deterministic"]["median_left_over_right_nonzero"], 0.6576946525, 1e-9)
    close(target["proxy_alignment"]["all_cells"]["pearson_proxy_vs_target_deterministic"], 0.9999114535, 1e-9)

    rows = list(csv.DictReader((diag / "matched_geometry" / "td3_prox3_run_diagnostics.csv").open(newline="")))
    paired: dict[tuple[str, float, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        paired.setdefault((row["env"], float(row["tau"]), int(row["seed"])), {})[row["method"]] = row
    final_lower = sum(
        float(methods["mpi3"]["final_displacement_sq_mean_metric_mean"])
        < float(methods["td3"]["final_displacement_sq_mean_metric_mean"])
        for methods in paired.values()
    )
    assert len(paired) == 90 and final_lower == 36

    route = json.loads((diag / "route_shadow" / "SUMMARY.json").read_text())
    assert route["n_runs"] == 8 and route["positive_delta_runs"] == 8
    close(route["mean_delta"], 0.1235147225, 1e-10)
    close(route["two_sided_exact_sign_flip_p"], 0.0078125, 1e-12)

    frozen = json.loads((diag / "frozen_critic_small_step" / "SUMMARY.json").read_text())
    eligible = [run for run in frozen["runs"] if run["local_points_eligible"] > 0]
    assert frozen["n_runs"] == 18 and len(eligible) == 15
    close(max(run["local_max_relative_difference"] for run in eligible), 0.0268159275, 1e-10)

    simulator = json.loads((diag / "simulator_calibration" / "SUMMARY.json").read_text())
    assert simulator["counts"] == {"cells": 60, "missing_cells": 0, "states": 720}
    assert simulator["groups"]["collapsed"]["cells"] == 5
    assert simulator["groups"]["stable"]["cells"] == 55
    assert simulator["groups"]["all"]["both_rankings_outside_dead_zone_states"] == 69
    assert simulator["groups"]["all"]["sign_discordance_states"] == 29
    print("PASS diagnostics: exposure, route shadow, frozen critic, and simulator screening")


def verify_figure_inputs(diag: Path) -> None:
    geometry = diag / "matched_geometry"
    movement = list(csv.DictReader((geometry / "lin2_run_movement.csv").open(newline="")))
    diagnostics = list(csv.DictReader((geometry / "lin2_run_diagnostics.csv").open(newline="")))
    movement_keys = {(row["env"], float(row["tau"]), int(row["seed"])) for row in movement}
    diagnostic_keys = {(row["env"], float(row["tau"]), int(row["seed"])) for row in diagnostics}
    assert len(movement) == 90 and movement_keys == diagnostic_keys
    for row in movement:
        assert float(row["critic_displacement_sq_mean_metric_mean"]) >= 0
        assert float(row["final_displacement_sq_mean_metric_mean"]) >= 0

    frontier = list(csv.DictReader((geometry / "hopper_lin_frontier.csv").open(newline="")))
    grouped: dict[tuple[int, float], set[int]] = {}
    for row in frontier:
        assert row["env"] == "hopper-medium-v2"
        key = int(row["total_hops"]), float(row["tau"])
        grouped.setdefault(key, set()).add(int(row["seed"]))
    assert len(frontier) == 28 and len(grouped) == 14
    assert all(seeds == {0, 1} for seeds in grouped.values())
    assert {tau for hops, tau in grouped if hops == 2} == {tau for hops, tau in grouped if hops == 3}

    manifest = json.loads((geometry / "figure_inputs_MANIFEST.json").read_text())
    assert manifest["outputs"] == {
        "lin2_run_movement.csv": 90,
        "hopper_lin_frontier.csv": 28,
    }
    print("PASS figure inputs: run-level movement and Hopper frontier rows are complete")


def verify_k4(root: Path) -> None:
    hopper: dict[float, list[float]] = {tau: [] for tau in (7.0, 10.0, 12.0, 14.0, 17.0, 20.0)}
    for seed in range(4):
        with (root / "K=4" / "Imp" / f"seed{seed}.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                tau = float(row["tau"])
                if tau in hopper:
                    hopper[tau].append(float(row["hopper-medium-v2"]))
    expected = {
        7.0: (84.812625, 7.974194),
        10.0: (100.447300, 1.048321),
        12.0: (101.229775, 0.422805),
        14.0: (101.102825, 0.703775),
        17.0: (100.874025, 1.734656),
        20.0: (41.810250, 43.973298),
    }
    for tau, scores in hopper.items():
        assert len(scores) == 4
        close(statistics.mean(scores), expected[tau][0], 2e-4)
        close(statistics.stdev(scores), expected[tau][1], 2e-4)
    print("PASS separate-machine K=4: four-seed Hopper-medium means and sample SDs")


def verify_manuscript(path: Path) -> None:
    text = "\n".join(file.read_text() for file in [path / "paper.tex", *sorted((path / "sections").glob("*.tex"))])
    required = (
        "$35/63$ to $17/63$",
        "to $24/63$",
        "$88/90$",
        "$36/90$",
        "all eight",
        "$41.8\\pm44.0$",
        "separate-machine",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise AssertionError(f"manuscript tokens missing: {missing}")
    print("PASS manuscript source: released-result claims are present with provenance wording")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "sweep_results")
    parser.add_argument("--manuscript-dir", type=Path)
    args = parser.parse_args()
    verify_complete_grid(args.results_dir)
    verify_controls(args.results_dir / "diagnostics")
    verify_diagnostics(args.results_dir / "diagnostics")
    verify_figure_inputs(args.results_dir / "diagnostics")
    verify_k4(args.results_dir)
    if args.manuscript_dir:
        verify_manuscript(args.manuscript_dir)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
