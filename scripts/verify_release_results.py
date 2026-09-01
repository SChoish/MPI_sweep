#!/usr/bin/env python3
"""Verify released BAR results and the manuscript-facing numerical claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from itertools import product
from pathlib import Path

import numpy as np

GRID = (0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0)
HIGH = tuple(tau for tau in GRID if tau >= 4.0)
AUDIT_TAUS = (4.0, 7.0, 10.0, 14.0, 20.0)
FRONTIER_TAUS = (4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0)
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
METHODS = {
    "td3": ("K=1", "Imp", (0, 1, 2, 3)),
    "prox2": ("K=2", "Imp", (0, 1, 2, 3)),
    "prox3": ("K=3", "Imp", (0, 1, 2, 3)),
    "prox4": ("K=4", "Imp", (0, 1, 2, 3)),
    "lin2": ("K=2", "Exp", (0, 1, 2, 3)),
    "lin3": ("K=3", "Exp", (0, 1, 2, 3)),
}
LINEAR_EXPECTED = {
    "lin2": (38.42734007936508, 25, 112),
    "lin3": (46.43985436507936, 21, 95),
}
PROXIMAL = (("k1", "td3"), ("k2", "prox2"), ("k3", "prox3"), ("k4", "prox4"))
LINEARIZED = (("l2", "lin2"), ("l3", "lin3"))
EXPECTED_CLAIMS = {
    "stability": {
        "four_seed_proximal": {
            "high_means": {
                "k1": 25.379536904761903,
                "k2": 41.2018873015873,
                "k3": 51.91607460317461,
                "k4": 63.971089285714285,
            },
            "cell_collapses": {
                "k1": (34, 63),
                "k2": (22, 63),
                "k3": (16, 63),
                "k4": (8, 63),
            },
            "raw_collapses": {
                "k1": (142, 252),
                "k2": (111, 252),
                "k3": (84, 252),
                "k4": (54, 252),
            },
        },
        "p4_vs_td3": {
            "mean_difference": 38.59155238095238,
            "task_bootstrap95": (21.392935714285713, 54.62618588293648),
            "positive_tasks": (8, 9),
            "raw_collapse_risk_difference": -0.34920634920634924,
            "raw_collapse_risk_cluster95": (-0.5238095238095238, -0.1746031746031746),
        },
        "p4_vs_p3": {
            "mean_difference": 12.055014682539685,
            "task_bootstrap95": (3.837716369047619, 21.111962301587305),
            "positive_tasks": (7, 9),
            "cell_median_difference": 0.342125,
            "rescued_cells": 8,
            "reverse_cells": 0,
            "rescued_gain_share": 0.5474808497826944,
            "raw_collapse_risk_difference": -0.11904761904761904,
            "raw_collapse_risk_cluster95": (-0.21825396825396823, -0.023809523809523808),
        },
        "lin3_vs_lin2": {
            "mean_difference": 8.012514285714285,
            "task_bootstrap95": (3.5571091269841255, 13.53030313492063),
            "positive_tasks": (8, 9),
            "cell_median_difference": 1.3682250000000025,
            "positive_cells": (45, 63),
            "rescued_cells": 5,
            "reverse_cells": 1,
            "rescued_gain_share": 0.4283758798736262,
            "raw_collapse_risk_difference": -0.06746031746031746,
            "raw_collapse_risk_cluster95": (-0.13095238095238093, -0.003968253968253951),
        },
        "linearized_seed_pairs": {
            "seeds_0_1": {
                "high_means": {
                    "l2": 36.68378968253968,
                    "l3": 43.911297619047616,
                },
                "cell_collapses": {"l2": 29, "l3": 24},
                "raw_collapses": {"l2": 58, "l3": 53},
                "mean_difference": 7.2275079365079336,
                "task_bootstrap95": (1.9911769841269846, 12.467574603174603),
                "positive_tasks": (8, 9),
            },
            "seeds_2_3": {
                "high_means": {
                    "l2": 40.17089047619048,
                    "l3": 48.96841111111111,
                },
                "cell_collapses": {"l2": 26, "l3": 20},
                "raw_collapses": {"l2": 54, "l3": 42},
                "mean_difference": 8.79752063492063,
                "task_bootstrap95": (4.009152400793652, 15.205206130952376),
                "positive_tasks": (9, 9),
            },
        },
        "host_separated_seed_pairs": {
            "seeds_0_1": {
                "high_means": {
                    "k1": 25.970757142857142,
                    "k2": 42.021298412698414,
                    "k3": 52.78328333333333,
                    "k4": 63.61039920634921,
                },
                "cell_collapses": {"k1": 35, "k2": 25, "k3": 17, "k4": 9},
                "raw_collapses": {"k1": 68, "k2": 53, "k3": 41, "k4": 29},
            },
            "seeds_2_3": {
                "high_means": {
                    "k1": 24.788316666666667,
                    "k2": 40.38247619047619,
                    "k3": 51.04886587301587,
                    "k4": 64.33177936507936,
                },
                "cell_collapses": {"k1": 35, "k2": 26, "k3": 16, "k4": 10},
                "raw_collapses": {"k1": 74, "k2": 58, "k3": 43, "k4": 25},
            },
        },
    },
    "route": {"mean_delta": 0.12351472249786054},
    "simulator": {
        "checkpoint_median_of_medians_stable": -41.367003891851624,
        "checkpoint_median_of_medians_collapsed": 1697181925378.9092,
    },
}


def close(actual: float, expected: float, tol: float = 5e-4) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{actual} != {expected} within {tol}")


def finite(value: str | float | int, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise AssertionError(f"{label} is not finite: {value!r}")
    return result


def csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise AssertionError(f"{path}: duplicate CSV headers")
        missing = required - set(fields)
        if missing:
            raise AssertionError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def unique_keys(
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
    label: str,
) -> set[tuple[str, ...]]:
    keys = {tuple(row[field] for field in fields) for row in rows}
    if len(keys) != len(rows):
        raise AssertionError(f"{label}: duplicate keys for {fields}")
    return keys


def load_sweep(
    root: Path,
    hops: str,
    integrator: str,
    seeds: tuple[int, ...],
) -> tuple[list[str], dict[tuple[float, int, str], float]]:
    values: dict[tuple[float, int, str], float] = {}
    for seed in seeds:
        path = root / hops / integrator / f"seed{seed}.csv"
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle)
            fields = rows.fieldnames or []
            if len(fields) != len(set(fields)):
                raise AssertionError(f"{path}: duplicate CSV headers")
            if set(fields) != {"tau", *ENVIRONMENTS}:
                raise AssertionError(f"{path}: unexpected environment columns {fields}")
            seen_taus: set[float] = set()
            for row in rows:
                tau = finite(row["tau"], f"{path}: tau")
                if tau in seen_taus:
                    raise AssertionError(f"{path}: duplicate tau row {tau}")
                seen_taus.add(tau)
                for env in ENVIRONMENTS:
                    if tau not in GRID:
                        if row[env] not in ("", "—"):
                            finite(row[env], f"{path}: extra tau={tau}, env={env}")
                        continue
                    score = finite(row[env], f"{path}: tau={tau}, env={env}")
                    key = (tau, seed, env)
                    if key in values:
                        raise AssertionError(f"{path}: duplicate sweep key {key}")
                    values[key] = score
            missing_taus = set(GRID) - seen_taus
            if missing_taus:
                raise AssertionError(f"{path}: missing GRID rows {sorted(missing_taus)}")
    expected_values = len(GRID) * len(seeds) * len(ENVIRONMENTS)
    if len(values) != expected_values:
        raise AssertionError(f"{hops}/{integrator}: {len(values)} values, expected {expected_values}")
    return list(ENVIRONMENTS), values


def task_bootstrap_interval(data: list[float], rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(data, dtype=float)
    picks = rng.integers(0, len(values), size=(100_000, len(values)))
    interval = np.percentile(values[picks].mean(axis=1), (2.5, 97.5))
    return float(interval[0]), float(interval[1])


def verify_complete_grid(root: Path, claims: dict[str, object]) -> None:
    stability = claims["stability"]
    four_seed = stability["four_seed_proximal"]
    proximal_keys = dict(PROXIMAL)
    for method, (hops, integrator, seeds) in METHODS.items():
        envs, values = load_sweep(root, hops, integrator, seeds)
        high = [score for (tau, _, _), score in values.items() if tau in HIGH]
        cells = [
            statistics.mean(values[(tau, seed, env)] for seed in seeds)
            for tau in HIGH
            for env in envs
        ]
        high_mean = statistics.mean(high)
        cell_collapses = sum(score < 20 for score in cells)
        raw_collapses = sum(score < 20 for score in high)
        if method in proximal_keys.values():
            short = next(key for key, name in PROXIMAL if name == method)
            exp_mean = float(four_seed["high_means"][short])
            exp_cells, cell_denominator = map(int, four_seed["cell_collapses"][short])
            exp_raw, raw_denominator = map(int, four_seed["raw_collapses"][short])
        else:
            exp_mean, exp_cells, exp_raw = LINEAR_EXPECTED[method]
            cell_denominator = len(HIGH) * len(envs)
            raw_denominator = cell_denominator * len(seeds)
        assert len(cells) == cell_denominator == 63
        assert len(high) == raw_denominator
        close(high_mean, exp_mean)
        assert (cell_collapses, raw_collapses) == (exp_cells, exp_raw)
        print(
            f"PASS complete-grid {method}: high={high_mean:.4f}, "
            f"collapse={cell_collapses}/{cell_denominator} and {raw_collapses}/{raw_denominator}"
        )


def load_proximal_sweeps(
    root: Path,
) -> tuple[list[str], dict[str, dict[tuple[float, int, str], float]]]:
    source: dict[str, dict[tuple[float, int, str], float]] = {}
    environments: list[str] | None = None
    for _, method in PROXIMAL:
        hops, integrator, seeds = METHODS[method]
        current_envs, values = load_sweep(root, hops, integrator, seeds)
        if environments is None:
            environments = current_envs
        assert current_envs == environments
        source[method] = values
    assert environments is not None
    return environments, source


def load_linearized_sweeps(
    root: Path,
) -> tuple[list[str], dict[str, dict[tuple[float, int, str], float]]]:
    source: dict[str, dict[tuple[float, int, str], float]] = {}
    environments: list[str] | None = None
    for _, method in LINEARIZED:
        hops, integrator, seeds = METHODS[method]
        current_envs, values = load_sweep(root, hops, integrator, seeds)
        if environments is None:
            environments = current_envs
        assert current_envs == environments
        source[method] = values
    assert environments is not None
    return environments, source


def verify_host_separated_pairs(root: Path, claims: dict[str, object]) -> None:
    envs, source = load_proximal_sweeps(root)
    pair_claims = claims["stability"]["host_separated_seed_pairs"]
    for pair_key, pair_seeds in (("seeds_0_1", (0, 1)), ("seeds_2_3", (2, 3))):
        expected = pair_claims[pair_key]
        observed_high: list[float] = []
        for short, method in PROXIMAL:
            values = source[method]
            high = [
                values[(tau, seed, env)]
                for tau in HIGH
                for seed in pair_seeds
                for env in envs
            ]
            cells = [
                statistics.mean(values[(tau, seed, env)] for seed in pair_seeds)
                for tau in HIGH
                for env in envs
            ]
            assert len(high) == 126 and len(cells) == 63
            high_mean = statistics.mean(high)
            observed_high.append(high_mean)
            close(high_mean, float(expected["high_means"][short]), 1e-12)
            assert sum(score < 20 for score in cells) == int(expected["cell_collapses"][short])
            assert sum(score < 20 for score in high) == int(expected["raw_collapses"][short])
        assert all(left < right for left, right in zip(observed_high, observed_high[1:]))
    print("PASS proximal host-separated pairs: both pairs reproduce strict K=1<K=2<K=3<K=4 ordering and counts")

    linear_envs, linear = load_linearized_sweeps(root)
    assert linear_envs == envs
    linear_claims = claims["stability"]["linearized_seed_pairs"]
    for pair_key, pair_seeds in (("seeds_0_1", (0, 1)), ("seeds_2_3", (2, 3))):
        expected = linear_claims[pair_key]
        observed_high: list[float] = []
        for short, method in LINEARIZED:
            values = linear[method]
            high = [
                values[(tau, seed, env)]
                for tau in HIGH
                for seed in pair_seeds
                for env in envs
            ]
            cells = [
                statistics.mean(values[(tau, seed, env)] for seed in pair_seeds)
                for tau in HIGH
                for env in envs
            ]
            assert len(high) == 126 and len(cells) == 63
            high_mean = statistics.mean(high)
            observed_high.append(high_mean)
            close(high_mean, float(expected["high_means"][short]), 1e-12)
            assert sum(score < 20 for score in cells) == int(expected["cell_collapses"][short])
            assert sum(score < 20 for score in high) == int(expected["raw_collapses"][short])
        assert observed_high[0] < observed_high[1]
        task_differences = [
            statistics.mean(
                linear["lin3"][(tau, seed, env)] - linear["lin2"][(tau, seed, env)]
                for tau in HIGH
                for seed in pair_seeds
            )
            for env in envs
        ]
        close(statistics.mean(task_differences), float(expected["mean_difference"]), 1e-12)
        assert sum(value > 0 for value in task_differences) == int(expected["positive_tasks"][0])
        assert len(task_differences) == int(expected["positive_tasks"][1]) == 9
        interval = task_bootstrap_interval(task_differences, np.random.default_rng(20260830))
        close(interval[0], float(expected["task_bootstrap95"][0]), 1e-12)
        close(interval[1], float(expected["task_bootstrap95"][1]), 1e-12)
    print("PASS linearized seed pairs: both disjoint pairs reproduce K=2<K=3 and reported intervals")


def verify_stability_claims(root: Path, claims: dict[str, object]) -> None:
    envs, source = load_proximal_sweeps(root)
    stability = claims["stability"]

    def high_task_mean(method: str, env: str) -> float:
        values = source[method]
        return statistics.mean(values[(tau, seed, env)] for tau in HIGH for seed in (0, 1, 2, 3))

    score_rng = np.random.default_rng(20260830)
    for baseline, claim_key in (("td3", "p4_vs_td3"), ("prox3", "p4_vs_p3")):
        data = [high_task_mean("prox4", env) - high_task_mean(baseline, env) for env in envs]
        claim = stability[claim_key]
        close(statistics.mean(data), float(claim["mean_difference"]), 1e-12)
        assert sum(value > 0 for value in data) == int(claim["positive_tasks"][0])
        assert len(data) == int(claim["positive_tasks"][1]) == 9
        interval = task_bootstrap_interval(data, score_rng)
        close(interval[0], float(claim["task_bootstrap95"][0]), 1e-12)
        close(interval[1], float(claim["task_bootstrap95"][1]), 1e-12)

    p4_values = source["prox4"]
    p3_values = source["prox3"]
    p4_cells = [
        statistics.mean(p4_values[(tau, seed, env)] for seed in (0, 1, 2, 3))
        for tau in HIGH
        for env in envs
    ]
    p3_cells = [
        statistics.mean(p3_values[(tau, seed, env)] for seed in (0, 1, 2, 3))
        for tau in HIGH
        for env in envs
    ]
    p4_p3 = stability["p4_vs_p3"]
    gains = [left - right for left, right in zip(p4_cells, p3_cells)]
    close(statistics.median(gains), float(p4_p3["cell_median_difference"]), 1e-12)
    assert sum(right < 20 <= left for left, right in zip(p4_cells, p3_cells)) == int(p4_p3["rescued_cells"])
    assert sum(left < 20 <= right for left, right in zip(p4_cells, p3_cells)) == int(p4_p3["reverse_cells"])
    rescued_gain = sum(
        gain
        for gain, left, right in zip(gains, p4_cells, p3_cells)
        if right < 20 <= left
    )
    assert sum(gains) > 0
    close(rescued_gain / sum(gains), float(p4_p3["rescued_gain_share"]), 1e-12)

    risk_rng = np.random.default_rng(20260830)
    for baseline, claim_key in (("td3", "p4_vs_td3"), ("prox3", "p4_vs_p3")):
        baseline_values = source[baseline]
        data = [
            statistics.mean(p4_values[(tau, seed, env)] < 20 for tau in HIGH for seed in (0, 1, 2, 3))
            - statistics.mean(
                baseline_values[(tau, seed, env)] < 20
                for tau in HIGH
                for seed in (0, 1, 2, 3)
            )
            for env in envs
        ]
        claim = stability[claim_key]
        close(statistics.mean(data), float(claim["raw_collapse_risk_difference"]), 1e-12)
        interval = task_bootstrap_interval(data, risk_rng)
        close(interval[0], float(claim["raw_collapse_risk_cluster95"][0]), 1e-12)
        close(interval[1], float(claim["raw_collapse_risk_cluster95"][1]), 1e-12)
    linear_envs, linear = load_linearized_sweeps(root)
    assert linear_envs == envs
    lin2_values = linear["lin2"]
    lin3_values = linear["lin3"]
    lin_claim = stability["lin3_vs_lin2"]
    task_differences = [
        statistics.mean(
            lin3_values[(tau, seed, env)] - lin2_values[(tau, seed, env)]
            for tau in HIGH
            for seed in (0, 1, 2, 3)
        )
        for env in envs
    ]
    close(statistics.mean(task_differences), float(lin_claim["mean_difference"]), 1e-12)
    assert sum(value > 0 for value in task_differences) == int(lin_claim["positive_tasks"][0])
    assert len(task_differences) == int(lin_claim["positive_tasks"][1]) == 9
    interval = task_bootstrap_interval(task_differences, np.random.default_rng(20260830))
    close(interval[0], float(lin_claim["task_bootstrap95"][0]), 1e-12)
    close(interval[1], float(lin_claim["task_bootstrap95"][1]), 1e-12)

    lin2_cells = [
        statistics.mean(lin2_values[(tau, seed, env)] for seed in (0, 1, 2, 3))
        for tau in HIGH
        for env in envs
    ]
    lin3_cells = [
        statistics.mean(lin3_values[(tau, seed, env)] for seed in (0, 1, 2, 3))
        for tau in HIGH
        for env in envs
    ]
    gains = [left - right for left, right in zip(lin3_cells, lin2_cells)]
    close(statistics.median(gains), float(lin_claim["cell_median_difference"]), 1e-12)
    assert sum(gain > 0 for gain in gains) == int(lin_claim["positive_cells"][0])
    assert len(gains) == int(lin_claim["positive_cells"][1]) == 63
    assert sum(right < 20 <= left for left, right in zip(lin3_cells, lin2_cells)) == int(
        lin_claim["rescued_cells"]
    )
    assert sum(left < 20 <= right for left, right in zip(lin3_cells, lin2_cells)) == int(
        lin_claim["reverse_cells"]
    )
    rescued_gain = sum(
        gain
        for gain, left, right in zip(gains, lin3_cells, lin2_cells)
        if right < 20 <= left
    )
    assert sum(gains) > 0
    close(rescued_gain / sum(gains), float(lin_claim["rescued_gain_share"]), 1e-12)

    risk_differences = [
        statistics.mean(lin3_values[(tau, seed, env)] < 20 for tau in HIGH for seed in (0, 1, 2, 3))
        - statistics.mean(lin2_values[(tau, seed, env)] < 20 for tau in HIGH for seed in (0, 1, 2, 3))
        for env in envs
    ]
    close(statistics.mean(risk_differences), float(lin_claim["raw_collapse_risk_difference"]), 1e-12)
    interval = task_bootstrap_interval(risk_differences, np.random.default_rng(20260830))
    close(interval[0], float(lin_claim["raw_collapse_risk_cluster95"][0]), 1e-12)
    close(interval[1], float(lin_claim["raw_collapse_risk_cluster95"][1]), 1e-12)
    print(
        "PASS stability claims: proximal and linearized task bootstraps, "
        "cell transitions, rescue shares, and raw collapse-risk intervals"
    )


def verify_controls(diag: Path) -> None:
    rows = csv_rows(
        diag / "control_final_scores.csv",
        {"experiment", "condition", "environment", "T", "seed", "step", "d4rl_score"},
    )
    summary = csv_rows(
        diag / "control_summary.csv",
        {
            "experiment",
            "condition",
            "n_runs",
            "mean_d4rl_score",
            "median_d4rl_score",
            "collapse_below_20",
        },
    )
    expected_groups = {
        ("compute", "sequential"),
        ("compute", "fixed_reference"),
        ("compute", "direct_final"),
        ("target_lag", "canonical"),
        ("target_lag", "actor_lag_3x"),
    }
    summary_groups = unique_keys(summary, ("experiment", "condition"), "control summary")
    row_groups = {(row["experiment"], row["condition"]) for row in rows}
    if not rows or not summary or row_groups != summary_groups or summary_groups != expected_groups:
        raise AssertionError(
            "control groups must be nonempty and exactly match the five expected summary groups"
        )
    run_keys = {
        (
            row["experiment"],
            row["condition"],
            row["environment"],
            finite(row["T"], "control T"),
            int(row["seed"]),
        )
        for row in rows
    }
    if len(run_keys) != len(rows):
        raise AssertionError("control_final_scores.csv: duplicate run keys")
    scores = [finite(row["d4rl_score"], "control d4rl_score") for row in rows]
    assert len(scores) == len(rows)
    for expected in summary:
        expected_key = expected["experiment"], expected["condition"]
        group = [
            finite(row["d4rl_score"], f"control score {expected_key}")
            for row in rows
            if (row["experiment"], row["condition"]) == expected_key
        ]
        n_runs = int(expected["n_runs"])
        if n_runs <= 0 or len(group) != n_runs:
            raise AssertionError(f"control group {expected_key}: {len(group)} rows, expected {n_runs}")
        close(statistics.mean(group), finite(expected["mean_d4rl_score"], "control mean"), 1e-6)
        close(statistics.median(group), finite(expected["median_d4rl_score"], "control median"), 1e-6)
        assert sum(score < 20 for score in group) == int(expected["collapse_below_20"])
    print("PASS targeted controls: exact groups, unique finite runs, and summaries agree")


def verify_diagnostics(diag: Path, claims: dict[str, object]) -> None:
    target_dir = diag / "target_policy_exposure"
    target = json.loads((target_dir / "SUMMARY.json").read_text())
    target_manifest = json.loads((target_dir / "MANIFEST.json").read_text())
    target_rows = csv_rows(
        target_dir / "target_policy_exposure_summary.csv",
        {"env", "tau", "seed", "method", "checkpoint_step", "n_states"},
    )
    expected_target_keys = set(
        product(ENVIRONMENTS, AUDIT_TAUS, (0, 1), ("td3", "mpi2", "mpi3"))
    )
    target_keys = {
        (
            row["env"],
            finite(row["tau"], "target exposure tau"),
            int(row["seed"]),
            row["method"],
        )
        for row in target_rows
    }
    if len(target_keys) != len(target_rows) or target_keys != expected_target_keys:
        raise AssertionError("target-policy exposure rows are not the exact 9x5x2x3 grid")
    assert all(int(row["checkpoint_step"]) == 1_000_000 for row in target_rows)
    assert len(target_rows) == target["completed_cells"] == target_manifest["completed_cells"] == 270
    assert target["missing_checkpoints"] == 0 and target_manifest["missing_checkpoints"] == []
    assert set(target["by_method"]) == {"td3", "mpi2", "mpi3"}
    assert all(target["by_method"][method]["cells"] == 90 for method in target["by_method"])
    pair = target["paired_method_comparisons"]["mpi3_vs_td3"]
    assert pair["pairs"] == 90
    assert pair["target_deterministic"]["left_smaller_cells"] == 88
    close(pair["target_deterministic"]["median_left_over_right_nonzero"], 0.6576946525, 1e-9)
    close(target["proxy_alignment"]["all_cells"]["pearson_proxy_vs_target_deterministic"], 0.9999114535, 1e-9)

    ext_target_dir = diag / "hosts" / "ext_csh" / "target_policy_exposure"
    ext_target_manifest = json.loads((ext_target_dir / "MANIFEST.json").read_text())
    ext_target_rows = csv_rows(
        ext_target_dir / "target_policy_exposure_summary.csv",
        {
            "env",
            "tau",
            "seed",
            "method",
            "checkpoint_step",
            "n_states",
            "target_deterministic_to_next_data_sq_mean_metric_mean",
        },
    )
    ext_envs = (
        "hopper-medium-replay-v2",
        "hopper-medium-v2",
        "walker2d-expert-v2",
        "walker2d-medium-v2",
    )
    expected_ext_target_keys = set(
        product(ext_envs, AUDIT_TAUS, (2, 3), ("mpi2", "mpi3"))
    )
    ext_target_keys = {
        (
            row["env"],
            finite(row["tau"], "ext_csh target tau"),
            int(row["seed"]),
            row["method"],
        )
        for row in ext_target_rows
    }
    if (
        len(ext_target_keys) != len(ext_target_rows)
        or ext_target_keys != expected_ext_target_keys
    ):
        raise AssertionError(
            "ext_csh target exposure rows are not the exact 4x5x2x2 grid"
        )
    assert (
        len(ext_target_rows)
        == ext_target_manifest["completed_cells"]
        == 80
    )
    assert ext_target_manifest["missing_checkpoints"] == []
    assert all(int(row["checkpoint_step"]) == 1_000_000 for row in ext_target_rows)
    assert all(int(row["n_states"]) == 4096 for row in ext_target_rows)

    ext_pairs: dict[tuple[str, float, int], dict[str, float]] = {}
    for row in ext_target_rows:
        key = (
            row["env"],
            finite(row["tau"], "ext_csh pair tau"),
            int(row["seed"]),
        )
        ext_pairs.setdefault(key, {})[row["method"]] = finite(
            row["target_deterministic_to_next_data_sq_mean_metric_mean"],
            "ext_csh deterministic target exposure",
        )
    assert len(ext_pairs) == 40
    assert all(set(methods) == {"mpi2", "mpi3"} for methods in ext_pairs.values())
    ext_ratios = [methods["mpi3"] / methods["mpi2"] for methods in ext_pairs.values()]
    assert sum(methods["mpi3"] < methods["mpi2"] for methods in ext_pairs.values()) == 36
    close(statistics.median(ext_ratios), 0.9053559676810246, 1e-12)

    geometry = diag / "matched_geometry"
    rows = csv_rows(
        geometry / "td3_prox3_run_diagnostics.csv",
        {
            "env",
            "tau",
            "seed",
            "method",
            "total_hops",
            "final_displacement_sq_mean_metric_mean",
        },
    )
    expected_matched_keys = set(
        product(ENVIRONMENTS, AUDIT_TAUS, (0, 1), ("td3", "mpi3"))
    )
    matched_keys = {
        (
            row["env"],
            finite(row["tau"], "matched diagnostics tau"),
            int(row["seed"]),
            row["method"],
        )
        for row in rows
    }
    if len(matched_keys) != len(rows) or matched_keys != expected_matched_keys:
        raise AssertionError("td3/prox3 matched diagnostics are not the exact 9x5x2x2 grid")
    paired: dict[tuple[str, float, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        expected_hops = 1 if row["method"] == "td3" else 3
        assert int(row["total_hops"]) == expected_hops
        finite(
            row["final_displacement_sq_mean_metric_mean"],
            "matched final displacement",
        )
        paired.setdefault(
            (row["env"], finite(row["tau"], "matched tau"), int(row["seed"])),
            {},
        )[row["method"]] = row
    expected_pair_keys = set(product(ENVIRONMENTS, AUDIT_TAUS, (0, 1)))
    assert set(paired) == expected_pair_keys
    assert all(set(methods) == {"td3", "mpi3"} for methods in paired.values())
    final_lower = sum(
        finite(
            methods["mpi3"]["final_displacement_sq_mean_metric_mean"],
            "mpi3 final displacement",
        )
        < finite(
            methods["td3"]["final_displacement_sq_mean_metric_mean"],
            "td3 final displacement",
        )
        for methods in paired.values()
    )
    assert len(rows) == 180 and len(paired) == 90 and final_lower == 36

    route_dir = diag / "route_shadow"
    route = json.loads((route_dir / "SUMMARY.json").read_text())
    route_manifest = json.loads((route_dir / "MANIFEST.json").read_text())
    route_runs = csv_rows(
        route_dir / "run_level.csv",
        {"env", "tau", "seed", "n_checkpoints", "mean_paired_delta_muk_minus_mu1"},
    )
    expected_route_keys = {
        ("hopper-medium-v2", 10.0, 0),
        ("hopper-medium-v2", 10.0, 1),
        ("hopper-medium-v2", 20.0, 0),
        ("hopper-medium-v2", 20.0, 1),
        ("walker2d-medium-v2", 4.0, 0),
        ("walker2d-medium-v2", 4.0, 1),
        ("walker2d-medium-v2", 7.0, 0),
        ("walker2d-medium-v2", 7.0, 1),
    }
    route_run_keys = {
        (row["env"], finite(row["tau"], "route tau"), int(row["seed"]))
        for row in route_runs
    }
    if len(route_run_keys) != len(route_runs) or route_run_keys != expected_route_keys:
        raise AssertionError("route run rows do not match the exact targeted 4x2 grid")
    assert len(route_runs) == route["n_runs"] == 8
    route_deltas = [
        finite(row["mean_paired_delta_muk_minus_mu1"], "route run delta")
        for row in route_runs
    ]
    close(statistics.mean(route_deltas), float(route["mean_delta"]), 1e-12)
    assert sum(delta > 0 for delta in route_deltas) == route["positive_delta_runs"] == 8
    close(route["mean_delta"], 0.1235147225, 1e-10)
    close(route["two_sided_exact_sign_flip_p"], 0.0078125, 1e-12)
    assert route_manifest["arguments"]["steps"] == "250000 500000 750000 1000000"

    route_checkpoints = csv_rows(
        route_dir / "checkpoint_pairs.csv",
        {"env", "tau", "seed", "step", "paired_delta_muk_minus_mu1"},
    )
    route_checkpoint_keys = {
        (
            row["env"],
            finite(row["tau"], "route checkpoint tau"),
            int(row["seed"]),
            int(row["step"]),
        )
        for row in route_checkpoints
    }
    expected_route_checkpoint_keys = {
        (*key, step)
        for key, step in product(expected_route_keys, (250_000, 500_000, 750_000, 1_000_000))
    }
    if (
        len(route_checkpoint_keys) != len(route_checkpoints)
        or route_checkpoint_keys != expected_route_checkpoint_keys
    ):
        raise AssertionError("route checkpoint rows do not match the exact 8x4 grid")
    assert len(route_checkpoints) == sum(int(row["n_checkpoints"]) for row in route_runs) == 32
    for row in route_checkpoints:
        finite(row["paired_delta_muk_minus_mu1"], "route checkpoint delta")

    frozen = json.loads((diag / "frozen_critic_small_step" / "SUMMARY.json").read_text())
    frozen_keys = {(run["environment"], int(run["seed"])) for run in frozen["runs"]}
    assert len(frozen_keys) == len(frozen["runs"]) == frozen["n_runs"] == 18
    eligible = [run for run in frozen["runs"] if run["local_points_eligible"] > 0]
    assert len(eligible) == 15
    close(max(run["local_max_relative_difference"] for run in eligible), 0.0268159275, 1e-10)

    simulator_dir = diag / "simulator_calibration"
    simulator = json.loads((simulator_dir / "SUMMARY.json").read_text())
    simulator_manifest = json.loads((simulator_dir / "MANIFEST.json").read_text())
    simulator_cells = csv_rows(
        simulator_dir / "cells.csv",
        {"env", "tau", "method", "seed", "d4rl_score", "collapsed", "n_states"},
    )
    simulator_envs = (
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
    )
    expected_simulator_keys = set(
        product(simulator_envs, AUDIT_TAUS, ("mpi2", "mpi3"), (0, 1))
    )
    simulator_keys = {
        (
            row["env"],
            finite(row["tau"], "simulator tau"),
            row["method"],
            int(row["seed"]),
        )
        for row in simulator_cells
    }
    if len(simulator_keys) != len(simulator_cells) or simulator_keys != expected_simulator_keys:
        raise AssertionError("simulator cells do not match the exact 3x5x2x2 grid")
    simulator_states = sum(int(row["n_states"]) for row in simulator_cells)
    assert simulator["counts"] == {
        "cells": len(simulator_cells),
        "missing_cells": 0,
        "states": simulator_states,
    } == {"cells": 60, "missing_cells": 0, "states": 720}
    assert simulator_manifest["environments"] == list(simulator_envs)
    assert set(simulator_manifest["methods"]) == {"mpi2", "mpi3"}
    assert set(map(float, simulator_manifest["taus"])) == set(AUDIT_TAUS)
    assert set(map(int, simulator_manifest["training_seeds"])) == {0, 1}
    for row in simulator_cells:
        finite(row["d4rl_score"], "simulator cell score")
    assert simulator["groups"]["collapsed"]["cells"] == sum(
        int(row["collapsed"]) for row in simulator_cells
    ) == 5
    assert simulator["groups"]["stable"]["cells"] == 55
    assert simulator["groups"]["all"]["both_rankings_outside_dead_zone_states"] == 69
    assert simulator["groups"]["all"]["sign_discordance_states"] == 29

    audit = diag / "audit_pack"
    audit_manifest = json.loads((audit / "audit_manifest.json").read_text())
    expected_audit_files = {
        "failure_diagnostics_run.csv",
        "route_intervention_run.csv",
        "sensitivity.csv",
        "simulator_checkpoint.csv",
        "target_exposure_run.csv",
    }
    assert set(audit_manifest["counts"]) == expected_audit_files
    audit_rows: dict[str, list[dict[str, str]]] = {}
    for filename in expected_audit_files:
        current = csv_rows(audit / filename, set())
        audit_rows[filename] = current
        assert len(current) == int(audit_manifest["counts"][filename])

    route_rows = audit_rows["route_intervention_run.csv"]
    audit_route_keys = {
        (row["env"], finite(row["T"], "audit route T"), int(row["seed"]))
        for row in route_rows
    }
    if len(audit_route_keys) != len(route_rows) or audit_route_keys != expected_route_keys:
        raise AssertionError("audit route rows do not match route-shadow run keys")
    assert len(route_rows) == route["n_runs"] == audit_manifest["integrity"]["route"]["rows"]
    assert sum(finite(row["delta"], "audit route delta") > 0 for row in route_rows) == 8
    assert sum(
        finite(row["delta_final_checkpoint_only"], "audit final route delta") > 0
        for row in route_rows
    ) == 5
    close(
        statistics.mean(finite(row["delta"], "audit route mean delta") for row in route_rows),
        float(claims["route"]["mean_delta"]),
        1e-12,
    )

    simulator_rows = audit_rows["simulator_checkpoint.csv"]
    audit_simulator_keys = {
        (
            row["env"],
            finite(row["T"], "audit simulator T"),
            row["method"],
            int(row["seed"]),
        )
        for row in simulator_rows
    }
    if (
        len(audit_simulator_keys) != len(simulator_rows)
        or audit_simulator_keys != expected_simulator_keys
    ):
        raise AssertionError("audit simulator rows do not match simulator cell keys")
    assert len(simulator_rows) == simulator["counts"]["cells"]
    for row in simulator_rows:
        expected_k = 2 if row["method"] == "mpi2" else 3
        assert int(row["K"]) == expected_k
        finite(row["score"], "audit simulator score")
        finite(row["median_signed_error"], "audit simulator signed error")
    stable = [
        finite(row["median_signed_error"], "stable simulator error")
        for row in simulator_rows
        if int(row["collapsed"]) == 0
    ]
    collapsed = [
        finite(row["median_signed_error"], "collapsed simulator error")
        for row in simulator_rows
        if int(row["collapsed"]) == 1
    ]
    assert (len(stable), len(collapsed)) == (55, 5)
    close(
        statistics.median(stable),
        float(claims["simulator"]["checkpoint_median_of_medians_stable"]),
        1e-9,
    )
    close(
        statistics.median(collapsed),
        float(claims["simulator"]["checkpoint_median_of_medians_collapsed"]),
        1e-3,
    )

    target_audit = audit_rows["target_exposure_run.csv"]
    target_audit_keys = {
        (
            row["env"],
            finite(row["T"], "audit target T"),
            int(row["seed"]),
            row["method"],
        )
        for row in target_audit
    }
    if len(target_audit_keys) != len(target_audit) or target_audit_keys != expected_target_keys:
        raise AssertionError("audit target-exposure rows do not match source diagnostic keys")
    for row in target_audit:
        finite(row["score"], "audit target score")

    failure_audit = audit_rows["failure_diagnostics_run.csv"]
    expected_failure_keys = set(
        product(("lin2", "prox2"), ENVIRONMENTS, AUDIT_TAUS, (0, 1))
    )
    failure_keys = {
        (
            row["method"],
            row["env"],
            finite(row["T"], "audit failure T"),
            int(row["seed"]),
        )
        for row in failure_audit
    }
    if len(failure_keys) != len(failure_audit) or failure_keys != expected_failure_keys:
        raise AssertionError("failure diagnostic audit rows are not the exact 2x9x5x2 grid")
    for row in failure_audit:
        finite(row["score"], "audit failure score")

    sensitivity = audit_rows["sensitivity.csv"]
    unique_keys(
        sensitivity,
        ("analysis", "variant", "subgroup", "metric"),
        "audit sensitivity",
    )
    for row in sensitivity:
        if row["status"] == "computed":
            finite(row["value"], "audit sensitivity value")
    assert audit_manifest["integrity"]["target_exposure"]["rows"] == len(target_audit)
    assert audit_manifest["integrity"]["failure"]["rows"] == len(failure_audit)
    assert audit_manifest["integrity"]["simulator"]["rows"] == len(simulator_rows)

    print(
        "PASS diagnostics: metadata counts, exact unique keys, exposure, route, "
        "frozen critic, and simulator medians"
    )

    bar_mcep = json.loads((diag / "bar_mcep_p3_paired" / "AUDIT.json").read_text())
    assert bar_mcep["pass"] is True
    assert bar_mcep["bar"]["n_ok"] == bar_mcep["mcep"]["n_ok"] == 90
    assert bar_mcep["paired_keys"] == 90
    assert bar_mcep["manuscript_status"].startswith("mechanism-control")

    route_k4_dir = diag / "route_shadow_k4"
    route_k4 = json.loads((route_k4_dir / "SUMMARY.json").read_text())
    route_k4_runs = csv_rows(
        route_k4_dir / "run_level.csv",
        {"env", "tau", "seed", "n_checkpoints", "mean_paired_delta_muk_minus_mu1"},
    )
    assert len(route_k4_runs) == route_k4["n_runs"] == 8
    assert route_k4["positive_delta_runs"] == 5
    close(route_k4["two_sided_exact_sign_flip_p"], 0.0625, 1e-12)
    assert route_k4["continuation_estimand"].startswith("source-native")

    print(
        "PASS diagnostics extras: BAR/MCEP paired 90/90 audit and K4 "
        "route-native exclusion (5/8, p=.0625)"
    )


def verify_figure_inputs(diag: Path) -> None:
    geometry = diag / "matched_geometry"
    movement_fields = {
        "env",
        "tau",
        "seed",
        "method",
        "total_hops",
        "d4rl_score",
        "critic_displacement_sq_mean_metric_mean",
        "final_displacement_sq_mean_metric_mean",
    }
    movement = csv_rows(geometry / "lin2_run_movement.csv", movement_fields)
    diagnostics = csv_rows(
        geometry / "lin2_run_diagnostics.csv",
        {"env", "tau", "seed", "method", "total_hops", "d4rl_score"},
    )
    expected_lin2 = set(product(ENVIRONMENTS, AUDIT_TAUS, (0, 1)))
    movement_keys = {
        (row["env"], finite(row["tau"], "lin2 movement tau"), int(row["seed"]))
        for row in movement
    }
    diagnostic_keys = {
        (row["env"], finite(row["tau"], "lin2 diagnostics tau"), int(row["seed"]))
        for row in diagnostics
    }
    if len(movement_keys) != len(movement) or movement_keys != expected_lin2:
        raise AssertionError("lin2_run_movement.csv: keys are not the exact 9x5x2 grid")
    if len(diagnostic_keys) != len(diagnostics) or diagnostic_keys != expected_lin2:
        raise AssertionError("lin2_run_diagnostics.csv: keys are not the exact 9x5x2 grid")
    for label, rows in (("movement", movement), ("diagnostics", diagnostics)):
        for row in rows:
            assert row["method"] == "exp2m" and int(row["total_hops"]) == 2
            finite(row["d4rl_score"], f"lin2 {label} score")
    for row in movement:
        critic = finite(
            row["critic_displacement_sq_mean_metric_mean"],
            "lin2 critic displacement",
        )
        final = finite(
            row["final_displacement_sq_mean_metric_mean"],
            "lin2 final displacement",
        )
        if critic < 0 or final < 0:
            raise AssertionError("lin2 displacement inputs must be nonnegative")

    frontier = csv_rows(geometry / "hopper_lin_frontier.csv", movement_fields)
    expected_frontier = {
        ("hopper-medium-v2", hops, tau, seed)
        for hops, tau, seed in product((2, 3), FRONTIER_TAUS, (0, 1))
    }
    frontier_keys = {
        (
            row["env"],
            int(row["total_hops"]),
            finite(row["tau"], "frontier tau"),
            int(row["seed"]),
        )
        for row in frontier
    }
    if len(frontier_keys) != len(frontier) or frontier_keys != expected_frontier:
        raise AssertionError("hopper_lin_frontier.csv: keys are not the exact 2x7x2 frontier")
    for row in frontier:
        hops = int(row["total_hops"])
        assert row["method"] == f"exp{hops}m"
        finite(row["d4rl_score"], "frontier score")
        critic = finite(
            row["critic_displacement_sq_mean_metric_mean"],
            "frontier critic displacement",
        )
        final = finite(
            row["final_displacement_sq_mean_metric_mean"],
            "frontier final displacement",
        )
        if critic < 0 or final < 0:
            raise AssertionError("frontier displacement inputs must be nonnegative")

    manifest = json.loads((geometry / "figure_inputs_MANIFEST.json").read_text())
    assert manifest["outputs"] == {
        "lin2_run_movement.csv": len(movement),
        "hopper_lin_frontier.csv": len(frontier),
    } == {
        "lin2_run_movement.csv": 90,
        "hopper_lin_frontier.csv": 28,
    }
    print("PASS figure inputs: exact unique movement, diagnostics, and Hopper frontier grids")


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
    print("PASS released K=4: four-seed Hopper-medium means and sample SDs")


def strip_latex_comments(text: str) -> str:
    clean_lines: list[str] = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                line = line[:index]
                break
        clean_lines.append(line)
    return "\n".join(clean_lines)


def canonical_latex(text: str) -> str:
    text = strip_latex_comments(text)
    unwrap = re.compile(
        r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|operatorname)"
        r"\s*\{([^{}]*)\}"
    )
    while True:
        updated = unwrap.sub(r"\1", text)
        if updated == text:
            break
        text = updated
    text = text.replace("~", " ")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = re.sub(r"\\([^A-Za-z])", r"\1", text)
    text = re.sub(r"[{}$]", "", text)
    return re.sub(r"\s+", "", text).lower()


def verify_manuscript(path: Path) -> None:
    sources = (
        path / "paper.tex",
        path / "supplement.tex",
        path / "reproducibility_checklist.tex",
    )
    missing_sources = [str(source) for source in sources if not source.is_file()]
    if missing_sources:
        raise AssertionError(f"manuscript sources missing: {missing_sources}")
    uncommented = "\n".join(strip_latex_comments(source.read_text()) for source in sources)
    text = canonical_latex(uncommented)
    required = {
        "four-depth high-budget sequence": "25.38to41.20to51.92to63.97",
        "collapse endpoints": "34/63to8/63",
        "P4 minus TD3 interval": "[21.39,54.63]",
        "P4 minus P3 estimate and interval": "gains12.06overk=3withtask-resamplinginterval[3.84,21.11]",
        "rescued-cell gain share": "54.7%",
        "four-seed collapse-risk interval": "-.119([-.218,-.024])",
        "four-seed linearized interval": "[3.56,13.53]",
        "four-seed linearized L2 row": "bar-l2&4&69.79&38.43&28.47&25/112",
        "four-seed linearized L3 row": "bar-l3&4&69.73&46.44&35.83&21/95",
        "linearized cell support": "positivein45/63continuouscellcontrasts",
        "linearized cell transitions": "itsfiverescuesandonereversal",
        "linearized rescue share": "42.8%",
        "linearized collapse-risk interval": "-.067([-.131,-.004])",
        "linearized pair gains": "gainsof7.23and8.80",
        "seed-pair table K1": "1&25.97&24.79&35&35",
        "seed-pair table K2": "2&42.02&40.38&25&26",
        "seed-pair table K3": "3&52.78&51.05&17&16",
        "seed-pair table K4": "4&63.61&64.33&9&10",
        "target displacement count": "88/90",
        "final proxy count": "36/90",
        "restricted exposure sensitivity": "36/40cells",
        "restricted exposure median ratio": "medianratio.905",
        "compressed route qualification": (
            "final-routeerrorislargerinallfour-checkpointaggregates"
            "butonlyfiveofeightfinalcheckpoints"
        ),
        "stable simulator median": "-41.4",
        "collapsed simulator median": "+1.70times10^12",
        "bootstrap protocol": "100,000drawsusinganalysisseed20260830",
        "audit scope": "matched270-checkpointtd3/p2/p3audit",
        "manifest limitation": "exactper-runmanifestswerenotretained",
        "provenance qualification": (
            "provenanceconsistsofthearchivedlaunchconfigurationandscorematrices"
        ),
    }
    missing = [label for label, token in required.items() if token not in text]
    if missing:
        raise AssertionError(f"manuscript claims missing after LaTeX normalization: {missing}")

    stale_tokens = {
        "old P4-P3 mean": "10.83",
        "old P4-P3 interval": "[2.23,21.97]",
        "old collapse-risk interval": "[-.206,0]",
        "old collapse-risk interval (leading zero)": "[-0.206,0]",
        "obsolete scale-match label": "scale-matched",
        "obsolete dataset name": "medium-expert-v2",
        "old evaluation cadence": "5,000criticupdates",
        "old evaluation cadence (unpunctuated)": "5000criticupdates",
        "obsolete two-seed scope": "completetwo-seedgrid",
        "obsolete two-seed linearized sweep": "two-seedlinearizedsweep",
        "obsolete linearized seed range": "linearizedgridusesseeds0--1",
        "obsolete linearized curve scope": "projected-linearizedcurvesusetwo",
        "obsolete linearized table scope": "linearizedrowsusetwo",
        "obsolete linearized denominator": "126forlinearized",
        "obsolete two-seed linearized estimand": "two-seedhigh-budgetmean",
        "obsolete linearized limitation": "linearizedgridandk=3movementauditusetwo",
        "old BAR-L2 aggregate row": "bar-l2&2&69.02&36.68&27.44&29/58",
        "old BAR-L3 aggregate row": "bar-l3&2&68.54&43.91&33.12&24/53",
        "obsolete precision caveat": "limitedseed-levelprecision",
        "ambiguous audit scope": "270-checkpointk=3",
        "overclaimed target measurement": "directlymeasuredtarget-policy",
        "overclaimed final route result": "alleightfinal-checkpointcomparisons",
        "overclaimed final route shorthand": "8/8final-checkpoint",
        "invalid actor-path evidence": "actor-path",
        "invalid semigroup-defect evidence": "semigroupdefect",
    }
    stale = [label for label, token in stale_tokens.items() if token in text]
    if re.search(r"35/63.{0,80}9/63", text):
        stale.append("old two-seed collapse endpoints")
    if re.search(
        r"\b(?:halfcheetah|hopper|walker2d)\s*-\s*me\b",
        uncommented,
        flags=re.IGNORECASE,
    ):
        stale.append("obsolete ME dataset abbreviation")
    if stale:
        raise AssertionError(f"stale contradictory manuscript claims present: {stale}")
    print(
        "PASS manuscript source: normalized uncommented four-seed proximal and "
        "linearized claims, audits, and limitations are current"
    )


def main() -> None:
    if not __debug__:
        raise RuntimeError("verification requires assertions; do not run with Python -O")
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=repo / "sweep_results")
    parser.add_argument("--manuscript-dir", type=Path, default=repo / "aistats26_manuscript")
    args = parser.parse_args()
    claims = EXPECTED_CLAIMS
    verify_complete_grid(args.results_dir, claims)
    verify_host_separated_pairs(args.results_dir, claims)
    verify_stability_claims(args.results_dir, claims)
    verify_controls(args.results_dir / "diagnostics")
    verify_diagnostics(args.results_dir / "diagnostics", claims)
    verify_figure_inputs(args.results_dir / "diagnostics")
    verify_k4(args.results_dir)
    verify_manuscript(args.manuscript_dir)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
