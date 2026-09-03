#!/usr/bin/env python3
"""Verify released PART results and the manuscript-facing numerical claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np

if __package__:
    from scripts.diagnostics.verify_p1_compact_release import (
        verify_p1_target_value_compact,
    )
else:
    from diagnostics.verify_p1_compact_release import verify_p1_target_value_compact

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
PRACTICAL_EXPECTED = {
    "fixed_1_5": {
        "td3": 74.78503888888889,
        "prox2": 78.04966944444445,
        "prox3": 78.95350555555555,
        "prox4": 77.7007361111111,
        "lin2": 75.12858888888888,
        "lin3": 77.73804166666667,
    },
    "grid_best": {
        "td3": (74.78503888888889, 1.5),
        "prox2": (80.65999722222222, 2.5),
        "prox3": (81.40987222222222, 2.5),
        "prox4": (82.06428055555556, 4.0),
        "lin2": (78.42564444444444, 2.5),
        "lin3": (78.20023611111111, 2.5),
    },
    "family_transfer": {
        "selected_t": {
            "hopper": {"td3": 1.5, "prox4": 2.5},
            "halfcheetah": {"td3": 1.5, "prox4": 7.0},
            "walker2d": {"td3": 1.5, "prox4": 4.0},
        },
        "fold_differences": {
            "hopper": 17.076475000000002,
            "halfcheetah": -8.938799999999993,
            "walker2d": 2.826158333333325,
        },
        "td3_mean": 74.78503888888889,
        "prox4_mean": 78.43965,
        "mean_difference": 3.654611111111109,
        "positive_tasks": 7,
    },
    "hierarchical95": {
        "prox4_minus_prox3_high": (3.0449487003968296, 22.7271641170635),
        "lin3_minus_lin2_high": (2.9018722023809533, 14.0653185515873),
        "prox4_minus_td3_fixed_1_5": (-0.2018227083333348, 8.455759027777766),
    },
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

ACTOR_COST_ARCHIVE_SHA256 = {
    "K=1.json": "d520365702caa5bb05034a2da837306fd8f3de36006fba67647f52e3e1b243e8",
    "K=2.json": "66bbe71b9e8f0553a1026fb17f2bff06ba756f7f34330043552568242155b79f",
    "K=3.json": "20dcdc16d002bb9140aef352812f2778f8ab4cd51500db11a48e9df5b6bb77ac",
    "K=4.json": "beb5ebd77808b147f9d556dab3aef8436baccd6c49903e663fec29cba91ec12a",
    "MANIFEST.json": "edaa1efeff7ca037124af8a12f2ded88b424d6c76132a9e0ddfe94fcc41e203a",
    "RUN_CONTEXT.json": "ca853bcda35732a00826149e3f74dcde68518ddee478f20a55d0f059eef833b0",
}
ACTOR_COST_EXPECTED = {
    1: (1.4345667500000001, 1.0, 9_684_736, 1.0),
    2: (1.5424702, 1.0752167509807402, 14_680_064, 1.5157939256165578),
    3: (1.6856721499999998, 1.1750391886609666, 15_627_520, 1.6136237477201236),
    4: (1.88734705, 1.3156216328030745, 18_045_440, 1.8632867225291427),
}
ACTOR_COST_SOURCE_REVISION = "481f7fe0786230db8b4106e44349be241a18414d"
ACTOR_COST_DATASET_SHA256 = "5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0"
ACTOR_COST_MEMORY_SCOPE = (
    "fresh worker process-lifetime backend high-water mark through dataset/batch "
    "transfer, actor/critic state initialization, lowering/compile, warmup, and "
    "timed trials; not an isolated steady-state actor-call peak"
)


P2_COMPACT_SCHEMA = "p2-relu-residence-compact-v1"
P2_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
P2_FINAL_PROTOCOL = "p2_relu_residence_final_v2"
P2_FINAL_RAW_STATUS = "analysis_complete_pending_verification"
P2_FINAL_VERIFIED_STATUS = "verified_complete"
P2_FINAL_ENVIRONMENTS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
)
P2_FINAL_SEEDS = (0, 1)
P2_FINAL_T = (0.025, 0.05, 0.1, 0.2)
P2_FINAL_K = (1, 2, 4, 8, 16)
P2_FINAL_N_STATES = 512
P2_FINAL_CATEGORIES = (
    "resident",
    "relu_cross",
    "box_cross",
    "tie_cross",
    "boundary_ambiguous",
    "boundary_touch",
    "nonfinite",
)
P2_CELL_FIELDS = (
    "environment",
    "seed",
    "T",
    "K",
    "h",
    "n_states",
    *(f"full_{name}" for name in P2_FINAL_CATEGORIES),
    *(f"step_{name}" for name in P2_FINAL_CATEGORIES),
    "full_residence_fraction",
    "step_residence_fraction",
    "full_affine_q_max_abs_residual",
)


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


def verify_practical_summaries(root: Path) -> None:
    source: dict[str, dict[tuple[float, int, str], float]] = {}
    for method, (hops, integrator, seeds) in METHODS.items():
        envs, values = load_sweep(root, hops, integrator, seeds)
        assert tuple(envs) == ENVIRONMENTS and seeds == (0, 1, 2, 3)
        source[method] = values

        fixed = statistics.mean(
            values[(1.5, seed, env)] for seed in seeds for env in envs
        )
        close(fixed, PRACTICAL_EXPECTED["fixed_1_5"][method], 1e-12)
        tau_means = {
            tau: statistics.mean(
                values[(tau, seed, env)] for seed in seeds for env in envs
            )
            for tau in GRID
        }
        best_tau = max(GRID, key=tau_means.__getitem__)
        expected_best, expected_tau = PRACTICAL_EXPECTED["grid_best"][method]
        close(tau_means[best_tau], expected_best, 1e-12)
        close(best_tau, expected_tau, 1e-12)

    families = {
        "hopper": tuple(env for env in ENVIRONMENTS if env.startswith("hopper-")),
        "halfcheetah": tuple(env for env in ENVIRONMENTS if env.startswith("halfcheetah-")),
        "walker2d": tuple(env for env in ENVIRONMENTS if env.startswith("walker2d-")),
    }
    expected_transfer = PRACTICAL_EXPECTED["family_transfer"]
    held_scores: dict[str, dict[str, float]] = {"td3": {}, "prox4": {}}
    for family, test_envs in families.items():
        train_envs = tuple(env for env in ENVIRONMENTS if env not in test_envs)
        for method in ("td3", "prox4"):
            values = source[method]
            train_means = {
                tau: statistics.mean(
                    values[(tau, seed, env)]
                    for seed in (0, 1, 2, 3)
                    for env in train_envs
                )
                for tau in GRID
            }
            selected_tau = max(GRID, key=train_means.__getitem__)
            close(
                selected_tau,
                expected_transfer["selected_t"][family][method],
                1e-12,
            )
            for env in test_envs:
                held_scores[method][env] = statistics.mean(
                    values[(selected_tau, seed, env)] for seed in (0, 1, 2, 3)
                )
        fold_difference = statistics.mean(
            held_scores["prox4"][env] - held_scores["td3"][env]
            for env in test_envs
        )
        close(fold_difference, expected_transfer["fold_differences"][family], 1e-12)

    td3_held = statistics.mean(held_scores["td3"].values())
    prox4_held = statistics.mean(held_scores["prox4"].values())
    close(td3_held, expected_transfer["td3_mean"], 1e-12)
    close(prox4_held, expected_transfer["prox4_mean"], 1e-12)
    close(prox4_held - td3_held, expected_transfer["mean_difference"], 1e-12)
    assert sum(
        held_scores["prox4"][env] > held_scores["td3"][env]
        for env in ENVIRONMENTS
    ) == expected_transfer["positive_tasks"]

    def hierarchical_interval(
        method_a: str,
        method_b: str,
        taus: tuple[float, ...],
    ) -> tuple[float, float]:
        aa = np.asarray(
            [
                [
                    statistics.mean(source[method_a][(tau, seed, env)] for tau in taus)
                    for seed in (0, 1, 2, 3)
                ]
                for env in ENVIRONMENTS
            ],
            dtype=float,
        )
        bb = np.asarray(
            [
                [
                    statistics.mean(source[method_b][(tau, seed, env)] for tau in taus)
                    for seed in (0, 1, 2, 3)
                ]
                for env in ENVIRONMENTS
            ],
            dtype=float,
        )
        rng = np.random.default_rng(20260902)
        task_indices = rng.integers(0, len(ENVIRONMENTS), size=(200_000, len(ENVIRONMENTS)))
        bootstrap = np.zeros(200_000, dtype=float)
        for occurrence in range(len(ENVIRONMENTS)):
            task = task_indices[:, occurrence]
            seed_a = rng.integers(0, 4, size=(200_000, 4))
            seed_b = rng.integers(0, 4, size=(200_000, 4))
            mean_a = np.take_along_axis(aa[task], seed_a, axis=1).mean(axis=1)
            mean_b = np.take_along_axis(bb[task], seed_b, axis=1).mean(axis=1)
            bootstrap += mean_a - mean_b
        bootstrap /= len(ENVIRONMENTS)
        interval = np.percentile(bootstrap, (2.5, 97.5))
        return float(interval[0]), float(interval[1])

    hierarchical = {
        "prox4_minus_prox3_high": hierarchical_interval("prox4", "prox3", HIGH),
        "lin3_minus_lin2_high": hierarchical_interval("lin3", "lin2", HIGH),
        "prox4_minus_td3_fixed_1_5": hierarchical_interval("prox4", "td3", (1.5,)),
    }
    for label, interval in hierarchical.items():
        expected_interval = PRACTICAL_EXPECTED["hierarchical95"][label]
        close(interval[0], expected_interval[0], 1e-10)
        close(interval[1], expected_interval[1], 1e-10)

    print(
        "PASS practical summaries: fixed/grid-best scores, leave-one-family-out "
        "budget transfer, and task-then-independent-seed intervals"
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


def verify_mcep_control(diag: Path, source_root: Path | None = None) -> None:
    bundle = diag / "bar_mcep_p3_paired"
    rows = csv_rows(
        bundle / "paired_final_scores.csv",
        {
            "environment",
            "tau",
            "seed",
            "bar_step",
            "bar_score",
            "mcep_step",
            "mcep_score",
            "delta_mcep_minus_bar",
            "bar_collapsed_lt20",
            "mcep_collapsed_lt20",
            "transition",
            "mcep_execution_class",
            "mcep_resume_step",
        },
    )
    expected_keys = set(product(ENVIRONMENTS, AUDIT_TAUS, (0, 1)))
    keys = {
        (row["environment"], finite(row["tau"], "MCEP tau"), int(row["seed"]))
        for row in rows
    }
    if len(rows) != 90 or len(keys) != len(rows) or keys != expected_keys:
        raise AssertionError("MCEP control must contain the exact 9 x 5 x 2 paired grid")

    bar_values: list[float] = []
    mcep_values: list[float] = []
    deltas: list[float] = []
    recovery_counts = {"gpu_designated": 0, "gpu_to_cpu_resume": 0, "cpu_only": 0}
    transitions = {
        "mcep_rescue": 0,
        "mcep_reversal": 0,
        "both_collapsed": 0,
        "both_stable": 0,
    }
    resume_steps = {
        (4.0, 0): 854528,
        (4.0, 1): 853376,
        (7.0, 0): 667392,
        (7.0, 1): 573504,
        (10.0, 0): 557632,
        (10.0, 1): 545344,
        (14.0, 0): 528000,
        (14.0, 1): 512000,
    }
    by_key: dict[tuple[str, float, int], tuple[float, float, float]] = {}
    for row in rows:
        env = row["environment"]
        tau = finite(row["tau"], "MCEP tau")
        seed = int(row["seed"])
        if int(row["bar_step"]) != 1_000_000 or int(row["mcep_step"]) != 1_000_000:
            raise AssertionError("MCEP control scores must come from the 1M checkpoint")
        bar = finite(row["bar_score"], "BAR-P3 score")
        mcep = finite(row["mcep_score"], "MCEP-inspired score")
        delta = finite(row["delta_mcep_minus_bar"], "MCEP minus BAR delta")
        close(delta, mcep - bar, 1e-12)
        bar_flag = row["bar_collapsed_lt20"] == "true"
        mcep_flag = row["mcep_collapsed_lt20"] == "true"
        if row["bar_collapsed_lt20"] not in {"true", "false"}:
            raise AssertionError("invalid BAR collapse flag")
        if row["mcep_collapsed_lt20"] not in {"true", "false"}:
            raise AssertionError("invalid MCEP collapse flag")
        assert bar_flag == (bar < 20)
        assert mcep_flag == (mcep < 20)
        expected_transition = (
            "mcep_rescue"
            if bar_flag and not mcep_flag
            else "mcep_reversal"
            if not bar_flag and mcep_flag
            else "both_collapsed"
            if bar_flag
            else "both_stable"
        )
        if row["transition"] != expected_transition:
            raise AssertionError(f"bad MCEP transition for {(env, tau, seed)}")
        transitions[expected_transition] += 1

        if env != "walker2d-expert-v2":
            expected_execution, expected_resume = "gpu_designated", ""
        elif tau == 20:
            expected_execution, expected_resume = "cpu_only", "0"
        else:
            expected_execution = "gpu_to_cpu_resume"
            expected_resume = str(resume_steps[(tau, seed)])
        if (
            row["mcep_execution_class"] != expected_execution
            or row["mcep_resume_step"] != expected_resume
        ):
            raise AssertionError(f"bad MCEP recovery classification for {(env, tau, seed)}")
        recovery_counts[expected_execution] += 1
        bar_values.append(bar)
        mcep_values.append(mcep)
        deltas.append(delta)
        by_key[(env, tau, seed)] = (bar, mcep, delta)

    summary = json.loads((bundle / "SUMMARY.json").read_text())
    primary = summary["primary"]
    close(statistics.mean(bar_values), float(primary["bar_mean"]), 1e-12)
    close(statistics.mean(mcep_values), float(primary["mcep_mean"]), 1e-12)
    close(statistics.mean(deltas), float(primary["mean_delta_mcep_minus_bar"]), 1e-12)
    close(statistics.median(deltas), float(primary["paired_median_delta_mcep_minus_bar"]), 1e-12)
    assert sum(delta > 0 for delta in deltas) == int(primary["mcep_wins"]) == 34
    assert sum(delta < 0 for delta in deltas) == int(primary["bar_wins"]) == 52
    assert sum(delta == 0 for delta in deltas) == int(primary["exact_ties"]) == 4

    task_means = [
        statistics.mean(by_key[(env, tau, seed)][2] for tau in AUDIT_TAUS for seed in (0, 1))
        for env in ENVIRONMENTS
    ]
    assert sum(value > 0 for value in task_means) == int(primary["positive_task_means"]) == 2
    interval = task_bootstrap_interval(task_means, np.random.default_rng(20260830))
    close(interval[0], float(primary["task_bootstrap_95"][0]), 1e-12)
    close(interval[1], float(primary["task_bootstrap_95"][1]), 1e-12)
    close(interval[0], -1.5186630416597202, 1e-12)
    close(interval[1], 5.443334193810723, 1e-12)

    _, released = load_sweep(diag.parent, "K=3", "Imp", (0, 1))
    rerun_differences = [
        by_key[(env, tau, seed)][0] - released[(tau, seed, env)]
        for env in ENVIRONMENTS
        for tau in AUDIT_TAUS
        for seed in (0, 1)
    ]
    sensitivity = summary["contemporaneous_bar_sensitivity"]
    released_values = [
        released[(tau, seed, env)]
        for env in ENVIRONMENTS
        for tau in AUDIT_TAUS
        for seed in (0, 1)
    ]
    close(statistics.mean(released_values), float(sensitivity["released_bar_p3_mean"]), 1e-12)
    close(
        statistics.mean(bar_values),
        float(sensitivity["contemporaneous_bar_p3_mean"]),
        1e-12,
    )
    close(
        statistics.mean(rerun_differences),
        float(sensitivity["mean_difference_contemporaneous_minus_released"]),
        1e-12,
    )
    close(
        math.sqrt(statistics.mean(value * value for value in rerun_differences)),
        float(sensitivity["cellwise_rmse"]),
        1e-12,
    )
    close(min(rerun_differences), float(sensitivity["cellwise_difference_min"]), 1e-12)
    close(max(rerun_differences), float(sensitivity["cellwise_difference_max"]), 1e-12)

    diagnostics = summary["diagnostics"]
    assert sum(score < 20 for score in bar_values) == diagnostics["raw_collapse_lt20"]["bar"] == 23
    assert sum(score < 20 for score in mcep_values) == diagnostics["raw_collapse_lt20"]["mcep"] == 21
    assert transitions == {
        "mcep_rescue": 5,
        "mcep_reversal": 3,
        "both_collapsed": 18,
        "both_stable": 64,
    }
    recorded_transitions = diagnostics["raw_collapse_lt20"]
    assert recorded_transitions["mcep_rescues"] == transitions["mcep_rescue"]
    assert recorded_transitions["mcep_reversals"] == transitions["mcep_reversal"]
    assert recorded_transitions["both_collapsed"] == transitions["both_collapsed"]
    assert recorded_transitions["both_stable"] == transitions["both_stable"]

    seed_mean_collapses = {}
    for method_index, method in enumerate(("bar", "mcep")):
        seed_mean_collapses[method] = sum(
            statistics.mean(by_key[(env, tau, seed)][method_index] for seed in (0, 1)) < 20
            for env in ENVIRONMENTS
            for tau in AUDIT_TAUS
        )
    assert seed_mean_collapses == {"bar": 9, "mcep": 9}
    assert diagnostics["seed_mean_collapse_lt20"] == {"bar": 9, "mcep": 9, "pairs": 45}

    opposite = sum(
        by_key[(env, tau, 0)][2] * by_key[(env, tau, 1)][2] < 0
        for env in ENVIRONMENTS
        for tau in AUDIT_TAUS
    )
    assert opposite == diagnostics["opposite_seed_signs"] == 30
    ordered = sorted(deltas)
    close(statistics.mean(ordered[9:-9]), float(diagnostics["trimmed_mean_delta_10_percent"]), 1e-12)
    non_recovery = [
        delta
        for (env, _tau, _seed), (_bar, _mcep, delta) in by_key.items()
        if env != "walker2d-expert-v2"
    ]
    close(
        statistics.mean(non_recovery),
        float(diagnostics["mean_delta_excluding_walker2d_expert"]),
        1e-12,
    )
    assert recovery_counts == {
        "gpu_designated": 80,
        "gpu_to_cpu_resume": 8,
        "cpu_only": 2,
    }

    manifest = csv_rows(
        bundle / "SOURCE_MANIFEST.csv",
        {
            "method",
            "environment",
            "tau",
            "seed",
            "execution_class",
            "resume_step",
            "config_path",
            "config_bytes",
            "config_sha256",
            "eval_path",
            "eval_bytes",
            "eval_sha256",
            "checkpoint_path",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "primary_log_path",
            "primary_log_bytes",
            "primary_log_sha256",
            "recovery_log_path",
            "recovery_log_bytes",
            "recovery_log_sha256",
        },
    )
    manifest_keys = {
        (row["method"], row["environment"], finite(row["tau"], "manifest tau"), int(row["seed"]))
        for row in manifest
    }
    expected_manifest = set(product(("bar", "mcep"), ENVIRONMENTS, AUDIT_TAUS, (0, 1)))
    if len(manifest) != 180 or manifest_keys != expected_manifest:
        raise AssertionError("MCEP source manifest must contain the exact 180 method-cell rows")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    for row in manifest:
        for prefix in ("config", "eval", "checkpoint"):
            source_path = Path(row[f"{prefix}_path"])
            if source_path.is_absolute() or ".." in source_path.parts:
                raise AssertionError(f"non-relative source path: {source_path}")
            if int(row[f"{prefix}_bytes"]) <= 0 or not sha_pattern.fullmatch(row[f"{prefix}_sha256"]):
                raise AssertionError(f"invalid {prefix} provenance for {source_path}")
        for prefix in ("primary_log", "recovery_log"):
            value = row[f"{prefix}_path"]
            if not value:
                assert not row[f"{prefix}_bytes"] and not row[f"{prefix}_sha256"]
                continue
            source_path = Path(value)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise AssertionError(f"non-relative log path: {source_path}")
            if int(row[f"{prefix}_bytes"]) <= 0 or not sha_pattern.fullmatch(row[f"{prefix}_sha256"]):
                raise AssertionError(f"invalid {prefix} provenance for {source_path}")

    if source_root is not None:
        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        for row in manifest:
            for prefix in ("config", "eval", "checkpoint", "primary_log", "recovery_log"):
                relative = row[f"{prefix}_path"]
                if not relative:
                    continue
                raw = source_root / relative
                if not raw.is_file():
                    raise AssertionError(f"missing local MCEP source artifact: {raw}")
                assert raw.stat().st_size == int(row[f"{prefix}_bytes"])
                if sha256(raw) != row[f"{prefix}_sha256"]:
                    raise AssertionError(f"MCEP source digest mismatch: {raw}")
        print("PASS MCEP local sources: all 180 configs/evals/checkpoints and logs match")
    print(
        "PASS MCEP mechanism control: exact paired grid, recomputed statistics, "
        "recovery classes, and source manifest"
    )


def verify_actor_cost_archive(diag: Path) -> None:
    """Verify the compact, hash-pinned actor-cost release without raw dependencies."""
    cost_dir = diag / "actor_cost"
    expected_files = set(ACTOR_COST_ARCHIVE_SHA256)
    actual_json = {path.name for path in cost_dir.glob("*.json")}
    assert actual_json == expected_files
    for name, expected_digest in ACTOR_COST_ARCHIVE_SHA256.items():
        assert sha256_file(cost_dir / name) == expected_digest

    context = json.loads((cost_dir / "RUN_CONTEXT.json").read_text())
    manifest = json.loads((cost_dir / "MANIFEST.json").read_text())
    assert context["schema_version"] == manifest["schema_version"] == "bar-actor-cost-v1"
    assert manifest["artifact"] == "actor_cost_manifest"
    assert manifest["status"] == "measured"
    assert manifest["accelerator_evidence"] is True
    assert manifest["exact_k_set"] == context["orchestrator"]["exact_k_set"] == [1, 2, 3, 4]
    assert manifest["only_swept_field"] == "k"
    assert context["orchestrator"]["fresh_process_per_k"] is True
    assert context["orchestrator"]["sequential_workers"] is True
    assert context["git"]["dirty"] is False
    assert context["git"]["revision"] == ACTOR_COST_SOURCE_REVISION
    assert manifest["git"] == context["git"]
    assert context["dataset"]["sha256"] == ACTOR_COST_DATASET_SHA256
    assert manifest["dataset"]["sha256"] == ACTOR_COST_DATASET_SHA256
    assert manifest["run_context"]["sha256"] == ACTOR_COST_ARCHIVE_SHA256["RUN_CONTEXT.json"]
    assert manifest["hashes"]["run_context_sha256"] == ACTOR_COST_ARCHIVE_SHA256["RUN_CONTEXT.json"]
    assert manifest["memory_scope"] == ACTOR_COST_MEMORY_SCOPE
    assert manifest["device"]["platform"] == "gpu"
    assert manifest["device"]["device_kind"] == "NVIDIA H200"
    assert manifest["selected_driver_record"]["name"] == "NVIDIA H200"
    assert manifest["device_binding"]["method"] == "resolved_uuid_visibility"
    assert manifest["worker_environment"]["JAX_PLATFORMS"] == "cuda"
    assert manifest["worker_environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert manifest["packages"]["jax"] == manifest["packages"]["jaxlib"] == "0.10.2"

    common = context["common_config"]
    assert manifest["common_config"] == common
    assert common["platform"] == "gpu"
    assert common["memory_fallback"] == "none"
    assert common["batch_size"] == 256
    assert common["tau"] == 12.0
    assert common["trials"] == common["calls_per_trial"] == 10
    assert common["warmup_calls"] == 3

    rows = {int(row["k"]): row for row in manifest["results"]}
    assert set(rows) == set(ACTOR_COST_EXPECTED)
    pids: set[int] = set()
    process_tokens: set[str] = set()
    k1_time, _, k1_peak, _ = ACTOR_COST_EXPECTED[1]
    invariant_hash = manifest["invariant_config_sha256"]
    source_hash = manifest["source"]["sha256"]

    for k, (expected_ms, expected_time_ratio, expected_peak, expected_peak_ratio) in (
        ACTOR_COST_EXPECTED.items()
    ):
        result_path = cost_dir / f"K={k}.json"
        raw = json.loads(result_path.read_text())
        row = rows[k]
        assert raw["schema_version"] == "bar-actor-cost-v1"
        assert raw["artifact"] == "actor_cost_worker_result"
        assert int(raw["k"]) == int(raw["config"]["k"]) == k
        config_without_k = dict(raw["config"])
        del config_without_k["k"]
        assert config_without_k == common
        assert raw["hashes"]["run_context_sha256"] == ACTOR_COST_ARCHIVE_SHA256["RUN_CONTEXT.json"]
        assert raw["hashes"]["dataset_sha256"] == ACTOR_COST_DATASET_SHA256
        assert raw["hashes"]["invariant_config_sha256"] == invariant_hash
        assert raw["hashes"]["code_sha256"] == source_hash
        assert raw["provenance"]["git"]["dirty"] is False
        assert raw["provenance"]["git"]["revision"] == ACTOR_COST_SOURCE_REVISION
        assert raw["provenance"]["visible_device_count"] == 1
        assert raw["provenance"]["device"]["device_kind"] == "NVIDIA H200"
        assert raw["actor_update"]["scope"] == "actor_only_no_critic_update"
        assert raw["actor_update"]["fixed_input_state_each_call"] is True

        call_rows = raw["timing"]["raw_call_ms"]
        assert len(call_rows) == common["trials"]
        assert all(len(calls) == common["calls_per_trial"] for calls in call_rows)
        trial_means = [statistics.fmean(map(float, calls)) for calls in call_rows]
        recorded_trials = list(map(float, raw["timing"]["raw_trial_ms"]))
        assert len(recorded_trials) == len(trial_means)
        assert all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(recorded_trials, trial_means, strict=True)
        )
        recomputed_median = statistics.median(trial_means)
        assert math.isclose(
            float(raw["timing"]["median_actor_update_ms"]),
            recomputed_median,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        memory = raw["memory"]
        assert memory["is_accelerator_memory"] is True
        assert memory["fallback_selected"] == "none"
        assert memory["source"] == "device.memory_stats"
        assert memory["peak_counter_scope"] == ACTOR_COST_MEMORY_SCOPE
        assert int(memory["baseline_adjusted_peak_bytes"]) == max(
            0,
            int(memory["absolute_peak_bytes"]) - int(memory["baseline_current_bytes"]),
        )

        assert row["result_file"] == result_path.name
        assert row["result_sha256"] == ACTOR_COST_ARCHIVE_SHA256[result_path.name]
        assert row["raw_trial_ms"] == raw["timing"]["raw_trial_ms"]
        assert math.isclose(float(row["median_actor_update_ms"]), recomputed_median)
        assert int(row["absolute_peak_bytes"]) == int(memory["absolute_peak_bytes"])
        assert int(row["baseline_adjusted_peak_bytes"]) == int(
            memory["baseline_adjusted_peak_bytes"]
        )

        close(recomputed_median, expected_ms, tol=1e-12)
        close(float(row["time_ratio_to_k1"]), expected_time_ratio, tol=1e-12)
        close(float(row["absolute_peak_ratio_to_k1"]), expected_peak_ratio, tol=1e-12)
        assert int(row["absolute_peak_bytes"]) == expected_peak
        close(recomputed_median / k1_time, expected_time_ratio, tol=1e-12)
        close(int(row["absolute_peak_bytes"]) / k1_peak, expected_peak_ratio, tol=1e-12)

        pids.add(int(raw["process"]["pid"]))
        process_tokens.add(str(raw["process"]["process_token"]))

    assert len(pids) == len(process_tokens) == 4
    print(
        "PASS actor cost: hash-pinned K=1..4 H200 archive, recomputed timing, "
        "fresh workers, and scoped accelerator high-water memory"
    )



def verify_p2_relu_residence_compact(diag: Path) -> None:
    """Independently rebuild the compact P2-v2 result from its 12 cell tables."""
    compact_dir = diag / "p2_relu_residence_final"
    compact_path = compact_dir / "COMPACT_MANIFEST.json"
    if not compact_path.is_file():
        raise FileNotFoundError(compact_path)

    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    expected_cells = {
        f"{environment}_seed{seed}/residence_cells.csv"
        for environment in P2_FINAL_ENVIRONMENTS
        for seed in P2_FINAL_SEEDS
    }
    expected_payload = {
        "FINAL_DESIGN_LOCK.json",
        "MANIFEST.json",
        "STATUS.json",
        "SUMMARY.json",
        "VERIFY.json",
        "README.md",
        *expected_cells,
    }
    inventory = compact.get("inventory")
    if not isinstance(inventory, dict) or set(inventory) != expected_payload:
        raise AssertionError("P2 compact payload inventory is not exact")
    actual_files = {
        path.relative_to(compact_dir).as_posix()
        for path in compact_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != {*expected_payload, "COMPACT_MANIFEST.json"}:
        raise AssertionError("P2 compact archive contains a missing or extra file")
    for relative, digest in inventory.items():
        if sha256_file(compact_dir / relative) != digest:
            raise AssertionError(f"P2 compact payload hash mismatch: {relative}")
    if compact.get("payload_tree_sha256") != canonical_sha256(inventory):
        raise AssertionError("P2 compact payload-tree hash mismatch")
    if (
        compact.get("schema_version") != P2_COMPACT_SCHEMA
        or compact.get("artifact") != "p2_relu_residence_final_compact"
        or compact.get("protocol") != P2_FINAL_PROTOCOL
        or compact.get("status") != "verified_compact"
        or compact.get("scientific_admissible") is not True
    ):
        raise AssertionError("P2 compact manifest identity/status mismatch")
    expected_grid = {
        "environments": list(P2_FINAL_ENVIRONMENTS),
        "seeds": list(P2_FINAL_SEEDS),
        "T": list(P2_FINAL_T),
        "K": list(P2_FINAL_K),
        "runs": 12,
        "cells_per_run": 20,
        "states_per_run": P2_FINAL_N_STATES,
        "anchors_total": 12 * P2_FINAL_N_STATES,
    }
    if compact.get("grid") != expected_grid:
        raise AssertionError("P2 compact grid metadata mismatch")

    design = json.loads((compact_dir / "FINAL_DESIGN_LOCK.json").read_text())
    raw_manifest = json.loads((compact_dir / "MANIFEST.json").read_text())
    status = json.loads((compact_dir / "STATUS.json").read_text())
    summary = json.loads((compact_dir / "SUMMARY.json").read_text())
    receipt = json.loads((compact_dir / "VERIFY.json").read_text())
    for label, document in (
        ("design", design),
        ("manifest", raw_manifest),
        ("status", status),
        ("summary", summary),
        ("verify", receipt),
    ):
        if document.get("protocol") != P2_FINAL_PROTOCOL:
            raise AssertionError(f"P2 compact {label} protocol mismatch")
    for label, document in (
        ("manifest", raw_manifest),
        ("status", status),
        ("summary", summary),
    ):
        if (
            document.get("status") != P2_FINAL_RAW_STATUS
            or document.get("scientific_admissible") is not False
            or document.get("scientific_admissible_when_verified") is not True
        ):
            raise AssertionError(f"P2 compact {label} bypasses the raw verification gate")

    design_without_hash = dict(design)
    claimed_design_hash = design_without_hash.pop("design_sha256", None)
    if claimed_design_hash != canonical_sha256(design_without_hash):
        raise AssertionError("P2 compact design self-hash mismatch")
    if raw_manifest.get("design_sha256") != claimed_design_hash:
        raise AssertionError("P2 compact raw-manifest design hash mismatch")
    if receipt.get("design_sha256") != claimed_design_hash:
        raise AssertionError("P2 compact VERIFY design hash mismatch")
    if receipt.get("manifest_sha256") != sha256_file(compact_dir / "MANIFEST.json"):
        raise AssertionError("P2 compact VERIFY does not bind MANIFEST.json")

    provenance = raw_manifest.get("git_provenance")
    if not isinstance(provenance, dict):
        raise AssertionError("P2 compact Git provenance missing")
    if (
        provenance.get("git_dirty") is not False
        or provenance.get("git_tracked_dirty") is not False
        or provenance.get("head_matches_origin_main") is not True
        or provenance.get("git_revision") != provenance.get("origin_main_revision")
        or not isinstance(provenance.get("git_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", provenance["git_revision"]) is None
        or provenance.get("git_status_porcelain_sha256") != P2_EMPTY_SHA256
        or provenance.get("git_tracked_status_porcelain_sha256") != P2_EMPTY_SHA256
        or provenance.get("source_snapshot_is_durable_provenance") is not True
    ):
        raise AssertionError("P2 compact run did not record clean origin/main provenance")
    if compact.get("source_git") != {
        "revision": provenance["git_revision"],
        "origin_main_revision": provenance["origin_main_revision"],
        "recorded_clean": True,
    }:
        raise AssertionError("P2 compact source-Git summary mismatch")
    runtime = raw_manifest.get("runtime", {})
    if runtime.get("backend") != "cpu" or runtime.get("jax_enable_x64") is not True:
        raise AssertionError("P2 compact raw runtime is not CPU float64")

    source_hashes = raw_manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or design.get("source_sha256") != source_hashes:
        raise AssertionError("P2 compact design/source snapshot hashes differ")
    expected_source_names = {
        "run_p2_relu_residence.py",
        "verify_p2_relu_residence.py",
        "verify_p2_relu_residence_final.py",
    }
    if set(source_hashes) != expected_source_names:
        raise AssertionError("P2 compact source inventory mismatch")
    snapshots = raw_manifest.get("source_snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != expected_source_names:
        raise AssertionError("P2 compact source-snapshot inventory mismatch")
    repo = diag.parents[1]
    for name, digest in source_hashes.items():
        if snapshots[name] != {
            "path": f"SOURCE_SNAPSHOT/{name}",
            "sha256": digest,
        }:
            raise AssertionError(f"P2 compact source-snapshot record mismatch: {name}")
        current = repo / "scripts" / "diagnostics" / name
        if not current.is_file() or sha256_file(current) != digest:
            raise AssertionError(f"P2 compact released source differs: {name}")

    required_receipt_gates = (
        "independent_geometry_recompute_pass",
        "finite_difference_pass",
        "exit_bracket_pass",
        "pooled_survival_pass",
        "task_survival_pass",
        "task_equal_survival_pass",
        "aggregate_boundary_categories_pass",
        "family_survival_pass",
    )
    if (
        receipt.get("status") != P2_FINAL_VERIFIED_STATUS
        or receipt.get("pass") is not True
        or receipt.get("scientific_admissible") is not True
        or any(receipt.get(field) is not True for field in required_receipt_gates)
        or int(receipt.get("n_runs", -1)) != 12
        or int(receipt.get("n_states_per_run", -1)) != P2_FINAL_N_STATES
        or int(receipt.get("n_anchors_total", -1)) != 12 * P2_FINAL_N_STATES
        or receipt.get("verifier_sha256")
        != source_hashes["verify_p2_relu_residence_final.py"]
    ):
        raise AssertionError("P2 compact raw VERIFY receipt did not pass every gate")
    if compact.get("raw_verification") != {
        "status": receipt["status"],
        "pass": receipt["pass"],
        "scientific_admissible": receipt["scientific_admissible"],
        "design_sha256": receipt["design_sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "verifier_sha256": receipt["verifier_sha256"],
    }:
        raise AssertionError("P2 compact raw-verification summary mismatch")

    if design.get("grid") != {"T": list(P2_FINAL_T), "K": list(P2_FINAL_K)}:
        raise AssertionError("P2 compact design T/K grid mismatch")
    if (
        design.get("environments") != list(P2_FINAL_ENVIRONMENTS)
        or design.get("seeds") != list(P2_FINAL_SEEDS)
        or int(design.get("n_runs", -1)) != 12
        or int(design.get("n_states_per_run", -1)) != P2_FINAL_N_STATES
        or design.get("coverage_gate") is not None
        or design.get("learned_slope") is not None
    ):
        raise AssertionError("P2 compact design scope/count mismatch")
    if raw_manifest.get("counts") != {
        "runs": 12,
        "states_per_run": P2_FINAL_N_STATES,
        "anchors_total": 12 * P2_FINAL_N_STATES,
    }:
        raise AssertionError("P2 compact raw-manifest counts mismatch")
    if (
        summary.get("environments") != list(P2_FINAL_ENVIRONMENTS)
        or summary.get("seeds") != list(P2_FINAL_SEEDS)
        or int(summary.get("n_runs", -1)) != 12
        or int(summary.get("n_states_per_run", -1)) != P2_FINAL_N_STATES
        or int(summary.get("n_anchors_total", -1)) != 12 * P2_FINAL_N_STATES
        or summary.get("coverage_gate") is not None
        or summary.get("learned_slope") is not None
    ):
        raise AssertionError("P2 compact SUMMARY scope/count mismatch")

    expected_horizons_decimal = sorted(
        {
            Decimal(str(total)) / Decimal(k)
            for total in P2_FINAL_T
            for k in P2_FINAL_K
        }
    )
    expected_horizons = [float(value) for value in expected_horizons_decimal]
    if len(expected_horizons) != 8:
        raise AssertionError("internal P2 unique-horizon contract drift")

    def category_counts(row: dict[str, str], prefix: str, label: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in P2_FINAL_CATEGORIES:
            raw = row[f"{prefix}{name}"]
            try:
                value = int(raw)
            except ValueError as error:
                raise AssertionError(f"{label}: noninteger category {raw!r}") from error
            if value < 0:
                raise AssertionError(f"{label}: negative category count")
            counts[name] = value
        if sum(counts.values()) != P2_FINAL_N_STATES:
            raise AssertionError(f"{label}: category denominator mismatch")
        return counts

    run_curves: dict[tuple[str, int], list[dict[str, object]]] = {}
    raw_artifacts = raw_manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise AssertionError("P2 compact raw artifact inventory missing")
    for copied in ("FINAL_DESIGN_LOCK.json", "SUMMARY.json"):
        if raw_artifacts.get(copied) != sha256_file(compact_dir / copied):
            raise AssertionError(f"P2 compact raw artifact hash mismatch: {copied}")

    summary_rows = summary.get("per_run")
    if not isinstance(summary_rows, list):
        raise AssertionError("P2 compact SUMMARY per_run is missing")
    summary_by_run = {
        (row.get("environment"), int(row.get("seed", -1))): row
        for row in summary_rows
        if isinstance(row, dict)
    }
    expected_run_keys = {
        (environment, seed)
        for environment in P2_FINAL_ENVIRONMENTS
        for seed in P2_FINAL_SEEDS
    }
    if len(summary_rows) != 12 or set(summary_by_run) != expected_run_keys:
        raise AssertionError("P2 compact SUMMARY per-run key grid mismatch")

    expected_cell_order = [
        (Decimal(str(total)), k) for total in P2_FINAL_T for k in P2_FINAL_K
    ]
    for environment in P2_FINAL_ENVIRONMENTS:
        for seed in P2_FINAL_SEEDS:
            relative = f"{environment}_seed{seed}/residence_cells.csv"
            path = compact_dir / relative
            if raw_artifacts.get(relative) != sha256_file(path):
                raise AssertionError(f"P2 compact raw cell hash mismatch: {relative}")
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != P2_CELL_FIELDS:
                    raise AssertionError(f"{relative}: exact CSV schema mismatch")
                rows = list(reader)
            if len(rows) != 20:
                raise AssertionError(f"{relative}: expected 20 cells")
            actual_order = [(Decimal(row["T"]), int(row["K"])) for row in rows]
            if actual_order != expected_cell_order or len(set(actual_order)) != 20:
                raise AssertionError(f"{relative}: exact ordered T/K grid mismatch")

            full_aliases: dict[Decimal, tuple[dict[str, int], float, float | None]] = {}
            step_aliases: dict[Decimal, tuple[dict[str, int], float]] = {}
            for row in rows:
                label = f"{relative}: T={row['T']}, K={row['K']}"
                if row["environment"] != environment or int(row["seed"]) != seed:
                    raise AssertionError(f"{label}: run identity mismatch")
                total = Decimal(row["T"])
                k = int(row["K"])
                horizon = Decimal(row["h"])
                if horizon != total / Decimal(k):
                    raise AssertionError(f"{label}: h != T/K")
                if int(row["n_states"]) != P2_FINAL_N_STATES:
                    raise AssertionError(f"{label}: n_states mismatch")
                full_counts = category_counts(row, "full_", label)
                step_counts = category_counts(row, "step_", label)
                full_fraction = finite(row["full_residence_fraction"], f"{label}: full fraction")
                step_fraction = finite(row["step_residence_fraction"], f"{label}: step fraction")
                if not math.isclose(
                    full_fraction,
                    full_counts["resident"] / P2_FINAL_N_STATES,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ) or not math.isclose(
                    step_fraction,
                    step_counts["resident"] / P2_FINAL_N_STATES,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise AssertionError(f"{label}: residence fraction/count mismatch")
                residual_text = row["full_affine_q_max_abs_residual"].strip()
                residual = (
                    None
                    if not residual_text
                    else finite(residual_text, f"{label}: residual")
                )
                if residual is not None and residual < 0.0:
                    raise AssertionError(f"{label}: negative affine residual")

                full_record = (full_counts, full_fraction, residual)
                if total in full_aliases and full_aliases[total] != full_record:
                    raise AssertionError(f"{relative}: full-horizon result changed with K")
                full_aliases[total] = full_record
                step_record = (step_counts, step_fraction)
                if horizon in step_aliases and step_aliases[horizon] != step_record:
                    raise AssertionError(f"{relative}: repeated T/K horizon aliases differ")
                step_aliases[horizon] = step_record
                if k == 1 and (
                    full_counts != step_counts
                    or not math.isclose(full_fraction, step_fraction, abs_tol=1e-12)
                ):
                    raise AssertionError(f"{relative}: K=1 full and step views differ")

            if set(full_aliases) != {Decimal(str(value)) for value in P2_FINAL_T}:
                raise AssertionError(f"{relative}: full-horizon aliases incomplete")
            if set(step_aliases) != set(expected_horizons_decimal):
                raise AssertionError(f"{relative}: unique step horizons incomplete")
            curve = [
                {
                    "horizon": float(horizon),
                    "n_resident": step_aliases[horizon][0]["resident"],
                    "resident_fraction": step_aliases[horizon][1],
                    "categories": step_aliases[horizon][0],
                }
                for horizon in expected_horizons_decimal
            ]
            run_curves[(environment, seed)] = curve

            stored = summary_by_run[(environment, seed)]
            if (
                stored.get("protocol") != P2_FINAL_PROTOCOL
                or stored.get("status") != "run_complete_pending_verification"
                or stored.get("scientific_admissible") is not False
                or stored.get("scientific_admissible_when_verified") is not True
                or int(stored.get("n_states", -1)) != P2_FINAL_N_STATES
                or int(stored.get("n_cells", -1)) != 20
            ):
                raise AssertionError(f"{relative}: stored run status/count mismatch")
            expected_full = {
                str(float(total)): full_aliases[total][0]
                for total in sorted(full_aliases)
            }
            if stored.get("full_category_counts_by_T") != expected_full:
                raise AssertionError(f"{relative}: full-category summary mismatch")
            for field in ("C_ref", "analytic_vs_jax_grad_max_abs"):
                value = finite(stored.get(field), f"{relative}: {field}")
                if value < 0.0 or (field == "C_ref" and value == 0.0):
                    raise AssertionError(f"{relative}: invalid {field}")

            actual_curve = stored.get("survival")
            if not isinstance(actual_curve, list) or len(actual_curve) != len(curve):
                raise AssertionError(f"{relative}: run survival length mismatch")
            for expected, actual in zip(curve, actual_curve, strict=True):
                if set(actual) != {
                    "horizon",
                    "n_resident",
                    "resident_fraction",
                    "categories",
                }:
                    raise AssertionError(f"{relative}: run survival schema mismatch")
                if (
                    not math.isclose(
                        float(actual["horizon"]),
                        expected["horizon"],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or int(actual["n_resident"]) != expected["n_resident"]
                    or not math.isclose(
                        float(actual["resident_fraction"]),
                        expected["resident_fraction"],
                        abs_tol=1e-12,
                    )
                    or set(actual["categories"]) != set(P2_FINAL_CATEGORIES)
                    or {name: int(actual["categories"][name]) for name in P2_FINAL_CATEGORIES}
                    != expected["categories"]
                ):
                    raise AssertionError(f"{relative}: run survival mismatch")

    def aggregate(keys: list[tuple[str, int]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, horizon in enumerate(expected_horizons):
            categories = {
                name: sum(int(run_curves[key][index]["categories"][name]) for key in keys)
                for name in P2_FINAL_CATEGORIES
            }
            n_total = len(keys) * P2_FINAL_N_STATES
            rows.append(
                {
                    "horizon": horizon,
                    "n_resident": categories["resident"],
                    "n_total": n_total,
                    "resident_fraction": categories["resident"] / n_total,
                    "categories": categories,
                }
            )
        return rows

    def assert_aggregate(actual: object, expected: list[dict[str, object]], label: str) -> None:
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{label}: aggregate length mismatch")
        required = {"horizon", "n_resident", "n_total", "resident_fraction", "categories"}
        for wanted, observed in zip(expected, actual, strict=True):
            if not isinstance(observed, dict) or set(observed) != required:
                raise AssertionError(f"{label}: aggregate schema mismatch")
            if (
                not math.isclose(
                    float(observed["horizon"]),
                    float(wanted["horizon"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or int(observed["n_resident"]) != wanted["n_resident"]
                or int(observed["n_total"]) != wanted["n_total"]
                or not math.isclose(
                    float(observed["resident_fraction"]),
                    float(wanted["resident_fraction"]),
                    abs_tol=1e-12,
                )
                or set(observed["categories"]) != set(P2_FINAL_CATEGORIES)
                or {name: int(observed["categories"][name]) for name in P2_FINAL_CATEGORIES}
                != wanted["categories"]
            ):
                raise AssertionError(f"{label}: aggregate values mismatch")

    if [float(value) for value in summary.get("horizons", [])] != expected_horizons:
        raise AssertionError("P2 compact SUMMARY unique horizons mismatch")
    all_keys = sorted(expected_run_keys)
    assert_aggregate(summary.get("pooled_survival"), aggregate(all_keys), "P2 pooled")

    task_expected = {
        environment: aggregate([(environment, seed) for seed in P2_FINAL_SEEDS])
        for environment in P2_FINAL_ENVIRONMENTS
    }
    if set(summary.get("task_survival", {})) != set(P2_FINAL_ENVIRONMENTS):
        raise AssertionError("P2 compact task-survival key grid mismatch")
    for environment, expected in task_expected.items():
        assert_aggregate(summary["task_survival"][environment], expected, f"P2 task/{environment}")

    family_expected = {
        family: aggregate(
            [
                key
                for key in all_keys
                if key[0].split("-", 1)[0] == family
            ]
        )
        for family in ("hopper", "walker2d")
    }
    if set(summary.get("family_survival", {})) != set(family_expected):
        raise AssertionError("P2 compact family-survival key grid mismatch")
    for family, expected in family_expected.items():
        assert_aggregate(summary["family_survival"][family], expected, f"P2 family/{family}")

    task_equal_expected = []
    for index, horizon in enumerate(expected_horizons):
        task_equal_expected.append(
            {
                "horizon": horizon,
                "n_tasks": len(P2_FINAL_ENVIRONMENTS),
                "resident_fraction": statistics.fmean(
                    task_expected[environment][index]["resident_fraction"]
                    for environment in P2_FINAL_ENVIRONMENTS
                ),
                "category_fractions": {
                    name: statistics.fmean(
                        task_expected[environment][index]["categories"][name]
                        / task_expected[environment][index]["n_total"]
                        for environment in P2_FINAL_ENVIRONMENTS
                    )
                    for name in P2_FINAL_CATEGORIES
                },
            }
        )
    actual_task_equal = summary.get("task_equal_survival")
    if not isinstance(actual_task_equal, list) or len(actual_task_equal) != len(
        task_equal_expected
    ):
        raise AssertionError("P2 compact task-equal length mismatch")
    required_task_equal = {
        "horizon",
        "n_tasks",
        "resident_fraction",
        "category_fractions",
    }
    for wanted, observed in zip(task_equal_expected, actual_task_equal, strict=True):
        if not isinstance(observed, dict) or set(observed) != required_task_equal:
            raise AssertionError("P2 compact task-equal schema mismatch")
        if (
            not math.isclose(float(observed["horizon"]), wanted["horizon"], abs_tol=1e-12)
            or int(observed["n_tasks"]) != wanted["n_tasks"]
            or not math.isclose(
                float(observed["resident_fraction"]),
                wanted["resident_fraction"],
                abs_tol=1e-12,
            )
        ):
            raise AssertionError("P2 compact task-equal values mismatch")
        fractions = observed.get("category_fractions", {})
        if set(fractions) != set(P2_FINAL_CATEGORIES):
            raise AssertionError("P2 compact task-equal category schema mismatch")
        for name in P2_FINAL_CATEGORIES:
            if not math.isclose(
                float(fractions[name]),
                wanted["category_fractions"][name],
                abs_tol=1e-12,
            ):
                raise AssertionError(f"P2 compact task-equal category mismatch: {name}")
        if not math.isclose(sum(map(float, fractions.values())), 1.0, abs_tol=1e-12):
            raise AssertionError("P2 compact task-equal category fractions do not sum to one")

    print(
        "PASS P2 compact: exact 12x20 cell archive, category denominators, "
        "horizon aliases, clean provenance, raw VERIFY, and all aggregates"
    )

def verify_retired_fixed_operator_archive(diag: Path) -> None:
    order_dir = diag / "fixed_operator_order"
    status = json.loads((order_dir / "STATUS.json").read_text())
    harness = json.loads((order_dir / "HARNESS.json").read_text())
    assert harness["pass"] is True
    slopes = harness["quadratic_Q"]["slopes_K4_8_16"]
    assert all(0.9 <= float(slopes[scheme]) <= 1.1 for scheme in ("explicit", "implicit"))

    if not status["checkpoints_ready"]:
        unresolved = json.loads(
            (order_dir / "CHECKPOINTS_UNRESOLVED.json").read_text()
        )
        assert unresolved["n_expected"] == 18
        assert unresolved["n_resolved"] == status["n_resolved"] < 18
        assert len(unresolved["entries"]) == len(
            {
                (entry["environment"], int(entry["seed"]))
                for entry in unresolved["entries"]
            }
        ) == 18
        draft = json.loads((order_dir / "PROTOCOL_DRAFT.json").read_text())
        assert draft["locked"] is False
        assert draft["checkpoint_inventory_status"] == "unresolved"
        assert draft["checkpoint_inventory_sha256"] == sha256_file(
            order_dir / "CHECKPOINTS_UNRESOLVED.json"
        )
        assert draft["harness_sha256"] == sha256_file(order_dir / "HARNESS.json")
        assert draft["code_sha256"] == sha256_file(
            diag.parents[1]
            / "scripts"
            / "diagnostics"
            / "run_fixed_operator_order.py"
        )
        assert not (order_dir / "FROZEN_PROTOCOL.json").exists()
        for name in (
            "endpoint_errors.csv",
            "raw_state_diagnostics.npz",
            "implicit_substeps.npz",
            "euler_paths.npz",
            "SUMMARY.json",
            "RESULT_MANIFEST.json",
        ):
            assert not (order_dir / name).exists()
        print(
            "PASS retired fixed-operator scaffold: archive integrity valid; "
            f"snapshot resolved {status['n_resolved']}/18 and has no learned result"
        )
        return

    checkpoint_path = order_dir / "CHECKPOINTS.json"
    checkpoints = json.loads(checkpoint_path.read_text())
    assert checkpoints["n_expected"] == checkpoints["n_resolved"] == 18
    checkpoint_keys = {
        (entry["environment"], int(entry["seed"]))
        for entry in checkpoints["entries"]
    }
    assert len(checkpoint_keys) == len(checkpoints["entries"]) == 18
    for entry in checkpoints["entries"]:
        assert entry["resolved"] is True
        for key in (
            "weights_sha256",
            "config_sha256",
            "dataset_sha256",
            "normalization_statistics_sha256",
            "state_indices_sha256",
        ):
            assert re.fullmatch(r"[0-9a-f]{64}", entry[key])

    protocol = json.loads((order_dir / "FROZEN_PROTOCOL.json").read_text())
    assert protocol["locked"] is True
    assert protocol["grid"]["T"] == [0.025, 0.05, 0.1, 0.2]
    assert protocol["grid"]["K"] == [1, 2, 4, 8, 16]
    assert set(protocol["grid"]["schemes"]) == {"explicit", "implicit"}
    assert protocol["grid"]["n_states"] == 512
    assert protocol["rk4"]["N_candidates"] == [256, 512, 1024, 2048, 4096]
    assert protocol["rk4"]["maximum_accepted_microsteps"] == 8192
    assert protocol["checkpoint_inventory_sha256"] == sha256_file(checkpoint_path)
    assert protocol["expected_counts"] == {
        "endpoint_cells": 720,
        "raw_state_rows": 368640,
        "implicit_substep_records": 1142784,
    }

    required_endpoint_fields = {
        "environment",
        "seed",
        "T",
        "K",
        "scheme",
        "E_abs",
        "E_rel",
        "M_ref",
        "ratio_to_2K",
        "slope_K4_8_16",
        "slope_status",
        "|I|",
        "primary_cell",
        "reference_stable",
        "failure_reason",
        "C_ref",
    }
    rows = csv_rows(order_dir / "endpoint_errors.csv", required_endpoint_fields)
    endpoint_keys = {
        (
            row["environment"],
            int(row["seed"]),
            float(row["T"]),
            int(row["K"]),
            row["scheme"],
        )
        for row in rows
    }
    expected_keys = set(
        product(
            ENVIRONMENTS,
            (0, 1),
            (0.025, 0.05, 0.1, 0.2),
            (1, 2, 4, 8, 16),
            ("explicit", "implicit"),
        )
    )
    assert len(rows) == len(endpoint_keys) == 720
    assert endpoint_keys == expected_keys

    grouped = {
        (
            row["environment"],
            int(row["seed"]),
            float(row["T"]),
            row["scheme"],
            int(row["K"]),
        ): row
        for row in rows
    }
    for key, row in grouped.items():
        k = key[-1]
        if k not in (1, 2, 4, 8):
            continue
        left = float(row["E_abs"])
        right = float(grouped[(*key[:-1], 2 * k)]["E_abs"])
        ratio = row["ratio_to_2K"]
        if math.isfinite(left) and math.isfinite(right) and right > 0:
            close(float(ratio), left / right, 1e-10)

    raw = np.load(order_dir / "raw_state_diagnostics.npz")
    implicit = np.load(order_dir / "implicit_substeps.npz")
    paths = np.load(order_dir / "euler_paths.npz")
    masks = np.load(order_dir / "masks.npz")
    assert len(raw["cell_index"]) == 368640
    assert len(implicit["run_index"]) == 1142784
    assert len(paths["run_index"]) == 2285568
    assert masks["common_mask"].shape == (72, 512)

    summary_path = order_dir / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    assert summary["n_runs"] == 18
    assert summary["n_endpoint_rows"] == summary["n_unique_endpoint_keys"] == 720
    assert summary["n_raw_state_rows"] == 368640
    assert summary["n_implicit_substep_records"] == 1142784
    assert summary["n_euler_path_records"] == 2285568
    assert len(summary["runs"]) == len(
        {(run["environment"], int(run["seed"])) for run in summary["runs"]}
    ) == 18
    for scheme in ("explicit", "implicit"):
        gate = summary[scheme]
        assert len(gate["two_seed_task_means"]) <= 9
        expected_support = (
            gate["aggregate_median_slope"] is not None
            and 0.75 <= float(gate["aggregate_median_slope"]) <= 1.25
            and int(gate["n_in_[0.5,1.5]"]) >= 14
            and int(gate["n_complete_runs"]) >= 15
        )
        assert gate["pre_specified_support"] is expected_support

    result_manifest = json.loads(
        (order_dir / "RESULT_MANIFEST.json").read_text()
    )
    assert result_manifest["protocol_sha256"] == protocol["protocol_sha256"]
    assert result_manifest["checkpoint_inventory_sha256"] == sha256_file(
        checkpoint_path
    )
    for name, expected_hash in result_manifest["artifacts"].items():
        assert sha256_file(order_dir / name) == expected_hash
    print(
        "PASS retired fixed-operator archive: exact grids/counts/fingerprints, "
        "ratios, masks, diagnostics, and gates agree; provenance only"
    )


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
    assert bar_mcep["bar"]["root"] == "results/bar_p3"
    assert bar_mcep["mcep"]["root"] == "results/mcep_p3"
    assert bar_mcep["manuscript_status"].endswith(
        "AUDIT.json is completeness/config only"
    )

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
        "four-seed linearized L2 row": "part-l2&75.13&78.43(2.5)&38.43&25/112",
        "four-seed linearized L3 row": "part-l3&77.74&78.20(2.5)&46.44&21/95",
        "fixed-budget practical contrast": "part-p3andp4score78.95and77.70versus74.79",
        "descriptive grid maxima": "part-p4reaches82.06att=4",
        "family-transfer summary": "p4averages78.44versus74.79fortd3+bc(+3.65;7/9",
        "hierarchical P4-P3 sensitivity": "intervals[3.04,22.73]and[2.90,14.07]",
        "hierarchical fixed-budget sensitivity": "interval[-.20,8.46]",
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
        "final proxy count": "37/90",
        "P4 final proxy count": "44/90",
        "P1 common-critic counts": "lowerin86/90and88/90cells",
        "P1 all-task direction": "allninetaskmediansbelowone",
        "P1 exposure-reach split": (
            "thefinalactorliesfartherfromthepairednextdatasetactionthanthefirstpolyakactor"
        ),
        "P1 interpretation boundary": "notcriticaccuracyorreturncausality",
        "exposure proxy definition": (
            "bootstrapexposuredenotesthistarget-action-displacementproxy,"
            "notdistancetodatasetsupport,criticerror,orbellman-targeterror"
        ),
        "metric quotient boundary": (
            "pseudometriconpointwisepolicymapsandametricafteridentifying"
            "mapsequalrho-almosteverywhere"
        ),
        "live algorithm order": (
            "polyak-updatemu_1^-andthetargetcriticswithrate.005"
        ),
        "live local-limit boundary": (
            "thislimitdoesnotexplainthecoupledlarge-tchain"
        ),
        "ReLU region identity": (
            "withinsucharegion,theidealexplicitupdateandthesame-region"
            "stationarybackward-eulerbranchcoincideexactly"
        ),
        "ReLU learned-audit boundary": (
            "ratherthanfittinganonzero1/kerrorslope"
        ),
        "calibrated abstract mechanism": (
            "matchedevidenceisconsistentwithexposure--reachseparation"
        ),
        "evidence map": "evidencemap",
        "actor-cost main endpoints": (
            "compiledactor-phasetimerisesfrom1.435to1.887ms"
        ),
        "actor-cost K4 row": "4&1.887347&1.316&18,045,440&1.863",
        "actor-cost evidence scope": "actorcost&k=1--4;h200",
        "actor-cost interpretation boundary": (
            "notanend-to-endtraining-timeorhardware-independentmemorylaw"
        ),
        "evidence map P3 movement scope": (
            "p3target/finaldisplacement&270checkpoints;90matchedcontrasts"
        ),
        "evidence map P3 control scope": (
            "p3two-actorcontrol&90pairs;180finalscores"
        ),
        "P4 audit scope": "fullseed-0/1finalgeometryshowsnotypicalfinalcontraction",
        "P2 clean residence values": (
            "residenceover6144eligibleinterioranchorsis.954,.911,.842,.748"
        ),
        "P2 task heterogeneity": "the.2-horizontaskrangeis.310--.980",
        "restricted exposure sensitivity": "36/40cells",
        "restricted exposure median ratio": "medianratio.905",
        "compressed route qualification": (
            "final-routeerrorislargerinallfour-checkpointaggregates"
            "butonlyfiveofeightfinalcheckpoints"
        ),
        "stable simulator median": "-41.4",
        "collapsed simulator median": "+1.70times10^12",
        "bootstrap protocol": "100,000drawsusinganalysisseed20260830",
        "audit scope": "theoriginaldirectauditcovers270methodcheckpoints",
        "manifest limitation": "exactper-runmanifestswerenotretained",
        "provenance qualification": (
            "provenanceconsistsofthearchivedlaunchconfigurationandscorematrices"
        ),
        "MCEP control mechanism conclusion": (
            "policyseparationattainstheobservedp3aggregateregimewithout"
            "sequentialre-centering"
        ),
        "MCEP main table PART row": "part-p3rerun&57.67&23/90&9/45",
        "MCEP main table control row": "two-actorcontrol&59.22&21/90&9/45",
        "PART title": (
            "part:proximalactorrefinementthroughtargetroutinginoffline"
            "reinforcementlearning"
        ),
        "P4 control evidence scope": (
            "p4two-actorscores&90pairs;180finalscores&0--1"
        ),
        "completed P4 control contrast": (
            "part-p4minuscontrolhastask-equalmean-2.31"
        ),
        "completed P4 control interval": "task-resamplinginterval[-7.25,2.41]",
        "completed P4 outcome": "thepredeclaredoutcomeisunresolved",
        "P0 within-pair stack boundary": (
            "everymatchedpairwastrainedandevaluatedwithinonehoststack"
        ),
        "P0 data-validity statement": (
            "theexistingscoresarenottreatedasabsentorinvalid"
        ),
        "MCEP control contrast": (
            "controlminuspartis+1.56withtask-resamplinginterval[-1.52,5.44]"
        ),
        "MCEP paired median and wins": "pairedmedianis-.42andpartwins52/90pairs",
        "MCEP scope boundary": "notpublished-mcepreproductions",
        "MCEP recovery": (
            "eightwalker2d-expertcellsresumedfromthematchinglogged"
            "emergency-checkpointsteps"
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
        "old PART-L2 aggregate row": "part-l2&2&69.02&36.68&27.44&29/58",
        "old PART-L3 aggregate row": "part-l3&2&68.54&43.91&33.12&24/53",
        "obsolete precision caveat": "limitedseed-levelprecision",
        "ambiguous audit scope": "270-checkpointk=3",
        "overclaimed target measurement": "directlymeasuredtarget-policy",
        "overclaimed final route result": "alleightfinal-checkpointcomparisons",
        "overclaimed final route shorthand": "8/8final-checkpoint",
        "invalid actor-path evidence": "actor-path",
        "invalid semigroup-defect evidence": "semigroupdefect",
        "overstrong abstract mechanism": (
            "resultsestablishtarget/deploymentseparation"
        ),
        "MCEP equivalence overclaim": "statisticallyindistinguishable",
        "MCEP equivalence shorthand": "equivalentperformance",
        "ambiguous four-seed P4 comparison": (
            "four-seedk=4auditlowerstargetdisplacementin89/90matchedseed-0/1"
        ),
        "old final proxy count": "36/90",
        "old P3 final ratio": "medianratio1.057",
        "old P4 missing-final boundary": "archiveomitsp4final-actorgeometry",
        "old missing P4 control": "nocorrespondingp4controlhasyettested",
        "retired frozen-critic result": "15of18environment--seedruns",
        "stale P4 invalidity claim": "failsthelockedprotocol",
        "stale P4 inadmissible label": "descriptive;inadmissible",
        "stale P4 rerun prerequisite": (
            "protocol-admissiblep4controlremainsneeded"
        ),
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
    parser.add_argument(
        "--mcep-source-root",
        type=Path,
        help="Optional local repository root for checking uncommitted raw-run hashes",
    )
    parser.add_argument(
        "--require-p2-compact",
        action="store_true",
        help="Fail unless the verified final-v2 P2 compact archive is present",
    )
    args = parser.parse_args()
    claims = EXPECTED_CLAIMS
    verify_complete_grid(args.results_dir, claims)
    verify_practical_summaries(args.results_dir)
    verify_host_separated_pairs(args.results_dir, claims)
    verify_stability_claims(args.results_dir, claims)
    verify_controls(args.results_dir / "diagnostics")
    verify_mcep_control(
        args.results_dir / "diagnostics",
        args.mcep_source_root,
    )
    verify_actor_cost_archive(args.results_dir / "diagnostics")
    verify_p1_target_value_compact(args.results_dir / "diagnostics")
    p2_compact = (
        args.results_dir
        / "diagnostics"
        / "p2_relu_residence_final"
        / "COMPACT_MANIFEST.json"
    )
    if p2_compact.is_file():
        verify_p2_relu_residence_compact(args.results_dir / "diagnostics")
    elif args.require_p2_compact:
        raise FileNotFoundError(p2_compact)
    else:
        print("SKIP P2 compact: no verified final-v2 COMPACT_MANIFEST.json")
    verify_retired_fixed_operator_archive(args.results_dir / "diagnostics")
    verify_diagnostics(args.results_dir / "diagnostics", claims)
    verify_figure_inputs(args.results_dir / "diagnostics")
    verify_k4(args.results_dir)
    verify_manuscript(args.manuscript_dir)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
