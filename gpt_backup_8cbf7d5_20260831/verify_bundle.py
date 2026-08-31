#!/usr/bin/env python3
"""Verify the AAAI-26 GPT manuscript bundle and its released-result claims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_STYLE_GIT_SHA1 = "989b761198b6d142da3372a06e032fd8979972e3"
GRID = (0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 10.0, 12.0, 14.0, 17.0, 20.0)
HIGH = tuple(t for t in GRID if t >= 4.0)
METHODS = {
    "TD3+BC": ("K=1", "Imp", (0, 1, 2, 3)),
    "BAR-Prox K=2": ("K=2", "Imp", (0, 1, 2, 3)),
    "BAR-Prox K=3": ("K=3", "Imp", (0, 1, 2, 3)),
    "BAR-Prox K=4": ("K=4", "Imp", (0, 1, 2, 3)),
    "BAR-Lin K=2": ("K=2", "Exp", (0, 1)),
    "BAR-Lin K=3": ("K=3", "Exp", (0, 1)),
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def close(actual: float, expected: float, tol: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise VerificationError(f"{message}: observed={actual}, expected={expected}, tol={tol}")


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"empty CSV: {path}")
    return rows


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_style_and_source() -> None:
    style = ROOT / "aaai2026.sty"
    require(style.is_file(), "aaai2026.sty is missing")
    require(git_blob_sha1(style) == EXPECTED_STYLE_GIT_SHA1,
            "aaai2026.sty differs from the official AAAI-26 author kit")
    style_text = style.read_text(encoding="utf-8")
    require("\\ProvidesPackage{aaai2026}[2026/06/17 AAAI 2026 Submission format]" in style_text,
            "unexpected AAAI style package declaration")

    forbidden_packages = {
        "hyperref", "geometry", "balance", "flushend", "fontenc", "fullpage",
        "setspace", "stfloats", "titlesec", "ulem", "wrapfig", "authblk",
    }
    usepackage = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}")
    for name in ("paper.tex", "supplement.tex"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        require("\\usepackage[submission]{aaai2026}" in text,
                f"{name} does not use anonymous AAAI-26 submission style")
        loaded: set[str] = set()
        for match in usepackage.finditer(text):
            loaded.update(part.strip() for part in match.group(1).split(","))
        bad = sorted(loaded & forbidden_packages)
        require(not bad, f"{name} loads forbidden AAAI packages: {bad}")
        require("medium-expert" not in text, f"{name} contains medium-expert, which is outside this experiment")
        require("-ME" not in text and "\\texttt{ME}" not in text,
                f"{name} contains stale ME abbreviation")
        require("directly measured target-policy displacement" not in text,
                f"{name} contains an overstated target-displacement phrase")
        require("required controlled rerun" not in text,
                f"{name} retains the withdrawn common-initialization requirement")

    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "supplement.tex").read_text(encoding="utf-8")
    checklist_path = ROOT / "ReproducibilityChecklist.tex"
    require(checklist_path.is_file(), "official reproducibility checklist source is missing")
    checklist = checklist_path.read_text(encoding="utf-8")
    require("\\input{ReproducibilityChecklist}" in paper,
            "paper.tex does not input the official reproducibility checklist")
    require("% The questions start here" in checklist,
            "reproducibility checklist question marker is missing")
    responses = checklist.split("% The questions start here", 1)[1]
    require(len(re.findall(r"^\s*\\question", responses, re.MULTILINE)) == 31,
            "reproducibility checklist does not contain all 31 official questions")
    require("Type your response here" not in responses,
            "reproducibility checklist has unanswered response slots")
    require(len(re.findall(r"^(?:yes|no|partial|NA)$", responses, re.MULTILINE)) == 31,
            "reproducibility checklist does not contain exactly 31 valid answers")
    joined = paper + "\n" + supplement
    required = (
        "sample-anchored next-state target-action displacement",
        "locally Lipschitz",
        "does not certify it",
        "action-metric-matched projected linearization",
        "in only five of eight final-checkpoint comparisons",
        "Identical critic initializations across $K$ are not required",
        "expert-v2",
        "$34/63$ to $8/63$",
        "$[21.39,54.63]$",
        "$[3.84,21.11]$",
        r"$T\in\{4,7,10,14,20\}$",
        "270 method checkpoints",
        "90 matched environment--budget--seed cells per method",
        "nominal TD3+BC integration-time parameter",
        "two disjoint, host-separated seed pairs",
        "100,000 draws",
        "does not measure P4 target movement",
        "official AAAI-26 checklist with all 31 responses",
    )
    missing = [token for token in required if token not in joined]
    require(not missing, f"required revised claims/qualifications missing: {missing}")
    require("every $5{,}000$" not in joined and "every 5,000" not in joined,
            "stale 5,000-update historical evaluation claim remains")
    blanket_50k = (
        "every $50{,}000$ critic updates",
        "every 50,000 critic updates",
        "evaluated every $50{,}000$",
        "evaluated every 50,000",
    )
    require(not any(token in joined for token in blanket_50k),
            "blanket every-50,000 evaluation-cadence claim remains")
    require("where $\n" not in paper and "reference $\n" not in paper,
            "paper contains an accidental newline inside a LaTeX command")
    require("C_{1,t}^{\\mathrm P}" in joined and "C_{1,t}^{\\mathrm L}" in joined,
            "implementation-specific first-hop normalization equations are missing")
    require("C_{k,t}^{\\mathrm P}" in joined and "C_{k,t}^{\\mathrm L}" in joined,
            "implementation-specific later-hop normalization equations are missing")
    print("PASS style/source: AAAI format, P0 corrections, and claim wording")


def load_sweep(
    results_dir: Path,
    hops: str,
    realization: str,
    seeds: tuple[int, ...],
) -> tuple[list[str], dict[tuple[float, int, str], float]]:
    values: dict[tuple[float, int, str], float] = {}
    envs: list[str] | None = None
    for seed in seeds:
        path = results_dir / hops / realization / f"seed{seed}.csv"
        rows = read_csv(path)
        current = [key for key in rows[0].keys() if key != "tau"]
        if envs is None:
            envs = current
        require(current == envs, f"environment columns differ across seeds: {hops}/{realization}")
        for row in rows:
            tau = float(row["tau"])
            if tau not in GRID:
                continue
            for env in envs:
                raw = row[env].strip()
                require(raw not in ("", "—"), f"missing complete-grid cell: {hops}/{realization}/{tau}/{seed}/{env}")
                key = (tau, seed, env)
                require(key not in values, f"duplicate sweep cell: {hops}/{realization}/{key}")
                values[key] = float(raw)
    assert envs is not None
    require(len(envs) == 9, f"expected 9 environments, found {len(envs)}")
    require(len(values) == len(GRID) * len(seeds) * len(envs),
            f"incomplete grid: {hops}/{realization} has {len(values)} values")
    return envs, values


def bundled_summary() -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv(ROOT / "data" / "aggregate_summary.csv")}


def bundled_budget_means() -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_csv(ROOT / "data" / "budget_means.csv")}


def bundled_environment_means() -> dict[str, dict[str, str]]:
    return {row["environment"]: row for row in read_csv(ROOT / "data" / "environment_high_means.csv")}


def verify_complete_sweeps(results_dir: Path) -> None:
    expected_summary = bundled_summary()
    expected_budget = bundled_budget_means()
    expected_env = bundled_environment_means()
    claims = json.loads((ROOT / "data" / "audit_claims.json").read_text(encoding="utf-8"))
    stability = claims["stability"]
    uncertainty = {
        row["contrast"]: row
        for row in read_csv(ROOT / "data" / "task_cluster_uncertainty.csv")
    }
    source: dict[
        str,
        tuple[list[str], dict[tuple[float, int, str], float], tuple[int, ...]],
    ] = {}

    reference_envs: list[str] | None = None
    for method, (hops, realization, seeds) in METHODS.items():
        envs, values = load_sweep(results_dir, hops, realization, seeds)
        if reference_envs is None:
            reference_envs = envs
        require(envs == reference_envs, f"environment order differs for {method}")
        source[method] = (envs, values, seeds)

        per_budget = {
            tau: statistics.mean(values[(tau, seed, env)] for seed in seeds for env in envs)
            for tau in GRID
        }
        low = statistics.mean(v for tau, v in per_budget.items() if tau <= 2.5)
        high = statistics.mean(v for tau, v in per_budget.items() if tau >= 4.0)
        tail = statistics.mean(v for tau, v in per_budget.items() if tau >= 10.0)
        cell_scores = [
            statistics.mean(values[(tau, seed, env)] for seed in seeds)
            for tau in HIGH for env in envs
        ]
        raw_scores = [
            values[(tau, seed, env)]
            for tau in HIGH for seed in seeds for env in envs
        ]
        require(method in expected_summary, f"aggregate summary missing method {method}")
        require(method in expected_budget, f"budget summary missing method {method}")
        exp = expected_summary[method]
        require(int(exp["seeds"]) == len(seeds), f"{method} seed-count mismatch")
        require(int(exp["total_high_budget_runs"]) == len(raw_scores),
                f"{method} high-budget denominator mismatch")
        close(low, float(exp["mean_T_le_2.5"]), 0.0000015, f"{method} low mean")
        close(high, float(exp["mean_T_ge_4"]), 0.0000015, f"{method} high mean")
        close(tail, float(exp["mean_T_ge_10"]), 0.0000015, f"{method} tail mean")
        require(sum(v < 20 for v in cell_scores) == int(exp["collapsed_cells_lt20_of63"]),
                f"{method} cell collapse count mismatch")
        require(sum(v < 20 for v in raw_scores) == int(exp["collapsed_runs_lt20"]),
                f"{method} raw collapse count mismatch")
        for tau, value in per_budget.items():
            close(value, float(expected_budget[method][f"{tau:g}"]), 0.0000015,
                  f"{method} budget mean at T={tau:g}")

    assert reference_envs is not None
    label_map = {
        "hopper-medium-v2": "Hopper-M",
        "hopper-medium-replay-v2": "Hopper-MR",
        "hopper-expert-v2": "Hopper-E",
        "halfcheetah-medium-v2": "HalfCheetah-M",
        "halfcheetah-medium-replay-v2": "HalfCheetah-MR",
        "halfcheetah-expert-v2": "HalfCheetah-E",
        "walker2d-medium-v2": "Walker2d-M",
        "walker2d-medium-replay-v2": "Walker2d-MR",
        "walker2d-expert-v2": "Walker2d-E",
    }
    for env in reference_envs:
        label = label_map[env]
        require(label in expected_env, f"environment summary missing {label}")
        for method, (_, values, seeds) in source.items():
            observed = statistics.mean(values[(tau, seed, env)] for tau in HIGH for seed in seeds)
            close(observed, float(expected_env[label][method]), 0.0000015,
                  f"{method} high mean for {label}")

    def high_task_mean(method: str, env: str, seeds: tuple[int, ...] | None = None) -> float:
        _, values, method_seeds = source[method]
        selected = method_seeds if seeds is None else seeds
        require(set(selected).issubset(method_seeds), f"invalid seed subset for {method}: {selected}")
        return statistics.mean(values[(tau, seed, env)] for tau in HIGH for seed in selected)

    # Recompute the four-seed task-score contrasts. A single seeded generator is
    # advanced in table order so the released percentile intervals are reproducible.
    score_specs = (
        ("BAR-Prox K=4 minus TD3+BC", "TD3+BC", "p4_vs_td3"),
        ("BAR-Prox K=4 minus BAR-Prox K=3", "BAR-Prox K=3", "p4_vs_p3"),
    )
    rng = np.random.default_rng(20260830)
    for contrast, baseline, claim_key in score_specs:
        data = np.asarray([
            high_task_mean("BAR-Prox K=4", env) - high_task_mean(baseline, env)
            for env in reference_envs
        ], dtype=float)
        claim = stability[claim_key]
        row = uncertainty[contrast]
        estimate = float(data.mean())
        close(estimate, float(claim["mean_difference"]), 1e-12,
              f"{contrast} audit-claim mean")
        close(estimate, float(row["estimate"]), 0.015, f"{contrast} table mean")
        positive = int(np.sum(data > 0))
        require(positive == int(claim["positive_tasks"][0]),
                f"{contrast} audit-claim positive-task count mismatch")
        require(positive == int(row["positive_tasks"]),
                f"{contrast} table positive-task count mismatch")
        require(len(data) == int(claim["positive_tasks"][1]) == int(row["total_tasks"]),
                f"{contrast} task denominator mismatch")
        picks = rng.integers(0, len(data), size=(100_000, len(data)))
        interval = np.percentile(data[picks].mean(axis=1), [2.5, 97.5])
        for index, endpoint in enumerate(("ci_low", "ci_high")):
            close(float(interval[index]), float(claim["task_bootstrap95"][index]), 0.12,
                  f"{contrast} audit-claim bootstrap {endpoint}")
            close(float(interval[index]), float(row[endpoint]), 0.12,
                  f"{contrast} table bootstrap {endpoint}")

    # Verify the lower-tail interpretation of the K=4 versus K=3 gain.
    _, p4_values, p4_seeds = source["BAR-Prox K=4"]
    _, p3_values, p3_seeds = source["BAR-Prox K=3"]
    require(p4_seeds == p3_seeds == (0, 1, 2, 3), "proximal contrast must use four seeds")
    p4_cells = [
        statistics.mean(p4_values[(tau, seed, env)] for seed in p4_seeds)
        for tau in HIGH for env in reference_envs
    ]
    p3_cells = [
        statistics.mean(p3_values[(tau, seed, env)] for seed in p3_seeds)
        for tau in HIGH for env in reference_envs
    ]
    p4_p3 = stability["p4_vs_p3"]
    close(statistics.median(a - b for a, b in zip(p4_cells, p3_cells)),
          float(p4_p3["cell_median_difference"]), 1e-12,
          "P4-P3 cell-median difference")
    require(sum(b < 20 <= a for a, b in zip(p4_cells, p3_cells)) == int(p4_p3["rescued_cells"]),
            "P4-P3 rescued-cell count mismatch")
    require(sum(a < 20 <= b for a, b in zip(p4_cells, p3_cells)) == int(p4_p3["reverse_cells"]),
            "P4-P3 reverse-cell count mismatch")
    cell_gains = [a - b for a, b in zip(p4_cells, p3_cells)]
    rescued_gain = sum(
        gain for gain, a, b in zip(cell_gains, p4_cells, p3_cells) if b < 20 <= a)
    require(sum(cell_gains) > 0, "P4-P3 total high-budget cell gain must be positive")
    close(rescued_gain / sum(cell_gains), float(p4_p3["rescued_gain_share"]), 1e-12,
          "P4-P3 rescued-cell gain share")

    # Recompute task-clustered raw-run collapse-risk contrasts.
    risk_specs = (
        ("collapse-risk BAR-Prox K=4 minus TD3+BC", "TD3+BC", "p4_vs_td3"),
        ("collapse-risk BAR-Prox K=4 minus BAR-Prox K=3", "BAR-Prox K=3", "p4_vs_p3"),
    )
    risk_rng = np.random.default_rng(20260830)
    for contrast, baseline, claim_key in risk_specs:
        _, baseline_values, baseline_seeds = source[baseline]
        require(baseline_seeds == p4_seeds, f"{contrast} requires matching four-seed procedures")
        data = np.asarray([
            statistics.mean(p4_values[(tau, seed, env)] < 20 for tau in HIGH for seed in p4_seeds)
            - statistics.mean(baseline_values[(tau, seed, env)] < 20
                              for tau in HIGH for seed in baseline_seeds)
            for env in reference_envs
        ], dtype=float)
        claim = stability[claim_key]
        row = uncertainty[contrast]
        estimate = float(data.mean())
        close(estimate, float(claim["raw_collapse_risk_difference"]), 1e-12,
              f"{contrast} audit-claim estimate")
        close(estimate, float(row["estimate"]), 0.00015, f"{contrast} table estimate")
        picks = risk_rng.integers(0, len(data), size=(100_000, len(data)))
        interval = np.percentile(data[picks].mean(axis=1), [2.5, 97.5])
        for index, endpoint in enumerate(("ci_low", "ci_high")):
            close(float(interval[index]), float(claim["raw_collapse_risk_cluster95"][index]), 0.012,
                  f"{contrast} audit-claim interval {endpoint}")
            close(float(interval[index]), float(row[endpoint]), 0.012,
                  f"{contrast} table interval {endpoint}")

    # Each host-separated two-seed pair must independently reproduce the full
    # proximal depth ordering and its released collapse counts.
    proximal = (
        ("k1", "TD3+BC"),
        ("k2", "BAR-Prox K=2"),
        ("k3", "BAR-Prox K=3"),
        ("k4", "BAR-Prox K=4"),
    )
    pair_specs = (("seeds_0_1", (0, 1)), ("seeds_2_3", (2, 3)))
    for pair_key, pair_seeds in pair_specs:
        pair_claim = stability["host_separated_seed_pairs"][pair_key]
        observed_high: list[float] = []
        for short, method in proximal:
            _, values, method_seeds = source[method]
            require(set(pair_seeds).issubset(method_seeds), f"{method} lacks {pair_key}")
            high_scores = [
                values[(tau, seed, env)]
                for tau in HIGH for seed in pair_seeds for env in reference_envs
            ]
            cell_scores = [
                statistics.mean(values[(tau, seed, env)] for seed in pair_seeds)
                for tau in HIGH for env in reference_envs
            ]
            high_mean = statistics.mean(high_scores)
            observed_high.append(high_mean)
            close(high_mean, float(pair_claim["high_means"][short]), 1e-12,
                  f"{pair_key}/{short} high mean")
            require(sum(v < 20 for v in cell_scores) == int(pair_claim["cell_collapses"][short]),
                    f"{pair_key}/{short} cell collapse count mismatch")
            require(sum(v < 20 for v in high_scores) == int(pair_claim["raw_collapses"][short]),
                    f"{pair_key}/{short} raw collapse count mismatch")
        require(all(left < right for left, right in zip(observed_high, observed_high[1:])),
                f"{pair_key} does not reproduce the strict K=1<2<3<4 ordering")

    independent = "disjoint seeds 2-3 BAR-Prox K=4 minus TD3+BC"
    row = uncertainty[independent]
    data = np.asarray([
        high_task_mean("BAR-Prox K=4", env, (2, 3))
        - high_task_mean("TD3+BC", env, (2, 3))
        for env in reference_envs
    ], dtype=float)
    close(float(data.mean()), float(row["estimate"]), 0.015, f"{independent} mean")
    require(int(np.sum(data > 0)) == int(row["positive_tasks"]),
            f"{independent} positive-task count mismatch")
    require(len(data) == int(row["total_tasks"]), f"{independent} task denominator mismatch")
    pair_rng = np.random.default_rng(20260830)
    picks = pair_rng.integers(0, len(data), size=(100_000, len(data)))
    interval = np.percentile(data[picks].mean(axis=1), [2.5, 97.5])
    close(float(interval[0]), float(row["ci_low"]), 0.12, f"{independent} bootstrap lower")
    close(float(interval[1]), float(row["ci_high"]), 0.12, f"{independent} bootstrap upper")

    print("PASS complete sweeps: dynamic seed grids, four-seed stability, risk, and disjoint host-separated pairs")

def indexed(rows: Iterable[dict[str, str]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    out: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        require(key not in out, f"duplicate key {key}")
        out[key] = row
    return out


def verify_audit_pack(results_dir: Path) -> None:
    audit = results_dir / "diagnostics" / "audit_pack"
    expected = json.loads((ROOT / "data" / "audit_claims.json").read_text(encoding="utf-8"))

    target_rows = read_csv(audit / "target_exposure_run.csv")
    require(len(target_rows) == 270, f"target exposure rows={len(target_rows)}, expected 270")
    require({float(row["T"]) for row in target_rows} == {4.0, 7.0, 10.0, 14.0, 20.0},
            "target exposure audit budget scope is not T={4,7,10,14,20}")
    require({row["method"] for row in target_rows} == {"td3", "mpi2", "mpi3"},
            "target exposure audit method scope mismatch")
    require({int(row["seed"]) for row in target_rows} == {0, 1},
            "target exposure audit seed scope mismatch")
    require(len({row["env"] for row in target_rows}) == 9,
            "target exposure audit environment scope mismatch")
    target_idx = indexed(target_rows, ("method", "env", "T", "seed"))
    def target_pair(left: str, right: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        lvals, rvals = [], []
        for (method, env, tau, seed), row in target_idx.items():
            if method != left:
                continue
            rkey = (right, env, tau, seed)
            require(rkey in target_idx, f"missing target pair {rkey}")
            lvals.append(float(row[metric])); rvals.append(float(target_idx[rkey][metric]))
        require(len(lvals) == 90, f"{left}/{right}/{metric}: expected 90 pairs")
        return np.asarray(lvals), np.asarray(rvals)
    for label, left, right in (("prox3_vs_td3", "mpi3", "td3"), ("prox3_vs_prox2", "mpi3", "mpi2")):
        lval, rval = target_pair(left, right, "d_target_deterministic")
        claim = expected["target_exposure"][label]
        require(int(np.sum(lval < rval)) == claim["lower"][0], f"{label} lower count mismatch")
        close(float(np.median(lval / rval)), float(claim["median_ratio"]), 1e-9, f"{label} median ratio")
    lval, rval = target_pair("mpi3", "td3", "d_final_current")
    claim = expected["target_exposure"]["final_proxy_prox3_vs_td3"]
    require(int(np.sum(lval < rval)) == claim["lower"][0], "final proxy lower count mismatch")
    close(float(np.median(lval / rval)), float(claim["median_ratio"]), 1e-9, "final proxy median ratio")

    proxy = np.asarray([float(row["current_state_proxy"]) for row in target_rows])
    det = np.asarray([float(row["d_target_deterministic"]) for row in target_rows])
    close(float(np.corrcoef(proxy, det)[0, 1]),
          float(expected["target_exposure"]["proxy_alignment"]["pearson_deterministic"]),
          1e-12, "proxy/target correlation")

    route_rows = read_csv(audit / "route_intervention_run.csv")
    require(len(route_rows) == 8, "route audit must contain 8 runs")
    require(sum(float(row["delta"]) > 0 for row in route_rows) == 8, "route primary direction is not 8/8")
    require(sum(float(row["delta_final_checkpoint_only"]) > 0 for row in route_rows) == 5,
            "route final-checkpoint sensitivity is not 5/8")
    close(statistics.mean(float(row["delta"]) for row in route_rows),
          float(expected["route"]["mean_delta"]), 1e-12, "route mean delta")

    simulator_rows = read_csv(audit / "simulator_checkpoint.csv")
    require(len(simulator_rows) == 60, "simulator audit must contain 60 checkpoints")
    stable = [float(r["median_signed_error"]) for r in simulator_rows if int(r["collapsed"]) == 0]
    collapsed = [float(r["median_signed_error"]) for r in simulator_rows if int(r["collapsed"]) == 1]
    require((len(stable), len(collapsed)) == (55, 5), "simulator stable/collapsed count mismatch")
    close(float(np.median(stable)), float(expected["simulator"]["checkpoint_median_of_medians_stable"]),
          1e-9, "simulator stable median of checkpoint medians")
    close(float(np.median(collapsed)), float(expected["simulator"]["checkpoint_median_of_medians_collapsed"]),
          1e-3, "simulator collapsed median of checkpoint medians")

    failure_rows = read_csv(audit / "failure_diagnostics_run.csv")
    require(len(failure_rows) == 180, "failure audit must contain 180 rows")
    for method in ("lin2", "prox2"):
        for group, pred in (("stable", lambda x: x >= 20), ("collapsed", lambda x: x < 20)):
            selected = [r for r in failure_rows if r["method"] == method and pred(float(r["score"]))]
            claim = expected["failure_diagnostics"][f"{method}_{group}"]
            require(len(selected) == int(claim["runs"]), f"{method}/{group} run count mismatch")
            mapping = {
                "td_p99": "td_error_p99", "q_abs": "q_abs_mean", "critic_loss": "critic_loss",
                "d_critic": "d_critic", "d_final": "d_final", "common_gain": "common_critic_gain",
            }
            for claim_field, csv_field in mapping.items():
                observed = statistics.median(float(r[csv_field]) for r in selected)
                exp = float(claim[claim_field])
                tol = max(1e-9, abs(exp) * 1e-12)
                close(observed, exp, tol, f"{method}/{group}/{claim_field}")
    print("PASS compact audit: target, route, simulator, and failure claims")


def verify_bundled_data_only() -> None:
    summary = bundled_summary()
    require(set(summary) == set(METHODS), "aggregate summary method set mismatch")

    claims = json.loads((ROOT / "data" / "audit_claims.json").read_text(encoding="utf-8"))
    stability = claims["stability"]
    four_seed = stability["four_seed_proximal"]
    require(four_seed["seeds"] == [0, 1, 2, 3], "four-seed proximal claim has wrong seeds")
    proximal = (
        ("k1", "TD3+BC"),
        ("k2", "BAR-Prox K=2"),
        ("k3", "BAR-Prox K=3"),
        ("k4", "BAR-Prox K=4"),
    )
    for short, method in proximal:
        row = summary[method]
        require(row["seeds"] == "4", f"{method} bundled seed count mismatch")
        require(row["total_high_budget_runs"] == "252",
                f"{method} bundled high-budget denominator mismatch")
        close(float(row["mean_T_ge_4"]), float(four_seed["high_means"][short]), 0.0000015,
              f"{method} bundled high mean")
        require(int(row["collapsed_cells_lt20_of63"]) == int(four_seed["cell_collapses"][short][0]),
                f"{method} bundled cell collapse count mismatch")
        require(int(four_seed["cell_collapses"][short][1]) == 63,
                f"{method} bundled cell denominator mismatch")
        require(int(row["collapsed_runs_lt20"]) == int(four_seed["raw_collapses"][short][0]),
                f"{method} bundled raw collapse count mismatch")
        require(int(four_seed["raw_collapses"][short][1]) == 252,
                f"{method} bundled raw denominator mismatch")
    for method in ("BAR-Lin K=2", "BAR-Lin K=3"):
        require(summary[method]["seeds"] == "2", f"{method} bundled seed count mismatch")
        require(summary[method]["total_high_budget_runs"] == "126",
                f"{method} bundled high-budget denominator mismatch")

    uncertainty = {
        row["contrast"]: row
        for row in read_csv(ROOT / "data" / "task_cluster_uncertainty.csv")
    }
    expected_contrasts = {
        "BAR-Prox K=4 minus TD3+BC",
        "BAR-Prox K=4 minus BAR-Prox K=3",
        "collapse-risk BAR-Prox K=4 minus TD3+BC",
        "collapse-risk BAR-Prox K=4 minus BAR-Prox K=3",
        "disjoint seeds 2-3 BAR-Prox K=4 minus TD3+BC",
    }
    require(set(uncertainty) == expected_contrasts,
            "task-cluster uncertainty contrast set mismatch")
    require(uncertainty["BAR-Prox K=4 minus TD3+BC"]["positive_tasks"] == "8",
            "bundled four-seed P4-TD3 positive-task count mismatch")
    require(uncertainty["BAR-Prox K=4 minus BAR-Prox K=3"]["positive_tasks"] == "7",
            "bundled four-seed P4-P3 positive-task count mismatch")

    configs = indexed(read_csv(ROOT / "data" / "config_summary.csv"), ("item",))
    require(configs[("nominal budget mapping",)]["value"] ==
            "T=tau=alpha/2; h=T/K; K1 uses alpha=2T",
            "configuration ledger budget mapping mismatch")
    require(configs[("Q-scale safeguard",)]["value"] == "1e-6",
            "configuration ledger Q-scale safeguard mismatch")
    cadence = configs[("intermediate evaluation cadence",)]["value"].lower()
    cadence_tokens = (
        "run-family-specific",
        "50k for k1/p2/p3",
        "mixed 50k/final-only for l2",
        "final-only for l3",
        "exact k4 configs unavailable",
    )
    require(all(token in cadence for token in cadence_tokens),
            "configuration ledger does not preserve the run-family-specific cadence")
    require(configs[("evaluation episodes",)]["value"] == "10",
            "configuration ledger evaluation-episode count mismatch")
    reported = configs[("reported score",)]["value"].lower()
    require(reported == "10-episode d4rl normalized mean at 1m",
            "configuration ledger final-score estimand mismatch")

    require(claims["target_exposure"]["prox3_vs_td3"]["lower"] == [88, 90],
            "bundled exposure claim mismatch")
    require(claims["route"]["primary_positive"] == [8, 8],
            "bundled route primary mismatch")
    require(claims["route"]["final_checkpoint_positive"] == [5, 8],
            "bundled route sensitivity mismatch")
    print("PASS bundled compact data: dynamic seed schemas, claims, and configuration ledger")

def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"command failed ({' '.join(command)}): {result.stderr.strip()}")
    return result.stdout


REWRITTEN_CHECKPOINT_MEMBERS = {
    "sweep_results/diagnostics/audit_pack/target_exposure_run.csv",
    "sweep_results/diagnostics/audit_pack/simulator_checkpoint.csv",
    "sweep_results/diagnostics/audit_pack/failure_diagnostics_run.csv",
}


def expected_archive_payload(name: str) -> bytes:
    logical = PurePosixPath(name)
    source = ROOT.parent.joinpath(*logical.parts)
    require(source.is_file(), f"bundle ZIP member has no current workspace source: {name}")
    if name not in REWRITTEN_CHECKPOINT_MEMBERS:
        return source.read_bytes()

    rows = read_csv(source)
    require("checkpoint" in rows[0], f"expected checkpoint column in {name}")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        checkpoint_name = Path(row["checkpoint"]).name
        identity = "/".join(
            f"{key}={row[key]}" for key in ("method", "env", "T", "seed") if key in row)
        row["checkpoint"] = f"artifact://checkpoint/{identity}/{checkpoint_name}"
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def verify_bundle_zip(bundle_zip: Path) -> int:
    require(bundle_zip.is_file() and bundle_zip.stat().st_size > 100_000,
            "missing or implausibly small complete bundle ZIP")
    with zipfile.ZipFile(bundle_zip) as archive:
        names = archive.namelist()
        require(archive.testzip() is None, "bundle ZIP contains a corrupt member")
        require(len(names) == len(set(names)) and len(names) >= 60,
                "bundle ZIP has duplicate members or an incomplete file set")
        required_members = {
            "gpt/paper.tex", "gpt/supplement.tex", "gpt/ReproducibilityChecklist.tex",
            "gpt/gpt_aaai26_manuscript.pdf", "gpt/gpt_aaai26_supplement.pdf",
            "gpt/verify_bundle.py", "gpt/package_bundle.py", "train_td3bc.py",
            "sweep_results/K=1/Imp/seed0.csv", "sweep_results/K=4/Imp/seed3.csv",
            "sweep_results/K=3/Exp/seed1.csv",
            "sweep_results/diagnostics/audit_pack/target_exposure_run.csv",
        }
        require(required_members.issubset(names),
                "bundle ZIP is missing manuscript, code, score, or audit members")
        for name in names:
            logical = PurePosixPath(name)
            require(not logical.is_absolute() and ".." not in logical.parts,
                    f"unsafe bundle ZIP member path: {name}")
            require(not name.endswith((".aux", ".log")),
                    f"temporary LaTeX file leaked into bundle ZIP: {name}")
            require("audit_manifest" not in name and not name.endswith("checksums.txt"),
                    f"local-path provenance file leaked into bundle ZIP: {name}")
            observed = archive.read(name)
            expected = expected_archive_payload(name)
            require(observed == expected,
                    f"bundle ZIP member is stale relative to the workspace: {name}")
        for name in REWRITTEN_CHECKPOINT_MEMBERS:
            payload = archive.read(name)
            local_home_marker = b"/" + b"home/"
            require(b"artifact://checkpoint/" in payload and local_home_marker not in payload,
                    f"checkpoint identifiers were not anonymized in {name}")
    return len(names)


def verify_built_pdfs() -> None:
    pdffonts = shutil.which("pdffonts")
    pdfinfo = shutil.which("pdfinfo")
    require(pdffonts is not None, "pdffonts is required for fail-closed font verification")
    require(pdfinfo is not None, "pdfinfo is required for fail-closed page verification")

    main_pdf = ROOT / "gpt_aaai26_manuscript.pdf"
    supp_pdf = ROOT / "gpt_aaai26_supplement.pdf"
    bundle_zip = ROOT / "gpt_aaai26_bundle.zip"
    for pdf in (main_pdf, supp_pdf):
        require(pdf.is_file() and pdf.stat().st_size > 10_000,
                f"missing or implausibly small PDF: {pdf.name}")
        fonts = command_output([pdffonts, str(pdf)])
        require(re.search(r"\bType\s+3\b", fonts, re.IGNORECASE) is None,
                f"Type 3 font detected in {pdf.name}")

    aux_path = ROOT / "paper.aux"
    require(aux_path.is_file(), "paper.aux is missing")
    aux = aux_path.read_text(encoding="utf-8", errors="replace")
    technical_match = re.search(
        r"newlabel\{sec:technical_end\}\{\{[^}]*\}\{([0-9]+)\}", aux)
    require(technical_match is not None,
            "technical-content boundary label missing from paper.aux")
    technical_page = int(technical_match.group(1))
    require(technical_page <= 7,
            f"technical content exceeds AAAI page 7: boundary page {technical_page}")
    checklist_match = re.search(
        r"newlabel\{sec:checklist_start\}\{\{[^}]*\}\{([0-9]+)\}", aux)
    require(checklist_match is not None,
            "checklist-start label missing from paper.aux")
    checklist_page = int(checklist_match.group(1))

    def page_count(pdf: Path) -> int:
        info = command_output([pdfinfo, str(pdf)])
        pages = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        require(pages is not None, f"pdfinfo did not report a page count for {pdf.name}")
        return int(pages.group(1))

    main_pages = page_count(main_pdf)
    supp_pages = page_count(supp_pdf)
    require(technical_page < checklist_page <= main_pages,
            "official checklist must begin after technical content and remain inside the main PDF")
    require(supp_pages >= 1, "supplement PDF has no pages")

    bundle_member_count = verify_bundle_zip(bundle_zip)

    for stem in ("paper", "supplement"):
        log_path = ROOT / f"{stem}.log"
        require(log_path.is_file(), f"{stem}.log is missing")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        blocking = (
            "Undefined control sequence", "Emergency stop", "Fatal error",
            "LaTeX Error", "Overfull \\hbox", "Overfull \\vbox",
            "There were undefined references", "There were undefined citations",
        )
        found = [token for token in blocking if token in log]
        require(not found, f"{stem}.log contains blocking warnings/errors: {found}")
    print(
        "PASS built PDFs: "
        f"technical boundary p.{technical_page}, checklist p.{checklist_page}, "
        f"main {main_pages} pp., supplement {supp_pages} pp., "
        f"ZIP {bundle_member_count} files, fonts and logs")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT.parent / "sweep_results")
    parser.add_argument("--prebuild", action="store_true")
    parser.add_argument("--postbuild", action="store_true")
    args = parser.parse_args()
    if not args.prebuild and not args.postbuild:
        args.prebuild = args.postbuild = True

    try:
        if args.prebuild:
            verify_style_and_source()
            verify_bundled_data_only()
            require(args.results_dir.is_dir(),
                    f"results directory is required for fail-closed prebuild verification: {args.results_dir}")
            verify_complete_sweeps(args.results_dir)
            verify_audit_pack(args.results_dir)
        if args.postbuild:
            verify_built_pdfs()
    except VerificationError as error:
        print(f"VERIFY FAILED: {error}", file=sys.stderr)
        return 1
    print("ALL REQUESTED CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
