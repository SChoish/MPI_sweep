#!/usr/bin/env python3
"""Build compact comparison tables from dump_mpi_frontier_geometry.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_dir", type=Path)
    parser.add_argument("--small-t-max", type=float, default=0.4)
    parser.add_argument("--collapse-threshold", type=float, default=20.0)
    parser.add_argument("--win-margin", type=float, default=1.0)
    args = parser.parse_args()
    root = args.geometry_dir.resolve()
    hops = _read(root / "hop_geometry.csv")
    runs = _read(root / "run_diagnostics.csv")

    final: dict[tuple[str, str, float, int, str], dict[str, str]] = {}
    for row in hops:
        if int(row["hop"]) == int(row["total_hops"]):
            final[
                (
                    row["method"],
                    row["env"],
                    float(row["tau"]),
                    int(row["seed"]),
                    row["critic_scope"],
                )
            ] = row

    curvature_rows: list[dict[str, Any]] = []
    env_tau_seed = sorted(
        {
            (row["env"], float(row["tau"]), int(row["seed"]))
            for row in runs
        }
    )
    for env_name, tau, seed in env_tau_seed:
        key = lambda method: (  # noqa: E731
            method,
            env_name,
            tau,
            seed,
            "canonical_td3_tau1",
        )
        if not all(key(method) in final for method in ("td3", "mpi2", "mpi3")):
            continue
        delta_q1 = float(final[key("td3")]["q1_gain_total_mean"])
        delta_q2 = float(final[key("mpi2")]["q1_gain_total_mean"])
        delta_q3 = float(final[key("mpi3")]["q1_gain_total_mean"])
        qhat12 = -4.0 * (delta_q2 - delta_q1) / (tau * tau)
        qhat13 = -3.0 * (delta_q3 - delta_q1) / (tau * tau)
        relative = abs(qhat12 - qhat13) / (
            abs(qhat12) + abs(qhat13) + 1e-8
        )
        curvature_rows.append(
            {
                "env": env_name,
                "tau": tau,
                "seed": seed,
                "delta_q_td3": delta_q1,
                "delta_q_mpi2": delta_q2,
                "delta_q_mpi3": delta_q3,
                "qhat_12": qhat12,
                "qhat_13": qhat13,
                "qhat_sign_match": qhat12 * qhat13 > 0.0,
                "qhat_relative_disagreement": relative,
            }
        )
    _write(root / "curvature_consistency.csv", curvature_rows)

    run_groups: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        run_groups[(row["env"], float(row["tau"]), row["method"])].append(row)
    seed_mean_rows: list[dict[str, Any]] = []
    for (env_name, tau, method), group in sorted(run_groups.items()):
        if len(group) != 2:
            continue
        own = [
            final[(method, env_name, tau, int(row["seed"]), "own")]
            for row in group
        ]
        canonical = [
            final[
                (
                    method,
                    env_name,
                    tau,
                    int(row["seed"]),
                    "canonical_td3_tau1",
                )
            ]
            for row in group
        ]
        seed_mean_rows.append(
            {
                "env": env_name,
                "tau": tau,
                "method": method,
                "d4rl_score_mean": _mean(group, "d4rl_score"),
                "d4rl_seed_gap": abs(
                    float(group[0]["d4rl_score"])
                    - float(group[1]["d4rl_score"])
                ),
                "critic_displacement_mean": _mean(
                    group, "critic_displacement_sq_mean_metric_mean"
                ),
                "final_displacement_mean": _mean(
                    group, "final_displacement_sq_mean_metric_mean"
                ),
                "decoupling_displacement_ratio_mean": _mean(
                    group, "decoupling_displacement_ratio"
                ),
                "canonical_q1_gain_total_mean": _mean(
                    canonical, "q1_gain_total_mean"
                ),
                "own_q1_gain_total_mean": _mean(own, "q1_gain_total_mean"),
                "own_secant_q_final_mean": _mean(own, "secant_q_mean"),
                "own_prox_residual_final_mean": _mean(
                    own, "prox_residual_mean"
                ),
                "own_saturation_final_mean": _mean(
                    own, "saturation_fraction_mean"
                ),
                "twin_gap_final_mean": _mean(own, "twin_gap_mean"),
                "path_length_mean": _mean(group, "path_length_mean"),
                "chord_length_mean": _mean(group, "chord_length_mean"),
                "path_tortuosity_mean": _mean(group, "path_tortuosity_mean"),
                "td_residual_mean": _mean(group, "td_residual_mean"),
                "td_target_abs_p99_mean": _mean(group, "td_target_abs_p99"),
                "max_critic_loss_mean": _mean(group, "max_critic_loss"),
            }
        )
    _write(root / "seed_mean_diagnostics.csv", seed_mean_rows)

    method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[tuple[str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in seed_mean_rows:
        method_groups[str(row["method"])].append(row)
        by_cell[(str(row["env"]), float(row["tau"]))][str(row["method"])] = row

    method_cell_summary: list[dict[str, Any]] = []
    for method, group in sorted(method_groups.items()):
        scores = [float(row["d4rl_score_mean"]) for row in group]
        method_cell_summary.append(
            {
                "method": method,
                "cells": len(group),
                "score_mean": statistics.mean(scores),
                "collapse_cells": sum(
                    score < args.collapse_threshold for score in scores
                ),
                "critic_displacement_median": statistics.median(
                    float(row["critic_displacement_mean"]) for row in group
                ),
                "final_displacement_median": statistics.median(
                    float(row["final_displacement_mean"]) for row in group
                ),
            }
        )
    _write(root / "method_cell_summary.csv", method_cell_summary)

    paired_td3_mpi3: list[dict[str, Any]] = []
    for (env_name, tau), methods in sorted(by_cell.items()):
        if "td3" not in methods or "mpi3" not in methods:
            continue
        td3 = methods["td3"]
        mpi3 = methods["mpi3"]
        td3_score = float(td3["d4rl_score_mean"])
        mpi3_score = float(mpi3["d4rl_score_mean"])
        td3_collapsed = td3_score < args.collapse_threshold
        mpi3_collapsed = mpi3_score < args.collapse_threshold
        paired_td3_mpi3.append(
            {
                "env": env_name,
                "tau": tau,
                "td3_score": td3_score,
                "mpi3_score": mpi3_score,
                "score_delta_mpi3_minus_td3": mpi3_score - td3_score,
                "td3_dcrit": float(td3["critic_displacement_mean"]),
                "mpi3_dcrit": float(mpi3["critic_displacement_mean"]),
                "mpi3_dfinal": float(mpi3["final_displacement_mean"]),
                "td3_collapsed": td3_collapsed,
                "mpi3_collapsed": mpi3_collapsed,
                "collapse_discordant": td3_collapsed != mpi3_collapsed,
            }
        )
    _write(root / "paired_td3_mpi3.csv", paired_td3_mpi3)

    discordant = [
        row for row in paired_td3_mpi3 if bool(row["collapse_discordant"])
    ]
    paired_summary = {
        "cells": len(paired_td3_mpi3),
        "mpi3_wins_by_margin": sum(
            float(row["score_delta_mpi3_minus_td3"]) > args.win_margin
            for row in paired_td3_mpi3
        ),
        "td3_wins_by_margin": sum(
            float(row["score_delta_mpi3_minus_td3"]) < -args.win_margin
            for row in paired_td3_mpi3
        ),
        "td3_collapsed_mpi3_stable": sum(
            bool(row["td3_collapsed"]) and not bool(row["mpi3_collapsed"])
            for row in paired_td3_mpi3
        ),
        "mpi3_collapsed_td3_stable": sum(
            bool(row["mpi3_collapsed"]) and not bool(row["td3_collapsed"])
            for row in paired_td3_mpi3
        ),
        "collapse_discordant_cells": len(discordant),
        "stable_has_smaller_dcrit_in_discordant": sum(
            (
                float(row["mpi3_dcrit"]) < float(row["td3_dcrit"])
                if bool(row["td3_collapsed"])
                else float(row["td3_dcrit"]) < float(row["mpi3_dcrit"])
            )
            for row in discordant
        ),
    }

    hop_groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in hops:
        hop_groups[
            (row["critic_scope"], row["method"], int(row["hop"]))
        ].append(row)
    hop_summary: list[dict[str, Any]] = []
    for (scope, method, hop), group in sorted(hop_groups.items()):
        gains = [float(row["q1_gain_hop_mean"]) for row in group]
        hop_summary.append(
            {
                "critic_scope": scope,
                "method": method,
                "hop": hop,
                "cells": len(group),
                "positive_q_gain_cells": sum(value > 0.0 for value in gains),
                "positive_q_gain_fraction": sum(value > 0.0 for value in gains)
                / len(gains),
                "q1_gain_hop_median": statistics.median(gains),
                "secant_q_median": statistics.median(
                    float(row["secant_q_mean"]) for row in group
                ),
                "prox_residual_median": statistics.median(
                    float(row["prox_residual_mean"]) for row in group
                ),
                "step_norm_median": statistics.median(
                    float(row["step_norm_mean"]) for row in group
                ),
            }
        )
    _write(root / "hop_summary.csv", hop_summary)

    small = [
        row for row in curvature_rows if float(row["tau"]) <= args.small_t_max
    ]
    relative = [float(row["qhat_relative_disagreement"]) for row in small]
    summary = {
        "run_rows": len(runs),
        "hop_rows": len(hops),
        "curvature_rows": len(curvature_rows),
        "small_t_max": args.small_t_max,
        "small_t_rows": len(small),
        "collapse_threshold": args.collapse_threshold,
        "win_margin": args.win_margin,
        "method_cell_summary": method_cell_summary,
        "paired_td3_mpi3": paired_summary,
        "qhat_sign_match_fraction_small_t": (
            sum(
                str(row["qhat_sign_match"]).lower() == "true"
                for row in small
            )
            / len(small)
            if small
            else None
        ),
        "qhat_relative_disagreement_median_small_t": (
            statistics.median(relative) if relative else None
        ),
        "qhat_relative_disagreement_lt_0p25_fraction_small_t": (
            sum(value < 0.25 for value in relative) / len(relative)
            if relative
            else None
        ),
        "all_numeric_outputs_finite": all(
            math.isfinite(value)
            for row in curvature_rows
            for value in (
                float(row["qhat_12"]),
                float(row["qhat_13"]),
                float(row["qhat_relative_disagreement"]),
            )
        ),
    }
    (root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[summary] runs={len(runs)} hops={len(hops)} "
        f"curvature={len(curvature_rows)} out={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
