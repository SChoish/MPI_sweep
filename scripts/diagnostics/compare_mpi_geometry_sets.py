#!/usr/bin/env python3
"""Combine geometry summaries and compare MPI variants against TD3+BC."""

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
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--high-t-min", type=float, default=4.0)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for geometry_dir in args.geometry_dirs:
        root = geometry_dir.resolve()
        source = root.name
        sources.append(str(root))
        for row in _read(root / "seed_mean_diagnostics.csv"):
            all_rows.append({"source_set": source, **row})

    by_cell: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        key = (row["source_set"], row["env"], float(row["tau"]))
        by_cell[key][row["method"]] = row

    delta_rows: list[dict[str, Any]] = []
    for (source, env_name, tau), methods in sorted(by_cell.items()):
        if "td3" not in methods:
            continue
        baseline = methods["td3"]
        for method in ("mpi2", "mpi3"):
            if method not in methods:
                continue
            candidate = methods[method]
            base_td = max(float(baseline["td_residual_mean"]), 1e-30)
            method_td = max(float(candidate["td_residual_mean"]), 1e-30)
            delta_rows.append(
                {
                    "source_set": source,
                    "env": env_name,
                    "tau": tau,
                    "method": method,
                    "score_delta_vs_td3": float(candidate["d4rl_score_mean"])
                    - float(baseline["d4rl_score_mean"]),
                    "canonical_q_gain_delta_vs_td3": float(
                        candidate["canonical_q1_gain_total_mean"]
                    )
                    - float(baseline["canonical_q1_gain_total_mean"]),
                    "log10_td_residual_ratio_vs_td3": math.log10(method_td / base_td),
                    "path_tortuosity": float(candidate["path_tortuosity_mean"]),
                    "seed_gap": float(candidate["d4rl_seed_gap"]),
                }
            )

    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "method_deltas.csv", delta_rows)

    correlations: dict[str, Any] = {}
    for method in ("mpi2", "mpi3"):
        rows = [row for row in delta_rows if row["method"] == method]
        scores = [float(row["score_delta_vs_td3"]) for row in rows]
        correlations[method] = {
            "cells": len(rows),
            "score_delta_vs_canonical_q_gain_delta_pearson": _pearson(
                scores,
                [float(row["canonical_q_gain_delta_vs_td3"]) for row in rows],
            ),
            "score_delta_vs_log10_td_residual_ratio_pearson": _pearson(
                scores,
                [float(row["log10_td_residual_ratio_vs_td3"]) for row in rows],
            ),
            "score_delta_vs_path_tortuosity_pearson": _pearson(
                scores, [float(row["path_tortuosity"]) for row in rows]
            ),
        }

    high_t_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        if float(row["tau"]) >= args.high_t_min:
            high_t_groups[(row["env"], row["method"])].append(row)
    high_t_rows: list[dict[str, Any]] = []
    for (env_name, method), rows in sorted(high_t_groups.items()):
        high_t_rows.append(
            {
                "env": env_name,
                "method": method,
                "cells": len(rows),
                "score_delta_vs_td3_mean": statistics.mean(
                    float(row["score_delta_vs_td3"]) for row in rows
                ),
                "canonical_q_gain_delta_vs_td3_mean": statistics.mean(
                    float(row["canonical_q_gain_delta_vs_td3"]) for row in rows
                ),
                "log10_td_residual_ratio_vs_td3_mean": statistics.mean(
                    float(row["log10_td_residual_ratio_vs_td3"]) for row in rows
                ),
            }
        )
    _write(output / "high_t_environment_summary.csv", high_t_rows)

    summary = {
        "sources": sources,
        "high_t_min": args.high_t_min,
        "delta_rows": len(delta_rows),
        "correlations": correlations,
        "all_numeric_outputs_finite": all(
            math.isfinite(float(row[key]))
            for row in delta_rows
            for key in (
                "score_delta_vs_td3",
                "canonical_q_gain_delta_vs_td3",
                "log10_td_residual_ratio_vs_td3",
                "path_tortuosity",
            )
        ),
    }
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[compare] sources={len(sources)} deltas={len(delta_rows)} out={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
