#!/usr/bin/env python3
"""Summarize published score matrices without averaging incomplete cohorts."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

BASE = (0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4, 7, 10, 12, 14, 17, 20)
HIGH = (24, 28, 34, 40)
ENVS = tuple(f"{domain}-{split}-v2" for domain in ("hopper", "halfcheetah", "walker2d")
             for split in ("medium", "medium-replay", "expert"))
METHODS = (("TD3+BC", 1, "Imp"), ("I-MART-2", 2, "Imp"),
           ("I-MART-3", 3, "Imp"), ("I-MART-4", 4, "Imp"),
           ("E-MART-2", 2, "Exp"), ("E-MART-3", 3, "Exp"))
COHORTS = ((0, 1), (2, 3), (0, 1, 2, 3))


def read_scores(root: Path):
    scores, grids = {}, {}
    for method, k, integrator in METHODS:
        grids[method] = set(BASE + HIGH)
        for seed in range(4):
            path = root / f"K={k}" / integrator / f"seed{seed}.csv"
            if not path.exists():
                continue
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != ["tau", *ENVS]:
                    raise ValueError(f"Unexpected score columns: {path}")
                seen = set()
                for row in reader:
                    t = float(row["tau"])
                    if not math.isfinite(t) or t <= 0 or t in seen:
                        raise ValueError(f"Invalid or duplicate tau: {path}: {t}")
                    seen.add(t)
                    grids[method].add(t)
                    for env in ENVS:
                        raw = row[env].strip()
                        if not raw:
                            continue
                        value = float(raw)
                        if not math.isfinite(value):
                            raise ValueError(f"Nonfinite score: {path}: {t}: {env}")
                        scores[(method, t, seed, env)] = value
    return scores, grids


def summarize(scores, method, t, seeds):
    keys = [(method, t, seed, env) for seed in seeds for env in ENVS]
    values = [scores[key] for key in keys if key in scores]
    complete = len(values) == len(keys)
    result = dict(method=method, T=f"{t:g}", seeds=",".join(map(str, seeds)),
                  available_runs=len(values), expected_runs=len(keys),
                  status="complete" if complete else "partial" if values else "missing",
                  mean_score="", std_seed_task_means="", score_below_20_runs="",
                  score_below_20_task_means="")
    if complete:
        seed_means = [mean(scores[(method, t, seed, env)] for env in ENVS) for seed in seeds]
        result.update(mean_score=mean(seed_means), std_seed_task_means=stdev(seed_means),
                      score_below_20_runs=sum(v < 20 for v in values),
                      score_below_20_task_means=sum(
                          mean(scores[(method, t, seed, env)] for seed in seeds) < 20
                          for env in ENVS))
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path):
    scores, grids = read_scores(root)
    long_rows, summaries = [], []
    for method, k, integrator in METHODS:
        for t in sorted(grids[method]):
            for seed in range(4):
                for env in ENVS:
                    value = scores.get((method, t, seed, env))
                    long_rows.append(dict(method=method, K=k, integrator=integrator,
                                          T=f"{t:g}", h=f"{t/k:.12g}", seed=seed,
                                          environment=env, score="" if value is None else value,
                                          status="missing" if value is None else "available",
                                          source_csv=f"K={k}/{integrator}/seed{seed}.csv"))
            for seeds in COHORTS:
                summaries.append(summarize(scores, method, t, seeds))
    write_csv(root / "scores_long.csv", long_rows)
    write_csv(root / "summary_by_T.csv", summaries)

    lines = ["# MART sweep results", "",
             "Canonical scores remain in `K=<depth>/<Imp|Exp>/seed<seed>.csv`. "
             "The `tau` column is the manuscript's total nominal horizon T; h = T/K.", "",
             "This index separates the original 14-budget grid from the extension "
             "T = {24, 28, 34, 40}. Counts describe published final scores, not live worker status.", "",
             "## Coverage", "",
             "| Method | Original grid, seeds 0–3 | Extension, seeds 0–1 | Extension, seeds 2–3 | Extension, seeds 0–3 |",
             "|---|---:|---:|---:|---:|"]
    for method, _, _ in METHODS:
        def count(taus, seeds):
            return sum((method, t, seed, env) in scores for t in taus for seed in seeds for env in ENVS)
        lines.append(f"| {method} | {count(BASE, range(4))}/504 | {count(HIGH, (0, 1))}/72 | "
                     f"{count(HIGH, (2, 3))}/72 | {count(HIGH, range(4))}/144 |")

    def display(method, t, seeds):
        row = summarize(scores, method, t, seeds)
        if row["status"] != "complete":
            return f"— ({row['available_runs']}/{row['expected_runs']})"
        return f"{row['mean_score']:.2f} ± {row['std_seed_task_means']:.2f}"

    lines += ["", "## Original grid: four training seeds", "",
              "Values are mean ± sample standard deviation of the four seed-level nine-task means. "
              "This standard deviation is across training seeds, not episodes or environments.", "",
              "| T | " + " | ".join(m[0] for m in METHODS) + " |",
              "|---:" + "|---:" * len(METHODS) + "|"]
    for t in BASE:
        lines.append(f"| {t:g} | " + " | ".join(display(m[0], t, (0, 1, 2, 3)) for m in METHODS) + " |")
    lines += ["", "## High-T extension: fixed seed pairs", "",
              "TD3+BC uses its existing seeds 0–1; the new MART extension uses seeds 2–3. "
              "Comparisons to TD3+BC in this table are not paired by training seed. "
              "Each complete entry uses all nine tasks and both stated seeds (18 runs). "
              "Incomplete entries show counts only; they are never averaged over the available subset.", "",
              "| T | TD3+BC (0–1) | I-MART-2 (2–3) | I-MART-3 (2–3) | I-MART-4 (2–3) | E-MART-2 (2–3) | E-MART-3 (2–3) |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for t in HIGH:
        lines.append(f"| {t:g} | " + " | ".join(display(m[0], t, (0, 1) if m[0] == "TD3+BC" else (2, 3)) for m in METHODS) + " |")
    lines += ["", "## Machine-readable tables", "",
              "- [scores_long.csv](scores_long.csv): method, T, h, seed, environment, score, availability, and source matrix. "
              "Missing cells remain empty; extra sampled budgets are retained.",
              "- [summary_by_T.csv](summary_by_T.csv): separate fixed cohorts {0,1}, {2,3}, and {0,1,2,3}, "
              "with coverage, mean, seed standard deviation, and both raw and seed-mean score-below-20 counts. "
              "Aggregate fields remain empty unless the entire cohort is present.", "",
              "Only the six headline methods are included. K=8 and diagnostic/control archives remain outside these aggregates. "
              "The common-h coordinate reindexes the same scores; it is not another experiment.", "",
              "## E-MART-2 score restoration", "",
              "The 126 original-grid scores in each of `K=2/Exp/seed2.csv` and `seed3.csv` were restored from "
              "[the archived matrices](https://github.com/SChoish/MPI_sweep/tree/2fe4b8f3729db31666cd3d1951fa93e9a8131126/sweep_results/K%3D2/Exp). "
              "Only empty cells at T ≤ 20 were filled: 252 values in total. "
              "All 72 already-published extension scores in those files were preserved. "
              "This restores historical results; no training or reevaluation was performed.", "",
              "## Regenerate", "", "```bash", "python3 scripts/summarize_sweep_results.py", "```", "",
              "The local score-export script also refreshes this index and both derived CSVs after merging matrices. "
              "After a manual matrix edit, run the command above before committing.", ""]
    (root / "README.md").write_text("\n".join(lines))
    print(f"Summarized {len(scores)} finite scores across six methods; incomplete cohorts left unaveraged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sweep_results")
    args = parser.parse_args()
    build(args.results_dir)


if __name__ == "__main__":
    main()
