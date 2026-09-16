# MART sweep results

Canonical scores remain in `K=<depth>/<Imp|Exp>/seed<seed>.csv`. The `tau` column is the manuscript's total nominal horizon T; h = T/K.

This index separates the original 14-budget grid from the extension T = {24, 28, 34, 40}. Counts describe published final scores, not live worker status.

## Coverage

| Method | Original grid, seeds 0–3 | Extension, seeds 0–1 | Extension, seeds 2–3 | Extension, seeds 0–3 |
|---|---:|---:|---:|---:|
| TD3+BC | 504/504 | 72/72 | 72/72 | 144/144 |
| I-MART-2 | 504/504 | 72/72 | 72/72 | 144/144 |
| I-MART-3 | 504/504 | 72/72 | 72/72 | 144/144 |
| I-MART-4 | 504/504 | 72/72 | 72/72 | 144/144 |
| E-MART-2 | 504/504 | 0/72 | 72/72 | 72/144 |
| E-MART-3 | 504/504 | 0/72 | 72/72 | 72/144 |

## Original grid: four training seeds

Values are mean ± sample standard deviation of the four seed-level nine-task means. This standard deviation is across training seeds, not episodes or environments.

| T | TD3+BC | I-MART-2 | I-MART-3 | I-MART-4 | E-MART-2 | E-MART-3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 63.81 ± 3.05 | 60.45 ± 1.49 | 60.14 ± 2.41 | 60.04 ± 1.99 | 62.28 ± 1.79 | 59.27 ± 1.59 |
| 0.1 | 62.57 ± 1.17 | 64.19 ± 2.81 | 60.82 ± 0.96 | 63.90 ± 3.41 | 63.45 ± 0.67 | 62.86 ± 3.37 |
| 0.2 | 64.58 ± 0.69 | 67.90 ± 1.16 | 68.08 ± 2.72 | 69.14 ± 2.81 | 66.62 ± 1.36 | 64.99 ± 2.38 |
| 0.4 | 65.94 ± 2.65 | 69.61 ± 1.24 | 71.01 ± 3.52 | 72.02 ± 3.20 | 71.63 ± 2.00 | 70.90 ± 0.65 |
| 0.7 | 67.69 ± 2.97 | 73.65 ± 1.22 | 73.32 ± 3.45 | 73.39 ± 1.40 | 70.98 ± 4.38 | 74.13 ± 5.30 |
| 1.5 | 74.79 ± 1.38 | 78.05 ± 2.28 | 78.95 ± 1.79 | 78.17 ± 2.75 | 75.13 ± 3.73 | 77.74 ± 3.58 |
| 2.5 | 67.41 ± 15.81 | 80.66 ± 1.43 | 81.41 ± 1.49 | 81.42 ± 0.71 | 78.43 ± 3.87 | 78.20 ± 3.62 |
| 4 | 45.93 ± 5.54 | 74.48 ± 9.82 | 81.26 ± 0.92 | 82.68 ± 1.08 | 75.16 ± 3.97 | 78.16 ± 4.93 |
| 7 | 30.36 ± 11.01 | 58.88 ± 9.73 | 71.00 ± 12.94 | 82.29 ± 0.80 | 51.49 ± 7.34 | 67.80 ± 9.83 |
| 10 | 23.77 ± 4.62 | 41.14 ± 10.49 | 51.14 ± 5.97 | 74.42 ± 10.89 | 37.81 ± 3.36 | 45.60 ± 8.50 |
| 12 | 21.64 ± 3.21 | 32.17 ± 4.91 | 49.74 ± 6.38 | 67.90 ± 9.54 | 26.03 ± 1.41 | 39.96 ± 11.99 |
| 14 | 18.96 ± 1.83 | 29.55 ± 4.70 | 41.68 ± 5.65 | 54.34 ± 10.72 | 27.46 ± 3.32 | 41.69 ± 4.93 |
| 17 | 18.92 ± 5.00 | 25.95 ± 5.14 | 32.75 ± 6.22 | 52.28 ± 4.44 | 27.50 ± 3.81 | 26.28 ± 2.19 |
| 20 | 18.09 ± 1.82 | 26.25 ± 6.40 | 35.83 ± 10.14 | 31.40 ± 6.08 | 23.55 ± 5.47 | 25.59 ± 4.06 |

## High-T extension: fixed seed pairs

TD3+BC uses its existing seeds 0–1; the new MART extension uses seeds 2–3. Comparisons to TD3+BC in this table are not paired by training seed. Each complete entry uses all nine tasks and both stated seeds (18 runs). Incomplete entries show counts only; they are never averaged over the available subset.

| T | TD3+BC (0–1) | I-MART-2 (2–3) | I-MART-3 (2–3) | I-MART-4 (2–3) | E-MART-2 (2–3) | E-MART-3 (2–3) |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 19.66 ± 5.14 | 23.82 ± 0.12 | 25.45 ± 1.28 | 42.96 ± 10.29 | 23.48 ± 8.08 | 24.98 ± 3.19 |
| 28 | 23.01 ± 11.25 | 19.54 ± 7.17 | 26.42 ± 1.39 | 26.81 ± 1.53 | 14.42 ± 2.86 | 22.22 ± 0.88 |
| 34 | 16.38 ± 2.29 | 21.85 ± 6.20 | 28.91 ± 4.09 | 29.57 ± 3.37 | 9.21 ± 1.76 | 26.26 ± 7.30 |
| 40 | 23.84 ± 0.53 | 20.25 ± 1.79 | 23.67 ± 0.29 | 27.94 ± 2.17 | 13.52 ± 3.21 | 15.82 ± 6.86 |

## Machine-readable tables

- [scores_long.csv](scores_long.csv): method, T, h, seed, environment, score, availability, and source matrix. Missing cells remain empty; extra sampled budgets are retained.
- [summary_by_T.csv](summary_by_T.csv): separate fixed cohorts {0,1}, {2,3}, and {0,1,2,3}, with coverage, mean, seed standard deviation, and both raw and seed-mean score-below-20 counts. Aggregate fields remain empty unless the entire cohort is present.

Only the six headline methods are included. K=8 and diagnostic/control archives remain outside these aggregates. The common-h coordinate reindexes the same scores; it is not another experiment.

## E-MART-2 score restoration

The 126 original-grid scores in each of `K=2/Exp/seed2.csv` and `seed3.csv` were restored from [the archived matrices](https://github.com/SChoish/MPI_sweep/tree/2fe4b8f3729db31666cd3d1951fa93e9a8131126/sweep_results/K%3D2/Exp). Only empty cells at T ≤ 20 were filled: 252 values in total. All 72 already-published extension scores in those files were preserved. This restores historical results; no training or reevaluation was performed.

## Regenerate

```bash
python3 scripts/summarize_sweep_results.py
```

The local score-export script also refreshes this index and both derived CSVs after merging matrices. After a manual matrix edit, run the command above before committing.
