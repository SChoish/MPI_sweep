# Archived offrl actor-path continuation audit

Local host dump of the exploratory endpoint-continuation analysis on every
**available 1M checkpoint** under offrl result trees. It is not a controlled
semigroup-error test and does not replace the archived ext_csh pack at
[`../../../actor_path_semigroup/`](../../../actor_path_semigroup/).

## Inputs (2026-08-31)

| Method | Tree | 1M cells used | Compose-ready splits |
| --- | --- | ---: | ---: |
| mpi1 | `results/mpi1_unnorm` | 304 (seeds 0–1) | 50 |
| mpi2 | `results/mpi2_unnorm` | 106 (seeds 0–1) | 6 |
| mpi4 | `results/mpi4_norm` | 504 (seeds 0–3) | 288 |
| mpi8 | `results/mpi8_norm` | 22 (partial Hopper-medium snapshot, seeds 0--1) | 8 |

- Integrators: explicit, implicit
- Shared interior states: 512 / env
- Missing lookups skipped (not listed); only cells with the needed triple/pair contribute rows

## Rows

| File | Rows |
| --- | ---: |
| `actor_composition.csv` | 704 |
| `cross_k_finals.csv` | 148 |
| `hop_paths.csv` | 1980 |
| `frozen_from_aD.csv` | 704 |

## Exploratory within-tree medians

These rows must not be read as a depth trend: K1/K2 use unnormalized result
trees, K4/K8 use normalized trees, coverage differs by method, and K8 contains
only 22 Hopper-medium checkpoints. Cross-K values mixing these trees are excluded from manuscript inference. The
within-tree values are also excluded from numerical-order claims because the
operator, scale, coverage, and learned optimization path are not held fixed.
See [`../../../../../TODO.md`](../../../../../TODO.md) for the replacement.

The within-tree `relative_pi_u_vs_continue_mean` medians are:

| Method | explicit | implicit |
| --- | ---: | ---: |
| mpi1 | 0.9135 | 0.9132 |
| mpi2 | 0.9477 | 0.9475 |
| mpi4 | 0.5500 | 0.2819 |
| mpi8 | 0.1506 | 0.1507 |

Cross-K final-action relative RMS medians include mpi1↔mpi4 **0.965**,
mpi4↔mpi8 **0.316** (small mpi8 sample).

## Reproduce

Canonical script:
[`scripts/diagnostics/run_actor_semigroup.py`](../../../../../scripts/diagnostics/run_actor_semigroup.py)
(`--method-dir` overrides; compose per method; pairwise cross-K among present methods).

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python -u \
  scripts/diagnostics/run_actor_semigroup.py \
  --results-root results \
  --data-dir "$HOME/.d4rl/datasets" \
  --out-dir sweep_results/diagnostics/hosts/offrl/actor_path_semigroup \
  --seeds 0 1 2 3 \
  --methods mpi1 mpi2 mpi4 mpi8 \
  --method-dir mpi1=results/mpi1_unnorm \
  --method-dir mpi2=results/mpi2_unnorm \
  --method-dir mpi4=results/mpi4_norm \
  --method-dir mpi8=results/mpi8_norm \
  --integrators explicit implicit --n-states 512
```
