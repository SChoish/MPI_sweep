# Archived actor-path continuation audit

This bundle preserves an exploratory comparison between learned actor endpoints,
frozen-critic continuations, and cross-depth final actions. It is not used as a
semigroup or discretization-error test in the manuscript. Learned comparisons
change critics, Q scales, and optimization histories across budgets, while
projection and convergence produce unequal eligible subsets.

## Contents

| File | Description |
| --- | --- |
| `run_actor_semigroup.py` | Exact analysis-code snapshot used for this bundle |
| `actor_composition.csv` | Direct actor endpoint versus continued endpoint |
| `frozen_from_aD.csv` | One-step versus composed frozen maps from \(a_D\) |
| `cross_k_finals.csv` | Pairwise final-action discrepancies among available methods |
| `hop_paths.csv` | Intermediate-hop paths relative to MPI-1 and \(a_D\) |
| `SUMMARY.json` | Aggregate medians, p90 values, maxima, and row counts |
| `MANIFEST.json` | Inputs, sample indices, CLI arguments, and provenance |

The canonical maintained script is
[`scripts/diagnostics/run_actor_semigroup.py`](../../../scripts/diagnostics/run_actor_semigroup.py).
The copy here makes this archived audit self-contained.

## Archived result (local full coverage)

The run completed on 2026-08-31 over nine D4RL environments, seeds 2 and 3,
512 shared interior states per environment, and every finished local 1M
checkpoint under `results/{mpi1,mpi2,mpi3,exp3}_s23` (820 files). Cross-K and
hop-path audits use the full \(\tau\) grid, including `0.7` and `12`.
Composition is evaluated for each available method on \(s+t\) splits.

- Rows: 940 actor-composition, 948 cross-K, and 1,704 hop-path records.
- Relative direct-versus-continued actor discrepancy median:
  explicit `0.5623`, implicit `0.4745` (overall; see SUMMARY for by-method cuts).
- Relative cross-K final-action discrepancy median (selected pairs):
  MPI-1/2 `0.4839`, MPI-1/3 `0.5045`, MPI-2/3 `0.3604`.
- Exp-3 appears only where local checkpoints exist
  (`hopper-medium-v2`, `walker2d-medium-replay-v2`, `walker2d-expert-v2`).

## Interpretation and limitations

The reported distances show that independently trained endpoints and frozen
continuations differ; they do not estimate numerical order. Exact semigroup
equality is also not the null for a first-order Euler map. Implicit convergence
is incomplete on larger steps, and the CSVs retain projection and convergence
fields for auditing rather than post-hoc filtering.

This archive uses seeds 2--3, one paper seed pair, and is neither an independent
implementation nor evidence for a cross-depth error trend. The controlled
replacement is specified in [`../../../TODO.md`](../../../TODO.md).

## Reproduction

From the repository root, using a Python environment with JAX and the
training dependencies:

```bash
python scripts/diagnostics/run_actor_semigroup.py \
  --methods mpi1 mpi2 mpi3 exp3 \
  --out-dir sweep_results/diagnostics/actor_path_semigroup
```

The manifest records the exact archived arguments and selected dataset
indices. Reproduction also requires the checkpoint and dataset trees named
there.
