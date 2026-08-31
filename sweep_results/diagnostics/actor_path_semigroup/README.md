# Actor-path semigroup audit

This bundle measures whether learned actor endpoints behave like a semigroup
in action space. It compares the direct endpoint \(\pi_{s+t}\) with a frozen
critic continuation from \(\pi_s\), a frozen map composed from dataset actions
\(a_D\), and final actions from MPI-1/2/3 (and Exp-3 where present) at fixed
total \(\tau\).

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
hop-path audits use the full \(\tau\) grid, including `0.7` and `12`. Missing
paths in the manifest are only absent Exp-3 cells (188 requested, not present
locally); no existing local 1M checkpoint was skipped.

- Rows: 288 actor-composition, 948 cross-K, and 1,704 hop-path records.
- Relative direct-versus-continued actor discrepancy median:
  explicit `0.5623`, implicit `0.4745`.
- Relative actor-versus-frozen-map discrepancy median:
  explicit `0.9738`, implicit `0.9853`.
- Relative cross-K final-action discrepancy median (selected pairs):
  MPI-1/2 `0.4839`, MPI-1/3 `0.5045`, MPI-2/3 `0.3604`,
  MPI-3/Exp-3 `0.2611`.
- The frozen-map semigroup defect remains small at short total steps and grows
  with step size.

## Interpretation and limitations

The frozen operator is locally close to semigroup composition at small
steps, but the learned actor endpoints do not satisfy a comparable strong
composition identity. Cross-K endpoints also differ materially. These are
action-space diagnostics, not return guarantees or an end-to-end policy
performance certificate.

Implicit statistics require caution: only 135 of 144 composition rows had
any converged samples, and the mean converged-sample fraction was `0.4606`.
The CSV files retain convergence and projection diagnostics for filtering.

This archived run includes seed 3. Under the current paper protocol, paper
seeds are `{0,1,2}` and seed 3 is non-paper evidence; therefore these
aggregates must not be presented as the current three-seed paper aggregate.
Exp-3 coverage is limited to the three environments present locally
(`hopper-medium-v2`, `walker2d-medium-replay-v2`, `walker2d-expert-v2`).

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
