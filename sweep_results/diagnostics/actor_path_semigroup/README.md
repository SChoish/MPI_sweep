# Actor-path semigroup audit

This bundle measures whether learned actor endpoints behave like a semigroup
in action space. It compares the direct endpoint \(\pi_{s+t}\) with a frozen
critic continuation from \(\pi_s\), a frozen map composed from dataset actions
\(a_D\), and final actions from MPI-1/2/3 at fixed total \(\tau\).

## Contents

| File | Description |
| --- | --- |
| `run_actor_semigroup.py` | Exact analysis-code snapshot used for this bundle |
| `actor_composition.csv` | Direct actor endpoint versus continued endpoint |
| `frozen_from_aD.csv` | One-step versus composed frozen maps from \(a_D\) |
| `cross_k_finals.csv` | Pairwise MPI-1/2/3 final-action discrepancies |
| `hop_paths.csv` | Intermediate-hop paths relative to MPI-1 and \(a_D\) |
| `SUMMARY.json` | Aggregate medians, p90 values, maxima, and row counts |
| `MANIFEST.json` | Inputs, sample indices, CLI arguments, and provenance |

The canonical maintained script is
[`scripts/diagnostics/run_actor_semigroup.py`](../../../scripts/diagnostics/run_actor_semigroup.py).
The copy here makes this archived audit self-contained.

## Archived result

The run completed on 2026-08-31 over nine D4RL environments, seeds 2 and 3,
512 shared interior states per environment, and MPI-1/2/3 checkpoints. No
checkpoint was missing.

- Rows: 288 actor-composition, 648 cross-K, and 1,296 hop-path records.
- Relative direct-versus-continued actor discrepancy median:
  explicit `0.5623`, implicit `0.4745`.
- Relative actor-versus-frozen-map discrepancy median:
  explicit `0.9738`, implicit `0.9853`.
- Relative cross-K final-action discrepancy median:
  MPI-1/2 `0.4761`, MPI-1/3 `0.5005`, MPI-2/3 `0.3563`.
- The frozen-map semigroup defect is small at short steps and grows with
  total step size. For example, explicit medians are `0.0037`, `0.0070`,
  and `0.0143` at totals `0.1`, `0.2`, and `0.4`.

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

## Reproduction

From the repository root, using a Python environment with JAX and the
training dependencies:

```bash
python scripts/diagnostics/run_actor_semigroup.py \
  --out-dir sweep_results/diagnostics/actor_path_semigroup
```

The manifest records the exact archived arguments and selected dataset
indices. Reproduction also requires the checkpoint and dataset trees named
there.
