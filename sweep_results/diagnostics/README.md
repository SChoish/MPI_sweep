# Diagnostic Results

This directory contains the compact, auditable outputs used by the MPI
manuscript. Complete sweep score tables remain under `K=*/`; diagnostics are
separate so they cannot enter complete-grid aggregates accidentally.

**Per-host post-hoc dumps:** package under
[`scripts/diagnostics/`](../../scripts/diagnostics/)
([README](../../scripts/diagnostics/README.md),
[HOST_RUNBOOK](../../scripts/diagnostics/HOST_RUNBOOK.md),
[ARTIFACT_MAP](../../scripts/diagnostics/ARTIFACT_MAP.md)).

```bash
bash scripts/diagnostics/run_host.sh --host "$(hostname -s)" --seeds 0 1 --job ckpt
```

Write CSVs under `hosts/<hostname>/` on the machine that owns the checkpoints,
then push only those compact outputs. This does **not** rebuild the archived
folders below in place — see ARTIFACT_MAP.

## Contents

| Path | Scope |
| --- | --- |
| `matched_geometry/` | Proximal/linearized movement, effective-step, critic-error, common-critic, and data-derived figure inputs |
| `target_policy_exposure/` | Next-state Polyak-target displacement for 270 audited runs |
| `simulator_calibration/` | Exploratory 60-checkpoint simulator-reference screening |
| `route_shadow/` | Paired first-hop versus final-hop shadow-critic alignment against the common \(Q_{\mathrm{MC}}^{\rho_1}\) first-route estimand on eight targeted runs |
| `frozen_critic_small_step/` | Explicit/implicit local-consistency audit on 18 frozen critics |
| `environment_t_sensitivity.csv` | Return versus realized first-hop displacement by environment |
| `control_final_scores.csv` | Run-level compute-matched and actor-target-lag scores |
| `control_summary.csv` | Means, medians, and descriptive collapse counts |
| `*_MANIFEST.json` | Original protocol and launch metadata |

## Manuscript-facing checks

- MPI-Prox `K=3` lowers deterministic target-policy displacement versus
  TD3+BC in 88/90 paired runs; its final-policy proxy is lower in 36/90.
- Final-hop shadow routing has higher symmetric relative error against the
  common \(Q_{\mathrm{MC}}^{\rho_1}\) first-route estimand in all eight
  run-level comparisons (mean delta `0.1235147`, exact sign-flip
  `p=0.0078125`).
- Compute-matched sequential, fixed-reference, and direct-final controls have
  means `71.3852`, `69.9662`, and `59.4114`, with collapse counts
  `2/8`, `2/8`, and `3/8`.
- Canonical and three-times-slower actor-target controls have means `13.5641`
  and `12.0090`, with collapse counts `6/8` and `7/8`.
- Fifteen of 18 frozen-critic runs have an eligible converged, projection-free
  local point; their maximum run-level local discrepancy is below `0.027`.

Here, collapse means final normalized return below 20. It is descriptive, not
a theoretical threshold.

## Verification

From the repository root:

```bash
python3 scripts/verify_release_results.py
```

Pass `--manuscript-dir /path/to/aistats27_manuscript` to also confirm that
the corresponding numerical claims and separate-machine wording are present
in the LaTeX source.

## Provenance and exclusions

Files were copied or derived from local audits completed on 2026-08-29 and
2026-08-30. `matched_geometry/lin2_run_movement.csv` and
`matched_geometry/hopper_lin_frontier.csv` are compact action-displacement
summaries derived from the original per-state NPZ audits; their formulas and
source directories are recorded in `figure_inputs_MANIFEST.json`. Large
checkpoints, cloned states, dataset-index arrays, and per-state rollouts are
excluded. Manifests retain original paths as provenance records and are not
portable run commands.

The `K=4/Imp/seed*.csv` files came from a separate machine. They are outside
this local diagnostic bundle and the two-seed complete-grid aggregate.
Hopper-medium at `T=20` is highly variable across its four transferred seeds,
so the manuscript treats `K=4` as a targeted stability-window extension.
