
# Diagnostic Results

This directory contains compact archived and host-specific BAR diagnostics.
Complete sweep score tables remain under `K=*/`; diagnostics are separate so
they cannot enter complete-grid aggregates accidentally. Top-level archives
support manuscript claims; host dumps enter the manuscript only where
explicitly identified.

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
folders below in place; see ARTIFACT_MAP.

## Contents

| Path | Scope |
| --- | --- |
| `matched_geometry/` | Archived proximal/linearized movement, critic-error, and common-critic figure inputs |
| `target_policy_exposure/` | Archived 270-checkpoint TD3/P2/P3 audit: nine tasks, five budgets, seeds 0--1 |
| `hosts/offrl/target_policy_exposure/` | 180-checkpoint P4 target-branch audit: nine tasks, five budgets, seeds 0--3; no final-actor metric |
| `hosts/ext_csh/target_policy_exposure/` | 80-checkpoint P2/P3 extension: four tasks, five budgets, seeds 2--3 |
| `hosts/ext_csh/matched_geometry/` | 80-run seed-2/3 geometry sensitivity audit using a separate \(T=.05\) reference critic |
| `simulator_calibration/` | Exploratory 60-checkpoint simulator-reference screening |
| `route_shadow/` | Eight-run K=3 shadow-critic audit (archived manuscript snapshot) |
| `route_shadow_k4/` | Eight-run K=4 route-native simulator audit; 5/8, p=.0625, excluded |
| `frozen_critic_small_step/` | Explicit/implicit local-consistency audit on 18 frozen critics |
| `fixed_operator_order/` | In-progress fixed-operator audit: harness and state indices ready; protocol scaffold needs TODO reconciliation; 0/18 critics resolved |
| `actor_path_semigroup/` | Archived endpoint-continuation diagnostic; excluded from manuscript error claims |
| `hosts/offrl/actor_path_semigroup/` | Archived mixed-normalization host extension; excluded from depth inference |
| `environment_t_sensitivity.csv` | Return versus realized first-hop displacement by environment |
| `control_final_scores.csv`, `control_summary.csv` | Targeted compute-matched and actor-target-lag results |
| `bar_mcep_p3_paired/` | 90-pair MCEP-inspired control: completeness audit, compact scores, summary, and raw-source hashes |
| `*_MANIFEST.json` | Audit protocol and provenance; host manifests are not historical training launch files |

## Manuscript-facing checks

- BAR-Prox P3 lowers deterministic target exposure versus TD3+BC in 88/90
  matched cells; its final-policy proxy is lower in 36/90.
- The P4 target-branch dump lowers exposure in 89/90 matched comparisons with
  TD3+BC and 85/90 with P3. It contains no P4 final-actor metric.
- The restricted four-environment seed-2/3 extension lowers P3 exposure versus
  P2 in 36/40 cells (median ratio `.905`); it is a sensitivity check.
- Route-native shadow evaluation yields larger final-route error in all eight
  run aggregates (mean `.124`, sign-flip `p=.0078`). Under the common
  first-route continuation the difference is positive in 6/8, but misses the
  pre-specified rule (mean `.116`, `p=.164`). Neither comparison estimates a
  return effect.
- Compute-matched sequential, fixed-reference, and direct-final controls score
  `71.3852`, `69.9662`, and `59.4114`, with collapse counts `2/8`, `2/8`,
  and `3/8`; slower K1 target averaging does not rescue the selected cells.
- The MCEP-inspired control and contemporaneous BAR-P3 rerun cover an exact
  90-pair grid. Control minus BAR is +1.5552, with task-resampling interval
  [-1.5187, 5.4433]; its paired median is -0.4187, and BAR wins 52/90.
  Both procedures have 9/45 seed-mean collapses. This is a bundled
  policy-separation mechanism control, not an equivalence result,
  published-MCEP reproduction, or re-centering-only ablation.
- Ten Walker2d-expert control cells used the documented CPU recovery chain:
  eight resumed from matching logged emergency-checkpoint steps and two
  started there from zero. Removing that entire task gives +1.8278, which is
  not a hardware effect estimate.
- Fifteen of 18 frozen-critic runs have an eligible converged, projection-free
  local point; every eligible run has maximum local discrepancy below `.027`.
- The actor-path bundles are retained for provenance only. Their changing
  critics, scales, eligibility masks, and optimization histories prevent a
  controlled semigroup-error interpretation. The replacement fixed-operator
  order audit is specified in [`../../TODO.md`](../../TODO.md).
- In the restricted four-environment seed-2/3 geometry extension, own-critic
  gain is positive in 80/80 runs. Under the separate TD3+BC \(T=.05\)
  reference critic it is positive in 45/45 stable runs and negative in 35/35
  collapsed runs.

Here, collapse means final normalized return below 20. It is descriptive, not
a theoretical threshold.

## Verification

From the repository root:

```bash
python3 scripts/verify_release_results.py
```

Pass `--manuscript-dir /path/to/aistats26_manuscript` to check the released
score aggregates and archived manuscript claims. Pass `--mcep-source-root /path/to/MPI_sweep` on the source host to rehash
all 180 MCEP-control configs, evaluations, final checkpoints, and available
logs.
Without that optional path, the verifier still recomputes every compact score
claim and checks the source manifest. Host-specific dumps remain auditable from
their CSVs and manifests but are not all covered by this verifier.

## Provenance and exclusions

Files were copied or derived from local audits completed on 2026-08-29 through
2026-08-31. `matched_geometry/lin2_run_movement.csv` and
`matched_geometry/hopper_lin_frontier.csv` are compact action-displacement
summaries derived from the original per-state NPZ audits; formulas and source
directories are recorded in `figure_inputs_MANIFEST.json`. Large checkpoints,
cloned states, dataset-index arrays, and per-state rollouts are excluded.
Manifests retain original paths as provenance records and are not portable run
commands.

The proximal `K=4/Imp/seed0.csv`--`seed3.csv` tables form a complete nine-task,
fourteen-budget grid. Training checkpoints are not committed, but the
`hosts/offrl/` bundle contains compact post-hoc outputs derived from all 504
local K4 checkpoints, including the 180-cell exposure audit. Exact historical
per-run training configurations for seeds 2--3 were not retained; host
manifests document audit execution and do not recreate those launches.
