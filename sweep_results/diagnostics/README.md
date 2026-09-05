
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
| `hosts/offrl/target_policy_exposure/` | 180-checkpoint P4 target-branch audit: nine tasks, five budgets, seeds 0--3 |
| `p1_target_value_audit/` | Protocol-v2 TD3/P3/P4 same-next-state geometry, target-value perturbation, and comparator residuals |
| `p2_relu_residence_final/` | Clean-source verified ReLU-region residence: six tasks, two seeds, 6144 anchors |
| `hosts/ext_csh/target_policy_exposure/` | 80-checkpoint P2/P3 extension: four tasks, five budgets, seeds 2--3 |
| `hosts/ext_csh/matched_geometry/` | 80-run seed-2/3 geometry sensitivity audit using a separate \(T=.05\) reference critic |
| `simulator_calibration/` | Exploratory 60-checkpoint simulator-reference screening |
| `route_shadow/` | Eight-run K=3 shadow-critic audit (archived manuscript snapshot) |
| `route_shadow_k4/` | Eight-run K=4 route-native simulator audit; 5/8, p=.0625, excluded |
| `frozen_critic_small_step/` | Retired explicit/implicit local-consistency archive; provenance only |
| `fixed_operator_order/` | Retired 0/18 fixed-operator scaffold; superseded by ReLU residence |
| `actor_path_semigroup/` | Archived endpoint-continuation diagnostic; excluded from manuscript error claims |
| `hosts/offrl/actor_path_semigroup/` | Archived mixed-normalization host extension; excluded from depth inference |
| `environment_t_sensitivity.csv` | Return versus realized first-hop displacement by environment |
| `control_final_scores.csv`, `control_summary.csv` | Targeted compute-matched and actor-target-lag results |
| `bar_mcep_p3_paired/` | 90-pair MCEP-inspired control: completeness audit, compact scores, summary, and raw-source hashes |
| `hosts/ext_csh/p0_manifest_tail90/` | Frozen-manifest entries 91--180; five optimizer-diverged Walker2d-expert cells retained |
| `p0_bar_p4_vs_two_actor_p4/` | Complete 90-pair score merge; numeric unresolved, but locked-P0 inadmissible because host stacks differ |
| `p0_bar_p4_vs_two_actor_p4/tail90_final_scores.csv` | ext_csh first-actor recovery; does not repair the primary gate |
| `i234_first_endpoint/` | Unmixed first/endpoint recovery: ext_csh I2/I3 s23 + P0 I4; ext_csv I2/I3 s01; historical I4-504 coverage only (first-actor 4/504) |
| `*_MANIFEST.json` | Audit protocol and provenance; host manifests are not historical training launch files |

## Manuscript-facing checks

- On the protocol-v2 same-next-state audit, BAR-P3 lowers target displacement
  versus TD3+BC in 88/90 cells (median ratio `.650`), while final displacement
  is lower in only 37/90 (`1.060`). BAR-P4 gives 89/90 (`.565`) and 44/90
  (`1.025`). Relative to the paired next dataset action, every P3 and P4 cell
  has a larger final- than first-actor RMS displacement.
- Under the matched TD3+BC target critic and identical smoothing, operational
  target-value perturbation is lower in 86/90 P3 and 88/90 P4 cells, with
  median ratios `.717` and `.617` and all nine task medians below one. This is
  critic-space sensitivity, not critic accuracy or return causality.
- The 180-checkpoint P4 target-branch dump uses seeds 0--3. Its comparisons use
  the 90 seed-0/1 cells matched to TD3+BC and P3: P4 is lower in 89/90 and
  85/90. P1 supplies the separate full 90-cell seed-0/1 final-actor geometry.
- The restricted four-environment seed-2/3 extension lowers P3 target-action
  displacement versus P2 in 36/40 cells (median ratio `.905`); it is a
  sensitivity check.
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
  published-MCEP reproduction, or re-centering-only ablation. It supports
  separation as a sufficient P3 candidate mechanism.
- The P4 two-actor experiment completed 90 pairs at score level: BAR minus
  control is `-2.3076`, task-bootstrap interval `[-7.2476, 2.4118]`, paired
  median `.6811`, and wins `55/34/1`, hence numeric `unresolved`. It is retained
  only as a descriptive sensitivity because the two host shards use materially
  different dependency stacks and five runs have nonfinite optimizer states.
- Ten Walker2d-expert control cells used the documented CPU recovery chain:
  eight resumed from matching logged emergency-checkpoint steps and two
  started there from zero. Removing that entire task gives +1.8278, which is
  not a hardware effect estimate.
- The verified ReLU audit gives task-equal full-horizon residence
  `.954/.911/.842/.748` at probe horizons `{.025,.05,.1,.2}`; at the `.2`
  horizon the task range is `.310`--`.980`. It validates the prevalence and
  heterogeneity of the local
  affine scope, not a learned order slope or live-chain certificate.
- The actor-path bundles are retained for provenance only. Their changing
  critics, scales, eligibility masks, and optimization histories prevent a
  controlled semigroup-error interpretation. The fixed-operator scaffold is
  also retired; ReLU-region residence is the replacement scope diagnostic.
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

Use `--require-p2-compact` for a submission build that must fail unless the
clean-source final-v2 compact archive is present and fully reconstructed.

Pass `--manuscript-dir /path/to/aistats26_manuscript` to check the released
score aggregates and archived manuscript claims. Pass `--mcep-source-root /path/to/MPI_sweep` on the source host to rehash
all 180 MCEP-control configs, evaluations, final checkpoints, and available
logs.
Without that optional path, the verifier still recomputes every compact score
claim and checks the source manifest. Host-specific dumps remain auditable from
their CSVs and manifests but are not all covered by this verifier.

## Provenance and exclusions

Files were copied or derived from local audits completed on 2026-08-29 through
2026-09-02. `matched_geometry/lin2_run_movement.csv` and
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
