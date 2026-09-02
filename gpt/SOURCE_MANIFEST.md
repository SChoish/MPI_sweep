# Source and Claim Manifest

## Canonical manuscript and implementation sources

- Prior AISTATS scientific source: `../aistats26_manuscript/`
- Trainer and dataset code: `../train_td3bc.py`, `../d4rl_data.py`
- Proximal four-seed score matrices: `../sweep_results/K={1,2,3,4}/Imp/seed{0,1,2,3}.csv`
- Projected-linearized score matrices: `../sweep_results/K={2,3}/Exp/`
- Current submission TODO/provenance ledger: `../TODO.md`

## Decisive control evidence

### P3 direct two-actor control

`../sweep_results/diagnostics/bar_mcep_p3_paired/`

Supported use:
- directly data-anchored target/deployment separation reaches the P3 aggregate regime without sequential re-centering;
- not an equivalence test and not a reproduction of published MCEP;
- exact historical Git revision was never persisted, so source identity is limited to retained file digests and the documented recovered implementation.

### P0 BAR-P4 versus two-actor-P4

`../sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4/`

Primary score-level facts:
- 90 matched pairs / 180 completed finals;
- BAR minus two-actor task-equal mean `-2.3075524705`;
- 95% task-resampling interval `[-7.2476019091, 2.4117874956]`;
- paired median `+0.6810655715`;
- BAR wins/ties/two-actor wins `55/1/34`;
- predeclared label `unresolved`.

Scope limitation:
- frozen cross-host score-level merge;
- `official_ckpt_bound_analyze_verify=false` in the score archive;
- do not describe the result as equivalence or as a verified chain/control superiority result.

## Mechanism diagnostics

### P1 target-value audit

`../sweep_results/diagnostics/p1_target_value_audit/`

Covers 270 TD3/P3/P4 checkpoints over 9 tasks, 5 high budgets, and seeds 0--1. Scientific outputs include:
- `td_target_value.csv` — same-noise within-run target-value RMS perturbation;
- `same_next_state_geometry.csv` — first/final geometry on the same next-state batch;
- `geometry_correlations.csv` — stable/collapsed geometry/value correlations;
- `comparator_residuals.csv`, `residual_aggregates.csv` — post-hoc feasible-comparator residual diagnostics;
- `VERIFY.json` and manifest files — audit integrity evidence.

Supported use: target-value perturbation is a critic-sensitive diagnostic distinct from action displacement. It is not critic accuracy and does not cause or certify environment return.

### P2 learned-ReLU local-scope audit

`../sweep_results/diagnostics/p2_relu_residence_final/`

Supported use: pooled full-region residence `0.954/0.911/0.842/0.748` at local horizons `.025/.05/.1/.2` delimits the local affine interpretation. Do not infer a learned convergence-order slope or global proximal solution.

### Critic-failure and simulator audits

Use the compact diagnostic artifacts under `../sweep_results/diagnostics/` only with their stated scopes. The common critic is a cross-critic diagnostic, not ground truth; simulator-reference results are targeted checkpoint-level calibration evidence.

## Practical cost

Actor-cost artifacts under `../sweep_results/diagnostics/` support one-stack H200/JAX 0.10.2 measurements: compiled actor-phase time about `1.435 -> 1.887 ms` and scoped backend high-water memory about `9.68 -> 18.05 MB` from K1 to K4. Do not report these as whole-training wall-clock or total memory.

## Claims intentionally excluded

- Sequential re-centering is necessary for the stability gain.
- BAR-P4 is superior to direct two-actor separation.
- Two-actor-P4 is superior or equivalent to BAR-P4.
- Target-action displacement is dataset-support distance, critic error, or a return certificate.
- The live persistent Adam chain is an exact Wasserstein proximal flow.
- The learned ReLU audit establishes a global `1/K` discretization law.
