# Submission TODO

Last audited: 2026-09-02.

This file tracks work that can change a manuscript claim. A completed run is
not included automatically; it must identify the intended quantity, pass its
pre-specified integrity checks, and add evidence not already carried by a
stronger result. Method names follow [README.md](README.md); P2, P3, and P4
mean BAR-Prox with `K=2,3,4`.

Priority follows the claim dependency: identify what the method adds, measure
the proposed mechanism, validate the local numerical interpretation, then
harden release provenance and cost reporting. A lower-priority completed run
does not displace a higher-priority unresolved claim.

## P0: identify what the sequential chain adds beyond policy separation

The completed P3 two-actor control changes the paper's mechanism claim. Its
conservative target actor uses `T/3`, its data-anchored deployment actor uses
`T`, and only the target branch enters Bellman backups. Across 90 cells it
reaches the same 9/45 seed-mean collapse count as contemporaneous BAR-P3 and
has no resolved aggregate difference. Thus target/deployment separation is a
sufficient candidate mechanism at P3; sequential re-centering is not a
prerequisite for those aggregate outcomes.

### A. Two-actor P4 versus contemporaneous BAR-P4 (GPU, decisive)

Implementation status is tracked separately from experimental status. The
preview/freeze/launch/analyze/verify harness is packaged in
[`scripts/experiments/P0_BAR_TWO_ACTOR_P4_RUNBOOK.md`](scripts/experiments/P0_BAR_TWO_ACTOR_P4_RUNBOOK.md),
but no P4 manifest has been frozen and no P4 control or contemporaneous BAR
job has been launched from it.

- [x] Generalize the existing control to a target coefficient `T/4` and a
  directly data-anchored deployment coefficient `T`. Keep independent actors;
  route only the target actor into Bellman backups. Call it a two-actor
  policy-separation control, not a published-MCEP reproduction.
- [ ] Run exactly nine paper tasks, `T={4,7,10,14,20}`, and seeds 0--1 to one
  million critic updates with ten final evaluation episodes: 90 control runs.
  Do not add seeds 4--7 and do not expand to a P2/P3/P4 factorial before this
  comparison is read.
- [ ] Rerun the same 90 BAR-P4 cells contemporaneously from the same clean code
  snapshot and environment lock. Historical P4 scores may be a marginal
  sensitivity, but cannot be spliced into cellwise pairing because individual
  collapse trajectories are unstable across reruns.
- [ ] Before launch, freeze a manifest with resolved configs, source revision
  and file hashes, dependency lock, dataset and normalization hashes, hardware,
  evaluation contract, output schema, and exact expected keys. Resume only
  from a checkpoint whose config and weights hashes match the manifest.
- [ ] Primary signed contrast is BAR-P4 minus two-actor-P4. Report the
  task-equal mean over the fixed 90-cell grid, paired median, wins/ties,
  nine task means, 100,000-draw task-resampling interval, and raw-run plus
  seed-mean collapse transitions at thresholds 0, 10, 20, 30, and 40.
- [ ] Before launch, freeze an author-chosen minimum worthwhile difference of
  three normalized-return points. Classify the primary BAR-minus-control
  contrast as four-actor-procedure-supporting only if its 95% task-resampling
  interval lies above zero and its point estimate is at least +3; as
  two-actor-procedure-supporting only if the interval lies below zero and the
  estimate is at most -3; and as author-band-comparable only if the full
  interval lies inside [-3,+3]. Apply these labels in the stated order and
  take the first match. All other outcomes are unresolved. The three-point
  band is an author-defined decision threshold, not an externally
  validated equivalence margin. Collapse transitions are secondary and cannot
  override this label.
- [ ] Treat the result as a full-procedure comparison. A
  four-actor-procedure-supporting result supports incremental value for the
  four-actor chain bundle, not re-centering alone. A
  two-actor-procedure-supporting or author-band-comparable result supports
  direct coefficient/routing separation as the simpler method.

### B. Completed P3 control and optional exact causal isolation

- [x] Complete BAR-P3 and the MCEP-inspired P3 control at 90/90 final
  checkpoints and archive the exact matched intersection in
  `sweep_results/diagnostics/bar_mcep_p3_paired/`.
- [x] Verify control minus BAR as `+1.56`, task-resampling interval
  `[-1.52,5.44]`, paired median `-.42`, 52/90 BAR wins, raw collapse 21/90
  versus 23/90, and seed-mean collapse 9/45 for both. Keep this as an
  exploratory final-score/config-level control, not an equivalence result.
- [x] Preserve CPU recovery provenance: eight Walker2d-expert cells resumed
  from matching logged emergency checkpoints and two `T=20` cells began on
  CPU. The task-removal sensitivity is not a hardware-effect estimate.
- [ ] Only if a re-centering-only causal claim is still required after P4,
  implement one joint driver sharing the target actor, critic, target networks,
  minibatches, and RNG, branching only the evaluation-policy construction.
  The current separately trained controls also change actor work, anchoring,
  and critic trajectory.

## P1: measure target-value perturbation and matched final reach (CPU)

The existing headline diagnostic directly measures target-action displacement,
not critic error, dataset support, or TD-target value perturbation. The current
`ext_csv` checkpoint inventory contains the 90 contemporaneous BAR-P3 and 90
P3-control finals but no complete headline TD3/P4 inventory. Do not substitute
those reruns for the released grid. Run this audit only after the exact matched
headline checkpoints are resolved and fingerprinted.

Implementation status is separate: the fail-closed runner and independent
artifact verifier are packaged in
[`scripts/diagnostics/P1_TARGET_VALUE_RUNBOOK.md`](scripts/diagnostics/P1_TARGET_VALUE_RUNBOOK.md).
They require the exact 270-checkpoint grid before writing scientific outputs.
No P1 audit has passed that inclusion gate on this host.

- [x] Package and test the exact-grid runner, frozen protocol, and independent
  verifier without substituting partial checkpoints.

### A. Direct TD-target-value perturbation

- [ ] Inventory matched TD3, P3, and P4 final checkpoints for all nine tasks,
  `T={4,7,10,14,20}`, and seeds 0--1. Record weights, config, source, dataset,
  normalization, and selected-transition hashes; stop on any missing cell.
- [ ] On one frozen next-state batch and the existing valid-transition mask,
  reuse identical clipped target-smoothing noise for the target actor and the
  recorded next dataset action. For each checkpoint compute
  `Delta_y_RMS = gamma * sqrt(mean((min_j Q_j^-(s',mu_1^-(s')+eps)
  - min_j Q_j^-(s',a_next+eps))**2))`.
- [ ] Use the checkpoint's frozen target critic for the operational within-run
  measurement and a separately labeled common-critic sensitivity for
  cross-procedure action effects. Do not interpret either as critic accuracy;
  simulator return remains the external calibration signal.
- [ ] Archive per-checkpoint RMS, target magnitude, action displacement,
  clipping, finite-value flags, common batch/noise hashes, and exact row counts.
  Report method contrasts with task-level aggregation and the same marginal
  seed interpretation as the score grid.

### B. Same-next-state exposure and reach geometry

- [ ] On that identical `s'`, `a_next` batch compute both the deterministic
  Polyak first-actor displacement and the online final-actor displacement.
  This removes the current-state/next-state asymmetry in the P3 comparison.
- [ ] Require full final-actor geometry for the 90 matched P4 cells. The current
  180-checkpoint P4 target dump and 24-cell frontier panel are insufficient for
  a full-grid exposure--reach claim.
- [ ] Report target/final ratios jointly, their correlation with
  `Delta_y_RMS`, stable/collapsed stratification defined only by external
  return, and sensitivity to target smoothing. Movement and value perturbation
  remain mechanism readouts, not causal return effects.

### C. Realized feasible-comparator residual

- [ ] For re-centered hops $k\ge2$, use the frozen audit batch with the same
  critic, realized $C_k$, and predecessor reference. Reconstruct one
  post-hoc final-checkpoint Adam update from the stored actor parameters and
  exact stored optimizer state and step, then record
  `r_k = L_h,k(mu_k | mu_(k-1)) - L_h,k(mu_(k-1) | mu_(k-1))` before and after
  that update. Stop rather than approximate if the stored optimizer state or
  configuration is incompatible. This is not the historical live update or
  its training minibatch. Store `max(0,r_k)` as the smallest observed
  comparator slack satisfying the conditional margin.
- [ ] Report hop-, budget-, task-, and stability-stratified residuals with
  batch and checkpoint hashes. Do not call this a global optimality gap or a
  bound on $\varepsilon_k$; it only tests the proposition's feasible-comparator
  premise on sampled states. Exclude the sample-anchored first hop from the
  literal deterministic-policy comparator claim.

Inclusion gate: promote the direct value metric or comparator residual only if
the checkpoint manifest, common batch/noise contract, exact grid, and verifier
all pass. Null, reversed, or heterogeneous outcomes must be reported rather
than filtered.

## P2: validate the local theory with a fixed-operator error audit

The archived actor-path analysis is no longer manuscript evidence for a
semigroup or discretization-error claim. Its learned-endpoint comparisons use
different critics at different budgets, recompute the Q scale across calls,
retain unequal convergence/projection subsets, and include persistent actors
with different optimization histories. Some cross-depth rows also mix
normalization conventions and coverage. Exact semigroup equality is also the
wrong null for first-order Euler maps: at fixed total time, expected global
discretization error scales with substep size rather than vanishing exactly.

Keep the archived CSVs for provenance, but do not cite their numerical defects
as an error rate or depth trend.

Current implementation status: the CPU harness and 18 deterministic
state-index files are archived in
`sweep_results/diagnostics/fixed_operator_order/`; the analytic linear and
quadratic checks pass. The numerical protocol file is a preflight scaffold, not
a cleared frozen protocol. The local implementation now includes the required
code/config hashes, output schemas, state/substep diagnostics, failure
retention, error ratios, task aggregation, the 8192-microstep RK4 cap, and
preflight/final verifier paths. These final-output paths are not marked
complete until exercised on the canonical critics. No learned-critic order
result exists because the checkpoint inventory resolves 0/18 canonical `T=1`
critics. Resolve and fingerprint those exact critics before freezing the
protocol; no learned aggregate has been read.

### A. Fixed-operator convergence order (CPU, primary)

#### Locked inputs and operator

- [x] Create `sweep_results/diagnostics/fixed_operator_order/`; do not
  overwrite the archived `actor_path_semigroup` or
  `frozen_critic_small_step` bundles.
- [ ] Reuse the 18 final TD3+BC `T=1` critics identified by
  `frozen_critic_small_step/MANIFEST.json`: all nine paper environments and
  seeds 0--1 at 1M critic updates. Its absolute paths are discovery hints, not
  a sufficient checkpoint inventory.
- [ ] Before freezing the protocol, write `CHECKPOINTS.json` with each exact
  checkpoint path, environment, seed, update, config hash, weights SHA-256,
  dataset identifier, dataset-file hash, and state-normalization-statistics
  hash. Stop if any of the 18 critics or fingerprints cannot be resolved; do
  not silently substitute another checkpoint.
- [x] For environment index `j` (the zero-based position in the existing
  manifest's `environments` list) and seed `z`, recreate the existing 512
  reference pairs with `numpy.random.default_rng(20260829 + 100*j + z)`,
  sampled without replacement from dataset rows satisfying
  `max(abs(a_D)) <= .95`. The 18 selected-index files and SHA-256 values are
  archived; recheck their dataset fingerprints when the critics are resolved.
- [ ] Use normalized dataset state `s`, dataset action
  `a(0)=a_D`, action dimension `d`, action box `[-1,1]^d`,
  `epsilon=1e-6`, and x64 arithmetic for the numerical audit. Record
  checkpoint, config, normalization, state-index, precision, and code hashes.
- [ ] Freeze one critic `Q_ref` and one batch scale per environment--seed:
  `C_ref = mean_i(abs(Q_ref(s_i,a_D_i))) + epsilon`. Define
  `f_s(a)=d*grad_a Q_ref(s,a)/C_ref`. Recomputing `C` along a path is a
  separately labeled sensitivity analysis, not part of the primary estimate.
- [ ] Use total local times `T={0.025,0.05,0.1,0.2}` and
  `K={1,2,4,8,16}`, with `dt=T/K`, for explicit Euler and backward Euler.
  Hold critic, scale, states, anchors, precision, and action metric fixed as
  `K` changes.

#### Reference, solver, and independence checks

- [ ] Implement the reference RK4 trajectory on a code path independent of the
  tested Euler maps. For `N=256,512,...,4096`, compute both `N`- and
  `2N`-microstep paths. Let `R_N` be their endpoint RMS difference and let
  `M_2N` be the RMS movement of the `2N` endpoint from `a_D`. Accept the first
  `2N` satisfying `R_N <= max(1e-10, 1e-6*M_2N)` and use that endpoint as the
  reference. If none passes through 8192 microsteps, mark the cell
  reference-unstable. This test must not use either Euler error.
- [x] Validate the harness before using learned critics, with `d=2`, `C=1`,
  `a(0)=(.2,-.3)`, and `T=.2`. For `Q(a)=b^T a`, `b=(.1,-.2)`, both schemes
  must match `a(0)+2Tb` within `1e-10`. For
  `Q(a)=a^T A a/2+b^T a`, `A=diag(-.25,-.5)`, `b=(.1,.05)`, compare with
  `exp(2AT)a(0)+(2A)^{-1}(exp(2AT)-I)2b`; both fitted slopes over
  `K={4,8,16}` must lie in `[.9,1.1]`.
- [ ] Solve each backward-Euler equation
  `a_next=a_prev+dt*f_s(a_next)` by fixed-point iteration initialized only at
  `a_prev`, with x64 infinity-norm equation residual at most `1e-10`, delta at
  most `1e-10`, and 1000 iterations maximum. Do not restart from alternative
  guesses or choose among roots after seeing errors. Store residual, delta,
  iterations, convergence, projection, and non-finite flags for every state
  and substep.
- [ ] Store the unprojected path. An out-of-box Euler candidate at any substep,
  or any out-of-box stage or endpoint of the accepted RK4 path, makes that
  state ineligible for the primary projection-free analysis; projected results
  may be reported only as a sensitivity.

#### Common mask, errors, and aggregation

For each environment--seed--`T`, define one primary mask `I` as the
intersection across all five `K` values, both Euler schemes, and the RK4
reference. A state is eligible only if every value is finite, every implicit
substep converges, and no path requires projection. The mask is shared by all
curves for that cell, not recomputed by scheme or `K`.

- [ ] Require `|I| >= 410` (80% of 512) for a primary cell. Report every
  exclusion reason. Fewer than three eligible `T` cells for an
  environment--seed makes that run incomplete for the order summary.
- [ ] For scheme `m` and substep count `K`, compute
  `E_abs(m,K)=sqrt(mean_{i in I}(||a_m,K,i(T)-a_ref,i(T)||_2^2/d))`.
  Define
  `M_ref=sqrt(mean_{i in I}(||a_ref,i(T)-a_D,i||_2^2/d))` and
  `E_rel=E_abs/max(M_ref,1e-8)`.
- [ ] Report `E_abs(K)/E_abs(2K)` for `K={1,2,4,8}` and fit
  `log(E_abs)` against `log(T/K)` over `K={4,8,16}` separately for each
  scheme and environment--seed--`T`. If any fitted error is below `1e-12`,
  report a below-floor result instead of inventing a slope.
- [ ] Take the median slope over eligible `T` values within each of the 18
  environment--seed runs. States are not independent replicates. Report all 18
  run summaries, the nine two-seed task means, and the task-equal aggregate.

#### Integrity and scientific outcome gates

- [ ] Before running learned critics, write `FROZEN_PROTOCOL.json` containing
  the grid, equations, hashes, tolerances, mask definition, coverage rule,
  formulas, output schema, and expected row counts. Do not change it after
  reading aggregate errors. Write each realized mask and its SHA-256 only to
  the result manifest after execution.
- [ ] The compact endpoint table must contain exactly
  `18*4*5*2=720` unique cells; the raw state table must contain
  `720*512=368640` rows before masking. The local implicit-substep diagnostic
  must contain `18*4*512*(1+2+4+8+16)=1142784` records; it may be stored as a
  hashed compressed array rather than committed as CSV. Missing, duplicate,
  schema, fingerprint, or analytic-harness failures stop the audit. Numerical
  state/cell failures remain recorded, cannot be silently filtered, and count
  against the coverage and completeness gates above.
- [ ] Treat numerical order as a scientific outcome, not an integrity check.
  Pre-specified support requires, for each scheme, an aggregate median run
  slope in `[.75,1.25]`, at least 14/18 run slopes in `[.5,1.5]`, and at
  least 15/18 complete runs. Otherwise report the result as failed or
  inconclusive.
- [ ] Add the exact grid, masks, fingerprints, expected counts, and recomputed
  summary claims to `scripts/verify_release_results.py`.

Expected interpretation: under this frozen, projection-free operator, both
schemes should have first-order global error, approximately `O(1/K)`. This
tests the local numerical proposition. It does not predict return or certify
the live actor chain.

### B. Live-path deviation (CPU, secondary and non-blocking)

This diagnostic is optional for submission and may run after section A passes
its integrity gates, regardless of whether the learned-critic order outcome is
positive, null, or mixed.

- [ ] Instrument a paired replay from one identical pre-update snapshot that
  contains the distinct persistent actors and their histories, the critic and
  targets, optimizer states, RNG, and minibatch. Do not compare independently
  trained budget checkpoints as if they were one operator.
- [ ] With the critic frozen, compare three paths: the ideal action-space map, a
  predecessor-warm-start parametric actor chain, and the actual persistent
  actor chain. Define one hop as one actor update with `dt=T/K`; reset every
  branch from the same snapshot before replay.
- [ ] Measure hop-wise absolute and movement-normalized deviations, the anchor
  gradient at `dt=0`, projection, Q scale, optimizer-step norm, and implicit
  residual. Use the same states and denominator floor as section A.
- [ ] Keep a live-critic replay, if added, as a sensitivity analysis. It changes
  the operator and cannot enter the fixed-operator convergence estimate.
- [ ] Describe endpoint-versus-frozen-continuation distance as a bridge
  residual, not a semigroup defect. Its purpose is to locate the gap between
  ideal action-space geometry and the implemented one-Adam-step chain.

The separate existing local-consistency result is the archive at
`sweep_results/diagnostics/frozen_critic_small_step/`; it is not the retired
actor-path bundle.

## P3: release provenance and practical cost

- [x] Add fixed-budget, descriptive grid-best, leave-one-dynamics-family-out
  budget-transfer, and task-then-independent-seed sensitivity summaries to the
  manuscript and verifier. They demonstrate a practical regime but do not
  create an offline `T,K` selection rule.
- [x] Release a clean runnable implementation of the recovered two-actor
  control mode. It preserves the BAR base evaluation columns for the target
  actor, writes separate deployment-actor columns, and validates method and
  depth on resume. This is a compatible recovered implementation, not proof of
  the exact historical source revision.
- [ ] Recover or otherwise verify the exact historical P3 control source
  revision. Existing run configs omit the Git revision, so the released source
  and recorded file digests cannot retroactively establish identity.
- [x] Package the fail-closed actor-update timing and accelerator-memory
  profiler plus an independently coded verifier. See
  [`scripts/diagnostics/ACTOR_COST_RUNBOOK.md`](scripts/diagnostics/ACTOR_COST_RUNBOOK.md).
- [ ] Execute that profiler for `K={1,2,3,4}` on one fixed hardware/software
  stack and archive the verified measurements. Keep the analytical `O(K)`
  actor-work statement, but do not replace measurements with asymptotic
  complexity.
- [ ] For all new runs, retain resolved YAML/JSON, source revision, environment
  lock, dataset and normalization hashes, host/accelerator inventory, and a
  compact final-checkpoint manifest. Historical seeds 2--3 limitations remain
  explicit rather than reconstructed from post-hoc P4 artifacts.

- Recorded evidence: the restricted four-environment seed-2/3 target-action
  extension is complete and remains a sensitivity check (P3 lower than P2 in
  36/40 cells; median ratio `.905`).
- Recorded exclusion: the eight-run K4 route-native simulator audit is complete
  but misses its pre-specified two-sided sign-flip rule (5/8 positive,
  `p=.0625`). Keep it outside the manuscript.
- Decision lock: do not resume seeds 4--7, restore archived actor-path defect
  values or mixed-normalization K8 comparisons, or add a completed experiment
  merely because it exists.

## Manuscript gate

- [ ] Add operator-order results only after the fixed-operator bundle and
  verifier pass from a clean `origin/main` checkout using
  `aistats26_manuscript/build_pdf.sh`. Until then, the paper should claim local
  explicit--implicit agreement from the existing frozen-critic audit and keep
  the live-chain boundary qualitative.
- Decision lock: do not restore the archived actor-path defect values or
  mixed-normalization K8 comparisons, promote the MCEP-inspired control to an
  exact re-centering or equivalence result, or add the inconclusive K4 route
  result merely because those runs exist.
