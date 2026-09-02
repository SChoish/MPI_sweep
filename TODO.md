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
[`scripts/experiments/P0_BAR_TWO_ACTOR_P4_RUNBOOK.md`](scripts/experiments/P0_BAR_TWO_ACTOR_P4_RUNBOOK.md).
On `ext_csv`, a 180-cell manifest was frozen at `29fea94` and the first 90
keys finished at 1M (`45` BAR-P4 + `45` two-actor-P4 `eval.csv`). The second
90 were cancelled on this host and are not a released compact archive. No
paired contrast, decision label, or manuscript promotion has passed yet.

- [x] Generalize the existing control to a target coefficient `T/4` and a
  directly data-anchored deployment coefficient `T`. Keep independent actors;
  route only the target actor into Bellman backups. Call it a two-actor
  policy-separation control, not a published-MCEP reproduction.
- [x] Before launch, freeze a manifest with resolved configs, source revision
  and file hashes, dependency lock, dataset and normalization hashes, hardware,
  evaluation contract, output schema, and exact expected keys. Resume only
  from a checkpoint whose config and weights hashes match the manifest.
- [ ] Run exactly nine paper tasks, `T={4,7,10,14,20}`, and seeds 0--1 to one
  million critic updates with ten final evaluation episodes: 90 control runs.
  Host progress: `45/90` two-actor-P4 finals on `ext_csv` (first half of the
  frozen 180-key order). Do not add seeds 4--7 and do not expand to a
  P2/P3/P4 factorial before this comparison is read.
- [ ] Rerun the same 90 BAR-P4 cells contemporaneously from the same clean code
  snapshot and environment lock. Host progress: `45/90` BAR-P4 finals on
  `ext_csv` paired with the first-half control keys. Historical P4 scores may
  be a marginal sensitivity, but cannot be spliced into cellwise pairing
  because individual collapse trajectories are unstable across reruns.
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
not critic error, dataset support, or TD-target value perturbation. The exact
270 candidate paths for headline TD3, BAR-P3, and BAR-P4 are now present on this
host. Protocol-v2 preflight resolved and fingerprinted all 270 in an external
bundle without writing scientific outputs. Path completeness and compatibility
admission do not prove historical method identity or make the audit a result.

Implementation status is separate: the fail-closed runner and independent
artifact verifier are packaged in
[`scripts/diagnostics/P1_TARGET_VALUE_RUNBOOK.md`](scripts/diagnostics/P1_TARGET_VALUE_RUNBOOK.md).
They require the exact 270-checkpoint grid before writing scientific outputs.
The input preflight has passed; no P1 scientific run or independent verification
has passed the full inclusion gate on this host.

- [x] Package and test the exact-grid runner, frozen protocol, and independent
  verifier without substituting partial checkpoints.

### A. Direct TD-target-value perturbation

- [x] Inventory matched TD3, P3, and P4 final checkpoints for all nine tasks,
  `T={4,7,10,14,20}`, and seeds 0--1. Protocol-v2 preflight resolved 270/270
  and recorded weights, raw/effective configs, compatibility resolution, source,
  dataset, normalization, and selected-transition hashes in an external bundle.
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
  critic, realized $C_k$, and predecessor reference. Apply one post-hoc
  final-checkpoint step with the frozen audit Adam transform to the stored actor
  parameters and exact stored optimizer state, using the independent stored step
  for the current layout or the unique stored Adam count for an admitted legacy
  layout, then record
  `r_k = L_h,k(mu_k | mu_(k-1)) - L_h,k(mu_(k-1) | mu_(k-1))` before and after
  that update. Stop rather than approximate if the stored optimizer state or
  configuration is incompatible. Compatibility admission does not recover the
  historical live update or its training minibatch. Store `max(0,r_k)` as the
  smallest observed comparator slack satisfying the conditional margin.
- [ ] Report hop-, budget-, task-, and stability-stratified residuals with
  batch and checkpoint hashes. Do not call this a global optimality gap or a
  bound on $\varepsilon_k$; it only tests the proposition's feasible-comparator
  premise on sampled states. Exclude the sample-anchored first hop from the
  literal deterministic-policy comparator claim.

Inclusion gate: promote the direct value metric or comparator residual only if
the checkpoint manifest, common batch/noise contract, exact grid, and verifier
all pass. Null, reversed, or heterogeneous outcomes must be reported rather
than filtered.

## P2: delimit the local theory on learned ReLU critics

The actor-path and fixed-operator-order archives remain provenance only. A
learned ReLU critic is piecewise affine in action at a fixed state. Before the
first activation or action-box boundary, its action gradient is constant, so
the ideal explicit and same-region backward-Euler maps coincide exactly for
any subdivision. At a boundary the smooth-field expansion stops applying.
A nonzero learned `1/K` slope is therefore the wrong target; the replacement
audit measures where the local affine scope ends. It does not test return,
identify a global proximal solution, or certify the persistent Adam actor chain.

### Completed development gate (CPU; not manuscript evidence)

- [x] Package a create-only activation-region runner, independent verifier,
  exact smooth-quadratic oracle, and hand-derived synthetic ReLU cases. The
  retired fixed-operator scaffold stays archived as a verified 0/18 provenance
  snapshot and is not a learned result.
- [x] Bind the exposed HalfCheetah-medium `T=1`, seed-0 checkpoint, config,
  dataset, and both canonical seed-0/1 exclusion files by predeclared SHA-256
  before semantic deserialization. Their 1,022-row union is excluded; the
  bundle preserves executable source snapshots and dirty-worktree provenance.
- [x] Independently recompute the critic geometry, 384 finite-difference
  coordinates, every ReLU and action-box bracket, combined event ordering,
  layer-2 propagated rounding bounds, every CSV/JSON cell, and same-region
  affine residuals. Tampered summaries, snapshots, and provenance must fail.
- [x] Run one 64-anchor development pilot and pass its snapshot verifier. Full
  residence is 55/64, 49/64, 44/64, and 36/64 at
  `T={.025,.05,.1,.2}`; the largest same-region affine residual is
  `4.13e-13`. These are development readouts from an already exposed family:
  `scientific_admissible=false`, no learned slope, and no coverage/support
  threshold. They are not added to the manuscript.

The implementation and exact local rerun command are documented in
[`scripts/diagnostics/P2_RELU_RESIDENCE_RUNBOOK.md`](scripts/diagnostics/P2_RELU_RESIDENCE_RUNBOOK.md).

### Final scientific scope audit (CPU; pending)

- [ ] Freeze the final protocol, checkpoint/config/dataset fingerprints, and
  full exclusion inventory before sampling. Use the six Hopper and Walker2d
  paper tasks, seeds 0--1, and 512 new anchors per run: 12 bundles. Exclude the
  entire HalfCheetah family, all archived index sets, and every failed,
  superseded, or successful development-pilot index set.
- [ ] Execute all 12 bundles from one clean `origin/main` source snapshot and
  pass the snapshot verifier for each. Missing inputs, ambiguity, boundary
  incidence, non-finite rows, and box exits are retained outcomes; there is no
  minimum-residence inclusion gate.
- [ ] Report the per-run empirical survival curve
  `Pr(first activation/box exit > t)` on the unique horizons induced by
  `T={.025,.05,.1,.2}` and `K={1,2,4,8,16}`. Full residence is
  `K`-invariant and first-step residence depends only on `h=T/K`; repeated
  `(T,K)` views are deterministic aliases, not independent evidence.
- [ ] Report all 12 run curves, the two-seed task summaries, and the task-equal
  aggregate with boundary-category counts. Do not fit a learned order slope or
  turn residence into a return, global-proximal, semigroup, or live-chain claim.
- [ ] Add a compact learned-critic scope result only if the complete frozen
  grid and all independent verifiers pass. Null, low-residence, or mixed-family
  results remain reportable outcomes rather than failed integrity checks.

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
- [x] Execute that profiler for `K={1,2,3,4}` on one fixed
  hardware/software stack and archive the verified measurements under
  `sweep_results/diagnostics/actor_cost/`. On one H200/JAX 0.10.2 stack,
  compiled actor-phase dispatch rises from 1.435 ms at K=1 to 1.887 ms at
  K=4 (1.316x), while the scoped fresh-worker backend high-water mark rises
  from 9,684,736 to 18,045,440 bytes (1.863x). Timing excludes compilation
  and warmup; memory includes initialization, compilation, warmup, and trials
  and is not an isolated steady-state actor-call peak. Keep the analytical
  `O(K)` actor-work statement separate from this one-stack measurement.
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

- [ ] Add learned activation-region rates only after the complete 12-run
  Hopper/Walker bundle and every snapshot verifier pass from a clean
  `origin/main` checkout using `aistats26_manuscript/build_pdf.sh`. Until then,
  the paper states only the mathematical ReLU-region boundary, the existing
  frozen-critic local consistency, and a qualitative live-chain boundary.
- Decision lock: never restore a nonzero learned `1/K` order slope.
- Decision lock: do not restore the archived actor-path defect values or
  mixed-normalization K8 comparisons, promote the MCEP-inspired control to an
  exact re-centering or equivalence result, or add the inconclusive K4 route
  result merely because those runs exist.
