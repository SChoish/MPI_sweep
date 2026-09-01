# Submission TODO

Last audited: 2026-09-02.

This file tracks work that can change a manuscript claim. A completed run is
not included automatically; it must identify the intended quantity, pass its
pre-specified integrity checks, and add evidence not already carried by a
stronger result. Method names follow [README.md](README.md); P2, P3, and P4
mean BAR-Prox with `K=2,3,4`.

## P0: replace the actor semigroup test with an operator-error audit

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
a cleared frozen protocol: it still needs the required code/config hashes, full
output schema and substep diagnostics, error-ratio/task aggregation, and the
RK4 cap reconciled with this specification. No learned-critic order result
exists because the checkpoint inventory resolves 0/18 canonical `T=1`
critics. Resolve and fingerprint those exact critics only after reconciling and
regenerating the protocol; no learned aggregate has been read.

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

## Result-inclusion gates

Here MCEP means the separately trained two-actor policy-separation control:
its target branch uses coefficient `tau/3`, its data-anchored evaluation branch
uses `tau`, and only the target branch enters Bellman backups. It is a mechanism
control inspired by MCEP, not a reproduction of the published algorithm.

- [x] Complete BAR-P3 and MCEP-inspired P3 at 90/90 final checkpoints and
  final evaluations with an exact environment--budget--seed intersection.
- [x] Archive paired scores, independently recomputed summary statistics, all
  raw config/eval/final-checkpoint hashes, and recovery provenance in
  `sweep_results/diagnostics/bar_mcep_p3_paired/`. Eight Walker2d-expert
  cells resumed from matching logged emergency-checkpoint steps on the CPU
  recovery path; the two `T=20` cells began there from zero.
- [x] Add the result to the main paper and supplement as an exploratory
  final-score/config-level mechanism control. Preserve the signed estimand:
  control minus contemporaneous BAR is +1.56 with task-resampling interval
  [-1.52, 5.44], paired median -0.42, and 52/90 BAR wins. Do not turn the
  interval into an equivalence claim.
- [ ] Release a clean runnable snapshot of the exact control-mode training
  changes. The run configs did not record a Git revision; `SUMMARY.json` records
  the two local source-file hashes, but hashes alone are not runnable code.
  Preserve the existing BAR evaluation-column contract and strengthen
  checkpoint config validation before shipping the implementation.
- [ ] If a re-centering-only causal control becomes necessary, implement a
  joint driver that shares the target actor, critic, target networks,
  minibatches, and RNG in one process and branches only the evaluation policy.
  The current separate-run control changes actor work, anchoring, and the
  realized critic trajectory.

- Recorded evidence: the restricted four-environment seed-2/3 target-exposure
  extension is complete and is used only as a sensitivity check. P3 is lower
  than P2 in 36/40 cells, with median ratio `.905`.
- Recorded exclusion: the eight-run K4 route-native simulator audit is complete
  but misses its pre-specified two-sided paired sign-flip permutation rule
  (5/8 positive, `p=.0625`). It remains outside the manuscript.
- [x] Archive compact K4 route-native artifacts in `sweep_results/diagnostics/route_shadow_k4/`; keep them excluded from the manuscript under the pre-specified rule.
- Decision lock: do not resume the stopped seeds 4--7 factorial plan unless a
  later review identifies a claim that the existing four-seed grids and
  targeted controls cannot answer.

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
