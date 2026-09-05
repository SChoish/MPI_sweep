# Submission TODO

Last audited: 2026-09-05.

This file tracks work that can change a manuscript claim. A completed run is
not included automatically; it must identify the intended quantity, pass its
pre-specified integrity checks, and add evidence not already carried by a
stronger result. The manuscript now uses MART / I-MART; P2, P3, and P4 below
retain the archive's PART-Prox names for `K=2,3,4`. Archived paths and
configuration fields retain historical `PART` / `BAR` identifiers.

The September 5 manuscript revision reanalyzes the original CSV bundle at
`2fe4b8f` and the additional logged/re-evaluated policy scores at `28ce69a4`.
The manuscript build itself does not launch training or re-evaluate checkpoints. Reproducible
inputs, aggregation code, and tables are in the manuscript repository's
[analysis directory](https://github.com/SChoish/PART/tree/main/analysis).

Priority follows the claim dependency: identify what the method adds, measure
the proposed mechanism, validate the local numerical interpretation, then
harden release provenance and cost reporting. A lower-priority completed run
does not displace a higher-priority unresolved claim.

The manuscript now includes the completed historical I4 504 first/endpoint
reevaluation and full P0 90-run-per-procedure recovery. The ext_csv hop-actor
CPU audit is complete for the local first90 P0 shard. Next action: keep P0.B
unlaunched until one environment-locked manifest exists; assess the TD3+BC
follow-up reported in progress by the author once its actual grid, endpoint
coverage, and evaluation protocol are available.
P0.C is a complementary independent-baseline comparison, not a prerequisite
for measuring extraction inside an existing I4 run. P0.B remains unlaunched;
its frozen shared-driver experiment awaits one environment-locked manifest.
P0.A's clean repeat is conditional on promoting a new P4 contrast to primary
evidence. The original P4
[frozen manifest](sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4/FROZEN_MANIFEST.json)
records the decision rule and integrity contract for that archive.

## P0: identify what the sequential chain adds beyond policy separation

The completed P3 two-actor control changes the paper's mechanism claim. Its
conservative target actor uses `T/3`, its data-anchored deployment actor uses
`T`, and only the target branch enters Bellman backups. Across 90 cells it
reaches the same 9/45 seed-mean collapse count as contemporaneous PART-P3 and
has no resolved aggregate difference. Thus target/deployment separation is a
sufficient candidate mechanism at P3; sequential re-centering is not a
prerequisite for those aggregate outcomes.

### A. Two-actor P4 versus contemporaneous PART-P4 (scores complete; secondary evidence)

The full grid is complete at source revision `29fea94`: nine tasks,
`T={4,7,10,14,20}`, seeds 0--1, and two procedures give 180 final
evaluations and 90 matched pairs. The merged archive is
`sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4/`. PART-P4 minus
two-actor-P4 has task-equal mean `-2.3076`, 95% task-bootstrap interval
`[-7.2476,2.4118]`, paired median `+0.6811`, and PART/control/tie counts
`55/34/1`. The predeclared `outcome_label=unresolved`.

`unresolved` describes the score contrast. Separately, the archived
`SUMMARY.json` records `scientific_admissible_under_frozen_p0_contract=false`
and `manuscript_primary_promotion_allowed=false`. Source hashes, resolved
configs, normalization, and run keys match within the recorded source revision,
but the two 45-pair host blocks use different resolved JAX/Flax stacks and the
merged archive does not meet the frozen checkpoint/receipt requirements.
Five Walker2d-expert checkpoints have nonfinite critic-optimizer states after
finite weights. All 180 final evaluations remain finite and are retained in
the descriptive comparison. Score completeness and finite evaluation do not
override the failed inclusion gate.

- [x] Run 90 two-actor-P4 and 90 contemporaneous PART-P4 cells to one million
  updates with ten final evaluation episodes.
- [x] Freeze the source revision, configs, expected keys, and per-host
  manifests; merge all 180 final scores into 90 matched pairs.
- [x] Use PART-P4 minus two-actor-P4 as the predeclared signed contrast and report
  the task-equal mean, paired median, wins/ties, task means, 100,000-draw
  task-resampling interval, and collapse transitions.
- [x] Apply the predeclared three-point decision rule. The result is
  `unresolved`; it is neither equivalence nor superiority for either
  procedure.
- [x] Retain all finite scores as a secondary full-procedure comparison. It does not
  isolate re-centering alone because actor work, anchoring, and critic
  trajectory also differ.
- [ ] Before promoting a new P4 comparison to primary evidence, repeat the
  same 180 cells on one frozen resolved stack with checkpoint-bound receipts,
  finite model and optimizer states, and an independent verifier. Retain and
  report failures under the frozen handling rule. Archive both online-first
  and endpoint evaluations with paired episode seeds. This clean repeat is
  required for renewed primary promotion, not for reporting the existing
  finite-score sensitivity.

### B. Completed P3 control and optional exact causal isolation

- [x] Complete PART-P3 and the MCEP-inspired P3 control at 90/90 final
  checkpoints and archive the exact matched intersection in
  `sweep_results/diagnostics/bar_mcep_p3_paired/`.
- [x] Verify control minus PART as `+1.56`, task-resampling interval
  `[-1.52,5.44]`, paired median `-.42`, 52/90 PART wins, raw collapse 21/90
  versus 23/90, and seed-mean collapse 9/45 for both. Keep this as an
  exploratory final-score/config-level control, not an equivalence result.
- [x] Preserve CPU recovery provenance: eight Walker2d-expert cells resumed
  from matching logged emergency checkpoints and two `T=20` cells began on
  CPU. The task-removal sensitivity is not a hardware-effect estimate.
- [x] Freeze the shared-driver protocol and implement the joint driver without
  launching. Protocol:
  [`scripts/experiments/P0_SHARED_DRIVER_PROTOCOL.json`](scripts/experiments/P0_SHARED_DRIVER_PROTOCOL.json).
  CLI: `--method shared --deployment-branch {recenter,data_anchor,fixed_ref,data_anchor_matched}`.
  Identity tests: `tests/test_shared_driver.py`. `recenter` reproduces BAR;
  `data_anchor` reproduces the two-actor control; `fixed_ref` proximal-updates
  hops `k>=2` toward the shared first actor; `data_anchor_matched` is the
  compute-controlled extra-update contrast. Do not splice historical BAR/mcep
  scores into future shared pairs.
- [ ] Launch the frozen shared-driver grid only after one resolved-stack
  `FROZEN_MANIFEST.json`. Primary contrast is shared-`recenter` minus
  shared-`data_anchor` endpoint return on nine tasks, `T={4,7,10,14,20}`,
  `K=4`, seeds 0--1. Record first-actor returns on every branch. This remains
  the first training experiment after the hop-actor audit: H6 is unresolved.

### E. Hop-actor audit on existing P0 K=4 checkpoints (CPU; first90 complete)

Local archive:
[`sweep_results/diagnostics/p0_hop_actor_audit/`](sweep_results/diagnostics/p0_hop_actor_audit/).
90/180 runs evalable on ext_csv (45 paired). Walker and halfcheetah-expert 1M
checkpoints remain on ext_csh and were not substituted. Episode repeats are
noise reduction, not extra training seeds. Oracle mid-actor scores are
diagnostics only.

- [x] Inventory P0 MART4 / two-actor-P4 1M checkpoints, hashes, actor lists,
  and coverage against the 180-run denominator.
- [x] Re-evaluate actors 1–4 and two-actor target/deployment with one
  10-episode protocol (`evaluation_seed=10000+ep`, gymnasium MuJoCo v4).
- [x] Measure adjacent-hop and mu4-vs-deploy action RMS on shared dataset
  states and on the mu4∪deploy rollout union.
- [x] Record hop-index return profiles, hop-ΔJ heatmap, and action-vs-return
  scatter without mixing archived first/endpoint columns.

Readout (local 45 pairs, two seeds): most return change is hop 1→2; at
`T∈{10,14,20}` later hops are often flat or negative. Dataset action RMS stays
nonzero on later hops (~0.08). 34/45 pairs have action RMS>0.05 with
`|J(μ4)-J(deploy)|<3`. hopper-expert mean μ4−deploy is `-13.45` against
hopper-medium `+3.82`. Cosine(δ2,δ3)=0.16, so hops are not a repeated
direction. H6 (independent first-actor/critic paths) is not isolated.

### Follow-up experiments from the hop audit (not launched)

Priority 1 — A. Shared-driver (existing P0.B above). Question: does sequential
re-centering still match direct extraction when first actor, critic, target,
minibatch, and RNG are shared? Resource: one frozen resolved stack, 90+90
cells. Status: protocol frozen, no host manifest. Supports H6 if first-actor
scores match across branches and the endpoint contrast stays small or
unresolved.

Priority 2 — B. Critic-error MC on Q-up / return-down cells. First action from
the compared actor, continuation from that critic's Bellman target policy
(Polyak actor 1 + target noise). Distinct from full-episode J(μk). Needs
simulator state restore and matching discount / termination / target noise.
Prefer hopper-expert and high-T cells where hop 3–4 ΔJ < 0. Status: design
only.

Priority 3 — C. Frozen-critic extra extraction, only if later-hop ΔJ
saturation could be optimizer shortfall. Same fixed critic, more actor steps;
report as extraction-on-frozen-Q, not as a causal account of the original run.
Distinguish `fixed_ref` (prox to actor 1 at hop size T/K) from a
remaining-horizon direct-from-first control. Status: not started.

Do not start A–C from this audit. Do not edit the manuscript from these
figures until the hop claims are chosen.

### C. Independently match the first actor's local budget (complementary baseline)

- [x] Reindex the complete four-seed I-MART/TD3+BC grid by `h=T/K` and report
  all nine exact intersections, without interpolation or outcome filtering.
  Eight have positive task-equal differences; four task-resampling intervals
  lie above zero. I4 at `T=.4` versus TD3 at `h=.1` gives `+8.69`
  `[1.58,18.09]`; I4 at `T=10` versus `h=2.5` gives `+6.94`
  `[-14.65,27.24]`. I2 at `T=14` gives `-.81`. Matching the coefficient does
  not make independently trained critics or RNG trajectories identical.
- [x] Inventory local K1 before any new training. ext_csh `results/mpi1_s23`
  has seeds 2--3 on the 14-tau I4 *total-budget* grid (252/252, `d4rl_score`
  only). None of the requested local budgets
  `h={.625,1,1.75,3,3.5,4.25,5}` are present. Receipt:
  [`sweep_results/diagnostics/i234_first_endpoint/K1_INVENTORY.json`](sweep_results/diagnostics/i234_first_endpoint/K1_INVENTORY.json).
- [ ] Complete the optional independent K1 comparison at the missing I4 local budgets:
  `h={.625,1,1.75,3,3.5,4.25,5}`, corresponding to
  `T={2.5,4,7,12,14,17,20}`. Require all nine tasks and four seeds for a
  headline contrast; existing seed-0/1-only extra budgets and the s23
  total-budget K1 grid cannot substitute. Record the actual training and
  evaluation protocol; use contemporaneous I4 comparisons when historical
  protocols cannot be matched. Freeze and report the complete requested grid.
  This is an independent-baseline check. The already trained I4 first actor
  supplies the within-run TD3+BC-form comparator at `h=T/4`, so additional K1
  training is not needed to compute endpoint-minus-first return. It does not
  isolate sequential re-centering from direct policy separation.

### D. Evaluate first-actor versus endpoint return (I4 full grid complete)

- [x] Recover both online-policy scores for all 90 runs per procedure from
  `sweep_results/diagnostics/p0_bar_p4_vs_two_actor_p4/all180_first_endpoint_scores.csv`.
  `target_d4rl` is the online first actor, not its Polyak copy. Both policies
  use the same ten evaluation seeds within each run; every endpoint matches
  the merged table. Mean endpoint-minus-first is I4 `+2.18` (61/90 positive)
  and control `+4.19` (63/90 positive), replacing the incomplete first-host
  subset in manuscript-facing summaries. Retain all nine tasks, including
  HalfCheetah-expert `-47.66/-33.92` and Hopper-expert `-21.08/-7.80`.
  Full column recovery does not repair the failed primary inclusion gate.
- [x] Use the simulator archive's complete 60-cell
  `actual_policy_value_difference_mean` as a separately labeled sampled-state
  continuation-value diagnostic: I2 `+16.13`, 22/30 positive; I3 `+24.70`,
  25/30 positive. These use raw discounted reward on cloned states, not D4RL
  episode scores or an initial-state-return estimate.
- [x] Recover the ext_csh tail90 first-actor columns from local `eval.csv`
  without re-evaluation. Compact tables:
  `tail90_final_scores.csv` and `all180_first_endpoint_scores.csv`. Endpoint
  values and eval SHA-256 match the published tail shard. Recovered columns
  do not repair the failed primary inclusion gate.
- [x] Recover the locally available I2/I3/I4 first/endpoint subset and freeze
  its inclusion grid. ext_csh has implicit I2 and I3 seeds 2--3 at 252/252,
  plus P0 BAR-P4 I4 seeds 0--1 at 90/90. Seeds 0--1 I2/I3 and historical I4
  504 are not on this host. Explicit I3 s23 is stored separately. Archive:
  [`sweep_results/diagnostics/i234_first_endpoint/`](sweep_results/diagnostics/i234_first_endpoint/).
- [x] Recover ext_csv implicit I2/I3 seeds 0--1 from local `eval.csv` without
  mixing families. Tables:
  [`i2_s01_first_endpoint.csv`](sweep_results/diagnostics/i234_first_endpoint/i2_s01_first_endpoint.csv)
  and
  [`i3_s01_first_endpoint.csv`](sweep_results/diagnostics/i234_first_endpoint/i3_s01_first_endpoint.csv)
  (252/252 each). Mean endpoint-minus-first is I2 `+5.20` (155/252 positive)
  and I3 `+5.45` (168/252 positive). Inclusion:
  [`EXT_CSV_INCLUSION_GRID.json`](sweep_results/diagnostics/i234_first_endpoint/EXT_CSV_INCLUSION_GRID.json).
  Do not concatenate with s23 or P0 tables or treat this as a four-seed
  headline.
- [x] Historical I4 504 first-actor columns cannot be recovered from
  `eval.csv` (finite `d4rl_pi4` 504/504, finite `d4rl_score` 4/504). Coverage:
  [`I4_HISTORICAL_504_COVERAGE.json`](sweep_results/diagnostics/i234_first_endpoint/I4_HISTORICAL_504_COVERAGE.json).
  Same-stack CPU re-eval of both online policies from `params_1000000.pkl`
  is complete:
  [`i4_historical_504_first_endpoint.csv`](sweep_results/diagnostics/i234_first_endpoint/i4_historical_504_first_endpoint.csv)
  (504/504). Mean endpoint-minus-first `+5.76` (367/504 positive). Summary:
  [`I4_HISTORICAL_504_REEVAL_SUMMARY.json`](sweep_results/diagnostics/i234_first_endpoint/I4_HISTORICAL_504_REEVAL_SUMMARY.json).
  Original `mpi4_norm` `eval.csv` is unchanged. Do not mix with P0 I4.
  At `T={2.5,4,7}`, the gain is `+11.43` (89/108 positive); at
  `T={10,12,14,17,20}`, it is `+0.26`. In the latter region, raw score-below-20
  counts rise from 32 first actors to 54 endpoints, with two recoveries and
  24 new failures. Report these as within-run deployment comparisons, not
  cross-depth collapse counts or proof of a re-centering-specific advantage.
  Published outputs lack a checkpoint-bound reevaluation manifest and
  individual episode returns; preserve this reproducibility limitation.

## P1: measure target-value perturbation and matched final reach (CPU)

The existing headline diagnostic directly measures target-action displacement,
not critic error, dataset support, or TD-target value perturbation. The exact
270-cell protocol-v2 audit completed on this host in
`/home/ext_csv/mpi_sweep_lab/p1-audit-20260902b/` with
`MANIFEST.json.status == "analysis_complete"`,
`finite_value_gate == true`, `VERIFY.json.pass == true`,
`artifact_integrity_pass == true`, and
`scientific_inclusion_gate_pass == true`. Compact CSVs live under
[`sweep_results/diagnostics/p1_target_value_audit/`](sweep_results/diagnostics/p1_target_value_audit/).
Path completeness still does not prove historical training-method identity.
Some cells have very large finite `Delta_y_RMS` tails; heterogeneous outcomes
are retained rather than filtered.

Implementation status is separate: the fail-closed runner and independent
artifact verifier are packaged in
[`scripts/diagnostics/P1_TARGET_VALUE_RUNBOOK.md`](scripts/diagnostics/P1_TARGET_VALUE_RUNBOOK.md).

- [x] Package and test the exact-grid runner, frozen protocol, and independent
  verifier without substituting partial checkpoints.

### A. Direct TD-target-value perturbation

- [x] Inventory matched TD3, P3, and P4 final checkpoints for all nine tasks,
  `T={4,7,10,14,20}`, and seeds 0--1. Protocol-v2 preflight resolved 270/270
  and recorded weights, raw/effective configs, compatibility resolution, source,
  dataset, normalization, and selected-transition hashes.
- [x] On one frozen next-state batch and the existing valid-transition mask,
  reuse identical clipped target-smoothing noise for the target actor and the
  recorded next dataset action. For each checkpoint compute
  `Delta_y_RMS = gamma * sqrt(mean((min_j Q_j^-(s',mu_1^-(s')+eps)
  - min_j Q_j^-(s',a_next+eps))**2))`.
- [x] Use the checkpoint's frozen target critic for the operational within-run
  measurement and a separately labeled common-critic sensitivity for
  cross-procedure action effects. Do not interpret either as critic accuracy;
  simulator return remains the external calibration signal.
- [x] Archive per-checkpoint RMS, target magnitude, action displacement,
  clipping, finite-value flags, common batch/noise hashes, and exact row counts.
  Report method contrasts with task-level aggregation and the same marginal
  seed interpretation as the score grid.

### B. Same-next-state exposure and reach geometry

- [x] On that identical `s'`, `a_next` batch compute both the deterministic
  Polyak first-actor displacement and the online final-actor displacement.
  This removes the current-state/next-state asymmetry in the P3 comparison.
- [x] Require full final-actor geometry for the 90 matched P4 cells.
- [x] Report target/final ratios jointly, their correlation with
  `Delta_y_RMS`, stable/collapsed stratification defined only by external
  return, and sensitivity to target smoothing. Movement and value perturbation
  remain mechanism readouts, not causal return effects.

### C. Realized feasible-comparator residual

- [x] For re-centered hops $k\ge2$, use the frozen audit batch with the same
  critic, realized $C_k$, and predecessor reference. Apply one post-hoc
  final-checkpoint step with the frozen audit Adam transform to the stored actor
  parameters and exact stored optimizer state, then record
  `r_k = L_h,k(mu_k | mu_(k-1)) - L_h,k(mu_(k-1) | mu_(k-1))` before and after
  that update. Store `max(0,r_k)` as the smallest observed comparator slack.
- [x] Report hop-, budget-, task-, and stability-stratified residuals with
  batch and checkpoint hashes. Do not call this a global optimality gap or a
  bound on $\varepsilon_k$.

Inclusion gate: all five recorded inclusion fields and the artifact-arithmetic
verifier passed for the 2026-09-02b bundle; the network checkpoints were not
reexecuted by that compact verifier and the historical training revision is
not embedded. Null, reversed, and heterogeneous outcomes remain archived.
For manuscript promotion, use only the matched-common-critic paired result:
P3/P4 lower operational target-value perturbation than TD3+BC in 86/90 and
88/90 cells (median ratios `.717` and `.617`; all nine task medians below one),
plus the full same-next-state geometry. Do not promote the arithmetic mean:
extreme finite critic-magnitude tails dominate it. Keep the comparator residual as a
sampled premise diagnostic, not an $\varepsilon_k$ bound.

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

### Final scientific scope audit (CPU; clean v2 complete and verified)

External create-only bundle:
`/home/ext_csv/mpi_sweep_lab/p2-relu-final-v2-06d64b3/`
(compact summary under
[`sweep_results/diagnostics/p2_relu_residence_final/`](sweep_results/diagnostics/p2_relu_residence_final/)).
Raw artifacts remain `analysis_complete_pending_verification` with
`scientific_admissible == false`; only `VERIFY.json` reports
`status == "verified_complete"`, `pass == true`, and
`scientific_admissible == true`. The run used clean `origin/main` revision
`06d64b3`, and snapshot verification plus `--check-existing` replay pass.
Task-equal residence on 6144 eligible interior anchors is
0.954 / 0.911 / 0.842 / 0.748 at
`T={.025,.05,.1,.2}`. No learned `1/K` slope is reported.

- [x] Freeze the final protocol, checkpoint/config/dataset fingerprints, and
  full exclusion inventory before sampling. Use the six Hopper and Walker2d
  paper tasks, seeds 0--1, and 512 new anchors per run: 12 bundles. Exclude the
  entire HalfCheetah family, all archived index sets, and every failed,
  superseded, or successful development-pilot index set.
- [x] Execute all 12 bundles from a fully clean `origin/main` worktree and pass the independent final verifier
  (`verify_p2_relu_residence_final.py`). Runner/verifier bytes are frozen in
  `SOURCE_SNAPSHOT/`; both full and tracked Git dirty states are false.
- [x] Report the per-run empirical survival curve
  `Pr(first activation/box exit > t)` on the unique horizons induced by
  `T={.025,.05,.1,.2}` and `K={1,2,4,8,16}`.
- [x] Report all 12 run curves, the two-seed task summaries, and the task-equal
  aggregate with boundary-category counts. Do not fit a learned order slope.
- [x] Add a compact learned-critic scope result only if the complete frozen
  grid and all independent verifiers pass. Compact archive:
  `sweep_results/diagnostics/p2_relu_residence_final/`.

## P3: release provenance and practical cost

- [ ] Recommended practical follow-up after the extraction comparisons:
  predeclare an offline-only `(T,K)` selection rule and evaluate it on held-out
  dynamics families, with equal tuning and compute allowances for TD3+BC and
  the two-actor control. Select without access to held-out environment return;
  use simulator scores only for final assessment. Compare against fixed-budget
  choices and report every family. The existing post-hoc family transfer
  analysis is a useful sensitivity, not validation of such a selection rule.

- [x] Add fixed-budget, descriptive grid-best, leave-one-dynamics-family-out
  budget-transfer, and task-then-independent-seed sensitivity summaries to the
  manuscript and verifier. They demonstrate a practical regime but do not
  create an offline `T,K` selection rule.
- [x] Release a clean runnable implementation of the recovered two-actor
  control mode. It preserves the PART base evaluation columns for the target
  actor, writes separate deployment-actor columns, and validates method and
  depth on resume. This is a compatible recovered implementation, not proof of
  the exact historical source revision.
- [x] Recover or otherwise verify the exact historical P3 control source
  revision. Existing run configs omit the Git revision, so the released source
  and recorded file digests cannot retroactively establish identity.
  Verification outcome: exact historical Git revision is verified
  unrecoverable; only file digests identify local training sources. Recorded in
  [`SOURCE_IDENTITY.md`](sweep_results/diagnostics/bar_mcep_p3_paired/SOURCE_IDENTITY.md).
  No commit hash is invented.
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
- [x] For all new runs, retain resolved YAML/JSON, source revision, environment
  lock, dataset and normalization hashes, host/accelerator inventory, and a
  compact final-checkpoint manifest. Historical seeds 2--3 limitations remain
  explicit rather than reconstructed from post-hoc P4 artifacts.
  Implemented as best-effort provenance: `train_td3bc.py` writes per-run
  `PROVENANCE.json`; `launch_mpi_sweep.py` writes `SWEEP_PROVENANCE.json`.
  Multi-GB HDF5 records path+size+mtime rather than a per-run rehash. Contract:
  [`scripts/PROVENANCE.md`](scripts/PROVENANCE.md).

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

- [x] Add the total-budget / first-actor-budget figure, all nine exact budget
  matches, available within-run I4 branch returns, and the separately labeled
  simulator continuation-value summary. Preserve negative task results and
  the partial archive's secondary scope. Keep the JKO interpretation tied to
  ideal fixed-critic operators and simplify repeated qualification in prose.
  Clarify that E-MART's population regression averages sampled first-hop
  targets and need not equal an Euler step from the mean behavior action.

- [x] Organize the main paper and supplement around one regime narrative:
  ideal JKO/gradient-flow behavior at small horizons, the observed
  `T=2.5--7` transition where TD3+BC leaves its high-return plateau, and the
  nonlocal `T>=10` tail where multi-hop shifts rather than removes the failure
  boundary. Treat these as descriptive reading regions, not predeclared bins.
- [x] Add learned activation-region rates only after the complete clean-source
  12-run Hopper/Walker bundle and both snapshot verifier passes. Compact rates are in
  the local-idealization scope paragraph ($0.954/0.911/0.842/0.748$ at
  $T\in\{.025,.05,.1,.2\}$); no nonzero learned `1/K` order slope is restored.
- Decision lock: never restore a nonzero learned `1/K` order slope.
- Decision lock: do not restore the archived actor-path defect values or
  mixed-normalization K8 comparisons, promote the MCEP-inspired control to an
  exact re-centering or equivalence result, or add the inconclusive K4 route
  result merely because those runs exist.
