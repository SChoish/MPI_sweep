# P0 intermediate-actor diagnostic

Last run: 2026-09-05, host `ext_csh`, CPU, no training.
Interpreter: `capo_jax` (Python 3.10, JAX 0.4.38, Flax 0.10.4), matching the
tail-shard training stack. Eval envs are Gymnasium `*-v4` from
`train_td3bc.EVAL_ENV`. Episode seeds are `1000–1009` for every compared
policy; these scores are **not** mixed with archived first/endpoint columns.

Actor 2 of a K=4 checkpoint is not a K=2 policy. All actors below are read
from the same `params_1000000.pkl` (actor index ≠ training timestep).

## Coverage (executed)

| Quantity | Value |
| --- | --- |
| Planned P0 grid | 180 runs, 90 pairs, 540 policies |
| Local 1M checkpoints | **90 / 180 (50%)** |
| Paired cells (both methods) | **45 / 90 (50%)** |
| Policies evaluated | **270 / 540 (50%)** |
| Episode rows | **2700** (10 per policy) |

Tasks present: `halfcheetah-expert-v2`, `halfcheetah-medium-replay-v2`,
`walker2d-medium-v2`, `walker2d-medium-replay-v2`, `walker2d-expert-v2`.

Tasks missing (ext_csv head shard; **not substituted**):
`hopper-medium-v2`, `hopper-medium-replay-v2`, `hopper-expert-v2`,
`halfcheetah-medium-v2`.

`halfcheetah-medium-replay-v2` is only a 5-cell subset on this host
(T=10 seed 1 and T∈{14,20} both seeds). Hopper-medium directories exist
locally but have **no** 1M checkpoints and are not in the frozen tail
manifest.

Restore/eval pilot: `halfcheetah-medium-replay-v2` T=10 seed 1, all six
policies, 10 episodes each, finite returns.

## What the returns show

Two-seed means of D4RL normalized score on the **local** MART profiles
(n=8–10 cells per T; not the nine-task paper grid):

| T | n | J(μ1) | J(μ2) | J(μ3) | J(μ4) | Δ12 | Δ23 | Δ34 | J(μ4)−J(μ1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 8 | 91.3 | 91.9 | 93.7 | 95.7 | +0.5 | +1.8 | +2.0 | +4.4 |
| 7 | 8 | 93.9 | 92.5 | 88.2 | 84.9 | −1.3 | −4.4 | −3.3 | −9.0 |
| 10 | 9 | 85.8 | 86.7 | 83.2 | 78.4 | +0.9 | −3.5 | −4.8 | −7.4 |
| 14 | 10 | 70.2 | 67.6 | 65.0 | 54.1 | −2.6 | −2.6 | −10.9 | −16.1 |
| 20 | 10 | 44.9 | 35.8 | 34.9 | 26.5 | −9.0 | −1.0 | −8.4 | −18.4 |

At T=4 later hops still help. From T=7 the later hops lose return; at
T∈{14,20} the last hop is the largest drop. High-T deterioration is visible
from hop 2 onward and is **not** confined to the last hop.

Task means of J(μ4)−J(μ1): HC-expert **−44.3**, walker-medium **−10.8**,
walker-medium-replay **+6.0**, HC-medium-replay **+6.4**, walker-expert
**+0.9**. The local average is dominated by HalfCheetah-expert collapse and
is not a nine-task result.

Oracle best actor index (not a selector): μ4 in 17/45, μ1 in 16/45, μ3 in
8/45, μ2 in 4/45. Final actor is worse than some earlier actor in **28/45**
profiles. Example: walker-medium T=14 seed 0, J = (88.3, 93.5, 95.1, **11.2**).

## MART vs two-actor (paired local cells)

Task-equal mean of MART μ4 minus two-actor deploy: **−2.02**.
Per-task: HC-expert −5.31, HC-mr +2.85, W-expert +1.05, W-mr +6.35,
W-medium **−15.06**. Gains and losses cancel.

Signed mean of MART μ1 minus two-actor target is **−0.09**, but mean absolute
cell gap is **6.52**. First actors can match in the average and still differ
cellwise.

Two-actor deploy minus target is **−7.51** on this shard (deployment worse
than the conservative actor in the local mix). That disagrees with the
archived nine-task first/endpoint recovery and must not be spliced with it:
different tasks, different episode seeds, different eval pass.

## Action geometry (same raw states)

Dataset states: 2048 transitions/task, seed 20260905. Rollout set B: 256
states from MART μ4 plus 256 from two-actor deploy; diagnostic only.

Mean RMS action distance (dataset / rollout):

| Contrast | Dataset | Rollout |
| --- | --- | --- |
| MART μ1→μ2 | 0.103 | 0.129 |
| MART μ2→μ3 | 0.086 | 0.131 |
| MART μ3→μ4 | 0.089 | 0.144 |
| MART μ1→μ4 | 0.188 | 0.191 |
| MART μ4 vs two-actor deploy | 0.242 | 0.282 |
| MART μ1 vs two-actor target | 0.126 | 0.181 |
| two-actor target vs deploy | 0.198 | 0.222 |

Later hops do **not** freeze. MART μ4 vs two-actor deploy is larger than
adjacent MART hops. Correlation of that RMS with the return gap is **0.01**;
cells above median action distance have mean |return gap| 14.9 vs 4.9 below.
Actions can differ while returns stay close, and large action gaps also
appear with large return gaps — the scatter is not a single mechanism.

Hop-direction cosine on dataset states (near-zero hops excluded):
δ2·δ3 **0.17**, δ3·δ4 **0.04**, rising with T (0.01 at T=4 to 0.31 at T=20).
Local alignment at large T is weak and does not support global equivalence.

Action-bound occupancy (median fraction of coords with |a|≥0.95):
μ1 0.18 → μ4 0.24, two-actor deploy 0.26.

## Q coefficients (current objective, median)

Arithmetic-mean Q is unusable: a few Walker2d-expert critic-magnitude tails
reach ~1e12. **Median** dataset Q(s,π(s)): μ1 368.5, μ2 369.8, μ3 370.5,
μ4 370.9. Learned Q still ticks up while high-T return falls. That is not
critic accuracy and not policy improvement.

Walker-medium T=14 seed 0 (return collapse on hop 4): Q 327.8→328.4→328.9→329.3
while J 88.3→93.5→95.1→11.2. Same-cell Q increase with return collapse is
the priority input to a later MC critic-error diagnostic.

λ uses the current formula `2·(T/K) / mean(|Q(s,ref)|)` with ref = previous
actor for hops k>1 and current π for dataset-anchored actors. Training
revision is `29fea94`; this dump labels the formula as current-code, not a
reconstructed historical tensor.

## Hypotheses after this dump

- **H1 saturation:** not in return except T=4. Action RMS stays ~0.09 per
  hop. Later actors are not copies of μ1.
- **H2 accumulation/reversion:** supported on this shard for T≥7, and as an
  oracle last-hop collapse in 28/45 profiles. Not an offline selector.
- **H3 direction alignment:** weak (cosine 0.04–0.17). Do not promote to
  large-T equivalence.
- **H4 similar return, different action:** possible. μ4 vs deploy RMS 0.24
  with task-equal return gap −2 and near-zero correlation.
- **H5 task cancellation:** supported locally (W-mr +6.4 vs W-medium −15.1).
  Missing Hopper/HC-medium could change the nine-task sign.
- **H6 extraction-independent path difference:** first-actor returns match
  on average (−0.09) but not cellwise (|gap| 6.52); first-actor actions
  already differ (RMS 0.13). This dump cannot isolate re-centering.

## Next experiment (do not auto-launch training)

**Critic-error MC on Q-up / return-down cells**, starting with
walker-medium T=14 seed 0 and HalfCheetah-expert T∈{14,20}. First action from
the compared actor, continuation from that run's Bellman target policy
(Polyak μ1). Separate from full-episode returns in this dump.

The first **training** experiment remains frozen P0.B shared-driver, which
is required before claiming that sequential re-centering is or is not the
mechanism. Do not start it until an environment-locked `FROZEN_MANIFEST.json`
exists.

## Reproduce

```bash
cd /home/ext_csh/MPI_sweep
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
"$PY" -u scripts/diagnostics/run_p0_intermediate_actors.py all --workers 4
"$PY" -u scripts/diagnostics/analyze_p0_intermediate_actors.py
```

Pilot only: add `--pilot` to `eval`. Resume is per-policy JSONL under
`episodes/`. Head90 cells remain missing until that host's checkpoints are
inventoried with the same protocol; do not fill them from `results/mpi4*`.
