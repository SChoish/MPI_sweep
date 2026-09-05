# P0 hop-actor audit

Built: 2026-09-06T00:21:39+09:00

Understood as: explain similar final scores and later-hop gains/losses
by splitting return into episode survival vs per-step reward, then
(next) hop-objective change on the frozen critic. No new training.
Episode repeats are noise reduction, not extra seeds.

## Coverage

- Expected runs: 180
- Evalable on this host: 90 (50%)
- Paired task/T/seed cells: 45 (50%)
- Missing tasks here: halfcheetah-expert-v2, walker2d-medium-v2, walker2d-medium-replay-v2, walker2d-expert-v2
- Tail90 Walker / halfcheetah-expert checkpoints stay on ext_csh and were not substituted.

## Executed measurements

- Deterministic gymnasium MuJoCo-v4 eval, 10 episodes, shared `evaluation_seed = 10000+ep`.
- Episode rows: 2700
- Paired return contrasts: 45
- Action distances on dataset states and, where both rollout dumps exist, the mu4∪deploy union.

## Return profiles

- Adjacent-hop mean ΔJ by T/hop: `{'T4_hop2': 14.692520292666543, 'T4_hop3': 0.706648308560419, 'T4_hop4': 1.4125123866694822, 'T7_hop2': 9.589596688697407, 'T7_hop3': 1.8719955960463728, 'T7_hop4': 4.720755412558216, 'T10_hop2': 14.439574276011781, 'T10_hop3': -0.5032346824947054, 'T10_hop4': -1.5770097765532556, 'T14_hop2': 20.31694202357846, 'T14_hop3': -5.746008588812503, 'T14_hop4': -2.17093390401533, 'T20_hop2': 14.870760125206692, 'T20_hop3': 2.598466194966325, 'T20_hop4': -6.538330715184507}`
- Oracle mid-actor better than mu4 in 18 cells.
- Do not read that oracle as an offline selection policy.
- Actor index is not a separately trained K=2/K=3 run; timestep is 1M for every stored actor.

## Hopper T=10 survival vs per-step reward

### hopper-medium

- μ2: score 66.94, length 653.0, timeout 1/20, r/step 3.305.
- μ4: score 100.73, length 1000.0, timeout 20/20, r/step 3.258.
- two-actor final: score 98.35, length 1000.0, timeout 20/20, r/step 3.181.
- μ4−μ2 score +33.80 (seeds +36.95, +30.64). Raw-return split: length term +1147.1, rate term -47.2 (length share 0.96).
- Reaching 1,000 steps is a timeout, not a return ceiling.

### hopper-med-replay

- μ2: score 99.90, length 1000.0, timeout 20/20, r/step 3.231.
- μ4: score 100.42, length 1000.0, timeout 20/20, r/step 3.248.
- two-actor final: score 99.76, length 1000.0, timeout 20/20, r/step 3.226.
- μ4−μ2 score +0.53 (seeds +1.32, -0.27). Raw-return split: length term +0.0, rate term +17.2 (length share 0.00).
- Reaching 1,000 steps is a timeout, not a return ceiling.

### hopper-expert

- μ2: score 99.40, length 874.9, timeout 12/20, r/step 3.675.
- μ4: score 51.52, length 456.5, timeout 0/20, r/step 3.629.
- two-actor final: score 100.17, length 878.6, timeout 11/20, r/step 3.688.
- μ4−μ2 score -47.89 (seeds -31.29, -64.48). Raw-return split: length term -1537.4, rate term -21.1 (length share 0.99).
- Reaching 1,000 steps is a timeout, not a return ceiling.

## Hopper T=10 hop objective (frozen critic, dataset states)

ΔL_k = -c_k E[Q(μ_k)-Q(μ_{k-1})] + E[||Δμ||^2 / d], c_k = 2(T/K) / mean(|Q(μ_{k-1})|), Q = critic head 1. Negative ΔL is an objective improvement. Residuals are ~10^{-3} on a loss of about -5.

### hopper-medium

- mean ΔL=+0.001272, mean ΔQ=+0.1608, objective-improved hops 0/6.
- class counts: `{'obj_flat_return_flat': 1, 'obj_flat_return_up': 5}`.

### hopper-med-replay

- mean ΔL=-0.0005412, mean ΔQ=+0.8748, objective-improved hops 3/6.
- class counts: `{'obj_flat_return_flat': 2, 'obj_flat_return_up': 1, 'obj_up_return_flat': 2, 'obj_up_return_up': 1}`.

### hopper-expert

- mean ΔL=+0.003409, mean ΔQ=+0.02711, objective-improved hops 0/6.
- class counts: `{'obj_flat_return_down': 5, 'obj_flat_return_up': 1}`.

Hopper-expert later hops sit in **obj not improved + return down**. Hopper-medium is the missing fifth cell: **obj not improved + return up**. ΔQ is positive in every hop, including expert. That is not treated as policy improvement: this critic's Bellman target is actor-1 continuation.

## Hypotheses (evidence, not verdicts)

### H1_saturation

later |ΔJ| and adjacent action RMS compared with hop 1→2; mean |ΔJ| hop2=16.839, hop3=6.407, hop4=5.996; mean adj RMS mu1→2=0.0960, mu2→3=0.0827, mu3→4=0.0787.

### H2_reversion

18 cells have mu4 below the best of mu1–mu3 (oracle diagnostic, not offline selection).

### H3_direction

mean cosine(δ2,δ3) on dataset states=0.159 after dropping near-zero hops.

### H4_similar_return_different_action

34/45 paired cells have dataset action RMS>0.05 and |ΔJ|<3.

### H5_task_offset

task-mean J(mu4)-J(deploy)={"halfcheetah-medium-replay-v2": 1.4439261052317947, "halfcheetah-medium-v2": -0.0028703428964746537, "hopper-expert-v2": -13.451350060424266, "hopper-medium-replay-v2": 0.7129885830427511, "hopper-medium-v2": 3.823958116761203}; grand mean=-1.821180144644642; sign mix=[1, -1, -1, 1, 1].

### H6_first_actor_path

Not isolated here: MART mu1 and two-actor target were trained on independent first-actor/critic trajectories. Shared-driver remains available but is deprioritized relative to the actor-path survival and hop-objective readout.

### H7_survival_not_rate

At Hopper T=10, mu2→mu4 per-step reward falls slightly on medium and expert while episode length moves in opposite directions. Return change is therefore mostly survival, not richer per-step reward.

### H8_hop_objective

Frozen-critic JKO ΔL at the 1M snapshot is tiny (~1e-3 on |L|≈5). Q(s,μ_k) still ticks up. Medium return rises and expert return falls anyway, so this critic's hop objective does not separate helpful from harmful hops. The critic is trained on actor-1 continuation.

## Figures

- `fig_actor_index_profiles.png`
- `fig_hop_return_heatmap.png`
- `fig_action_vs_return_scatter.png`
- `fig_survival_hopper_T10.png`
- `fig_return_decomposition_hopper_T10.png`
- `fig_hop_objective_vs_return.png`

## Limits

- Two training seeds only.
- Cross-stack P0 merge remains scientifically inadmissible; this CPU re-eval is a same-host diagnostic.
- Learned Q increase is not treated as policy improvement.

## Next measurement

Shared-driver is deprioritized. Next is only to extend hop-objective
coverage beyond Hopper T=10 if that contrast is needed, not to launch
a new training grid.

