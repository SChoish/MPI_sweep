# P0 hop-actor audit

Built: 2026-09-05T23:38:09+09:00

Understood as: explain similar MART vs two-actor *final* scores by measuring
each hop's return and action change on existing P0 K=4 checkpoints.
No new training. Episode repeats are noise reduction, not extra seeds.

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

Not isolated here: MART mu1 and two-actor target were trained on independent first-actor/critic trajectories. Shared-driver P0.B remains the experiment that holds that path fixed.

## Figures

- `fig_actor_index_profiles.png`
- `fig_hop_return_heatmap.png`
- `fig_action_vs_return_scatter.png`

## Limits

- Two training seeds only.
- Cross-stack P0 merge remains scientifically inadmissible; this CPU re-eval is a same-host diagnostic.
- Learned Q increase is not treated as policy improvement.

## Next experiment

P0.B shared-driver (`recenter` minus `data_anchor`) remains the first training experiment,
because H6 is unresolved and hop diagnostics cannot hold the first actor/critic path fixed.

