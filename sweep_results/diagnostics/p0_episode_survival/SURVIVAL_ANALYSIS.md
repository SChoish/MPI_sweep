# Episode survival and hop objectives

Return is split into how long the episode lasts and how much reward
arrives per step. Survival curves use stored `episode_length` only.
Hopper T=10 uses the ext_csv first90 episode CSV (commit `0c8c50cd`,
10 episodes per training seed). Seeds are not extra samples; pooling
two seeds gives the 20-episode table below. Tail90 hop objectives use
local frozen critics; Hopper checkpoints are not on this host.

## Hopper T=10 pooled two-seed table (20 episodes = 2×10)

| Task | Actor | Score | Mean L | Timeouts | Reward / step |
| --- | --- | ---: | ---: | ---: | ---: |
| hopper-medium-v2 | MART μ2 | 66.94 | 653 | 1/20 | 3.305 |
| hopper-medium-v2 | MART μ4 | 100.73 | 1000 | 20/20 | 3.258 |
| hopper-medium-v2 | two-actor deploy | 98.35 | 1000 | 20/20 | 3.181 |
| hopper-medium-replay-v2 | MART μ2 | 99.90 | 1000 | 20/20 | 3.231 |
| hopper-medium-replay-v2 | MART μ4 | 100.42 | 1000 | 20/20 | 3.248 |
| hopper-medium-replay-v2 | two-actor deploy | 99.76 | 1000 | 20/20 | 3.226 |
| hopper-expert-v2 | MART μ2 | 99.40 | 875 | 12/20 | 3.675 |
| hopper-expert-v2 | MART μ4 | 51.52 | 456 | 0/20 | 3.629 |
| hopper-expert-v2 | two-actor deploy | 100.17 | 879 | 11/20 | 3.688 |

Hopper-medium μ2→μ4: reward/step 3.305→3.258 but L 653→1000.
Hopper-expert μ2→μ4: reward/step 3.675→3.629 but L 875→457.
The score change is an episode-survival change, not a per-step bonus.
Reaching 1000 steps is not a claim that return is at its task maximum.

Per-seed MART μ4−μ2 at T=10: Hopper-medium +36.95 / +30.64;
Hopper-expert −31.29 / −64.48. Not a one-seed artifact.

The survival curves (seeds not pooled) show where the lifetime change
happens, and that it is cohort-wide rather than a few failed episodes:

- Hopper-medium: μ3 converts most episodes to timeouts (seed 0: 1/10→6/10;
  seed 1: 0/10→9/10); μ4 finishes 10/10. Two-actor deploy matches μ4.
- Hopper-medium-replay: μ1 still fails early; μ2 already 10/10 at 1000.
  Later hops cannot add lifetime.
- Hopper-expert seed 0: timeouts 7/10→7/10→4/10→0/10; μ4 lengths 472–815.
  Seed 1: μ3 already 9/10 below 400 steps; μ4 mean L=294. Two-actor
  deploy stays with μ1/μ2.

## Hop objective ΔL on local tail90 (dataset states)

MART hops k=2,3,4 vs predecessor, local 5-task tail90 shard, dataset
states, frozen final critic. 135 hops; 9 hops
carry a nonfinite critic optimizer and are kept but not averaged.

Q(μ_k)−Q(μ_{k-1}) is positive on 134/135 hops, but the hop
loss ΔL is negative on only 25/135. Transport usually
exceeds the scaled Q term, especially after hop 2:

- hop 2: ΔL<0 on 17/45
- hop 3: ΔL<0 on 5/45
- hop 4: ΔL<0 on 3/45

Labels (loss decrease = objective improvement; |ΔJ|>1 = return change):

- `other_obj_not_up_return_up`: 55
- `other_obj_not_up_return_flat`: 28
- `objective_not_up_return_down`: 27
- `objective_up_return_down`: 12
- `objective_up_return_flat`: 8
- `objective_up_return_up`: 5

Walker-medium T=14 seed 0 is the local survival-collapse analog of
Hopper-expert. Hop 2 is `objective_up_return_up`. Hop 4 is
`objective_not_up_return_down`: ΔL=0.002344, mean ΔQ=0.384,
ΔJ=-83.9. Episode length 1000→195 (10/10→1/10 timeouts). Q still rose; the hop loss did not improve because
transport exceeded c_k ΔQ. This is objective-not-up / return-down,
not automatic critic error.

HalfCheetah-expert is a different failure mode. At T=20 seed 0,
μ1 and μ4 both timeout 10/10 and 10/10; reward/step 5.735→0.326. Return drops
without early termination. Do not treat every harmful hop as a
Hopper-style survival failure.

Hopper-expert ΔL is **not** computed here (checkpoints stay on ext_csv).
Its return drop is already a survival drop. Seed 0 μ4 lengths are all
in 472–815 (0/10 timeouts); seed 1 μ3 already puts 9/10 episodes
below 400 steps. Q-up at a frozen first-actor critic would still not
by itself prove critic error, because backups follow Polyak μ1
continuation rather than μ4 rollouts.

## Next

Shared-driver is deprioritized. The manuscript question is how extra hops
change episode survival, not whether two methods match under a shared critic.
First no-train follow-up remains hop-objective coverage on Hopper first90
checkpoints when that host is available. MC continuation only where ΔL
improved and lifetime collapsed (local: walker-medium T=20 seed 0 hops
2–3). Walker-medium T=14 seed 0 hop 4 is the wrong cell for that test:
its hop loss did not improve.

