# Continuation-matched MC (Walker-medium T=20 seed 0, hops 2–3)

First action from the compared MART actor; remaining steps from the
checkpoint Polyak μ1 (`target_actor_params`). This is the Bellman
continuation, not a full-episode rollout of the hop actor.
Eval seeds `1000–1009`, Gymnasium v4, no training.

Checkpoint `/home/ext_csh/p0_bar_p4_two_actor_p4_29fea94_gpu_v2/runs/bar_p4/walker2d-medium-v2_tau20_mpi4_seed0/params_1000000.pkl`
sha256 `b4b797880ead7909ae76ecc56e48ad20eaa9c253ce63c609fce240a2ea7ea945`; discount `0.99`.

## Protocol means (10 episodes)

| Protocol | First | Rest | Score | Mean L | Timeouts | Q(s0,a0) | Disc. G | Disc. G+boot |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_mu1` | μ1 | μ1 | 75.65 | 935 | 9/10 | 274.62 | 163.40 | 163.41 |
| `full_mu2` | μ2 | μ2 | 59.79 | 745 | 6/10 | 275.27 | 119.80 | 119.82 |
| `full_mu3` | μ3 | μ3 | 50.41 | 673 | 5/10 | 275.81 | 132.62 | 132.62 |
| `full_target` | Polyak μ1 | Polyak μ1 | 83.87 | 1000 | 10/10 | 274.55 | 178.13 | 178.15 |
| `first_mu1_then_target` | μ1 | Polyak μ1 | 83.77 | 1000 | 10/10 | 274.62 | 176.83 | 176.85 |
| `first_mu2_then_target` | μ2 | Polyak μ1 | 84.26 | 1000 | 10/10 | 275.27 | 176.83 | 176.85 |
| `first_mu3_then_target` | μ3 | Polyak μ1 | 83.05 | 1000 | 10/10 | 275.81 | 173.06 | 173.08 |

## Hop contrasts (new − ref)

| Hop | ΔJ full | ΔJ hybrid (1-step then Polyak μ1) | ΔQ(s0) | ΔG disc. | ΔG disc.+boot |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 (mart_mu1→mart_mu2) | -15.86 | 0.50 | 0.65 | -0.00 | -0.01 |
| 3 (mart_mu2→mart_mu3) | -9.39 | -1.21 | 0.54 | -3.77 | -3.76 |

Full ΔJ is the online hop actor rolled out for the whole episode.
Hybrid ΔJ uses the training continuation after one step. Q is the
frozen online critic at the episode start. A positive ΔQ with a
negative hybrid ΔG is critic misranking of the hop action under the
Bellman continuation. A near-zero hybrid ΔJ with a large negative
full ΔJ means the first action is not the failure; continuing with
the hop actor is.

## Readout

Hop 2: full ΔJ=−15.86 while hybrid ΔJ=+0.50 and hybrid length is unchanged
(10/10 timeouts). Q(s0) also rises. The first μ2 action under Polyak μ1
is not the failure. Rolling out μ2 for the rest of the episode is.

Hop 3: full ΔJ=−9.39, hybrid ΔJ=−1.21, hybrid still 10/10 timeouts.
Q(s0) rises (+0.54) while discounted hybrid G falls (−3.77). That is a
small start-state misranking under the Bellman continuation, much
smaller than the full-episode drop. Most of the hop-3 harm is still
from continuing with μ3, not from one step plus Polyak μ1.

Polyak μ1 alone scores 83.87 with 10/10 timeouts, above online μ1
(75.65). The training continuation is healthier than the hop actors'
own rollouts.

