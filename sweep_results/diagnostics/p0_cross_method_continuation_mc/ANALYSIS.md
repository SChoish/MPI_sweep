# Cross-method continuation MC (Walker-medium T=20 seed 0)

Tail90 only. Same eval seeds `1000–1009`. First action and continuation
may come from different checkpoints; each actor uses its own mean/std.
Q(s0,a0) is the continuation critic. Start-state only: later visited
states are not restored here. Hybrid vs full rollout is not a percent
decomposition of return, because later actions and visited states both
change.

MART `/home/ext_csh/p0_bar_p4_two_actor_p4_29fea94_gpu_v2/runs/bar_p4/walker2d-medium-v2_tau20_mpi4_seed0/params_1000000.pkl`
sha256 `b4b797880ead7909ae76ecc56e48ad20eaa9c253ce63c609fce240a2ea7ea945`
two-actor `/home/ext_csh/p0_bar_p4_two_actor_p4_29fea94_gpu_v2/runs/two_actor_p4/walker2d-medium-v2_tau20_mcep4_seed0/params_1000000.pkl`
sha256 `c4da38a9b2f8825acaee1e63a01a56bd681359b73b611527b7f426857cad09f2`; discount `0.99`.

## Full rollouts

| Protocol | Score | Mean L | Timeouts | Q(s0,a0) | Disc. G |
| --- | ---: | ---: | ---: | ---: | ---: |
| MART μ2 | 59.79 | 745 | 6/10 | 275.27 | 119.80 |
| MART μ3 | 50.41 | 673 | 5/10 | 275.81 | 132.62 |
| two-actor deployment | 73.24 | 778 | 7/10 | 225.88 | 274.46 |
| MART Polyak μ1 | 83.87 | 1000 | 10/10 | 274.55 | 178.13 |
| two-actor Polyak target | 91.31 | 1000 | 10/10 | 225.11 | 266.71 |

## 2×2 hybrid (first action × continuation)

### Hop 2: MART μ2 vs two-actor deployment

| First \\ rest | MART Polyak μ1 | two-actor Polyak target |
| --- | ---: | ---: |
| MART hop | J=84.26, L=1000 (10/10), G=176.83 | J=90.76, L=1000 (10/10), G=260.26 |
| two-actor deploy | J=84.30, L=970 (9/10), G=207.89 | J=91.65, L=1000 (10/10), G=272.99 |

### Hop 3: MART μ3 vs two-actor deployment

| First \\ rest | MART Polyak μ1 | two-actor Polyak target |
| --- | ---: | ---: |
| MART hop | J=83.05, L=1000 (10/10), G=173.06 | J=90.39, L=1000 (10/10), G=257.32 |
| two-actor deploy | J=84.30, L=970 (9/10), G=207.89 | J=91.65, L=1000 (10/10), G=272.99 |

## Paired contrasts (new − ref, 10 episodes)

Q deltas are reported only when both arms use the same continuation
critic. Continuation swaps change the critic, so those rows leave ΔQ blank.

| Hop | Contrast | ΔJ mean (se, mean/se) | ΔG disc. mean (se, mean/se) | ΔQ(s0) |
| ---: | --- | --- | --- | --- |
| 3 | MART μ3−μ2 | MART Polyak μ1 | -1.21 (se 0.46, -2.64; 2+/8-) | -3.77 (se 1.52, -2.48; 3+/7-) | 0.54 (se 0.02, 29.23; 10+/0-) |
| 3 | MART μ3−μ2 | two-actor Polyak target | -0.37 (se 0.48, -0.76; 3+/7-) | -2.93 (se 0.89, -3.29; 0+/10-) | -0.21 (se 0.02, -8.52; 0+/10-) |
| 2 | MART μ2 − two-actor deploy | MART Polyak μ1 | -0.04 (se 2.59, -0.01; 3+/7-) | -31.06 (se 4.54, -6.84; 0+/10-) | 3.17 (se 0.07, 48.67; 10+/0-) |
| 2 | MART μ2 − two-actor deploy | two-actor Polyak target | -0.90 (se 0.58, -1.54; 3+/7-) | -12.73 (se 2.50, -5.09; 1+/9-) | -2.60 (se 0.10, -27.15; 0+/10-) |
| 3 | MART μ3 − two-actor deploy | MART Polyak μ1 | -1.25 (se 2.68, -0.47; 1+/9-) | -34.82 (se 5.01, -6.95; 0+/10-) | 3.71 (se 0.06, 65.09; 10+/0-) |
| 3 | MART μ3 − two-actor deploy | two-actor Polyak target | -1.26 (se 0.50, -2.53; 1+/9-) | -15.66 (se 2.30, -6.82; 0+/10-) | -2.81 (se 0.08, -33.86; 0+/10-) |
| 2 | MART Polyak μ1 − two-actor Polyak | first MART μ2 | -6.49 (se 0.79, -8.26; 0+/10-) | -83.43 (se 2.25, -37.10; 0+/10-) | n/a |
| 3 | MART Polyak μ1 − two-actor Polyak | first MART μ3 | -7.34 (se 0.57, -12.93; 0+/10-) | -84.26 (se 3.36, -25.07; 0+/10-) | n/a |
| 0 | MART Polyak μ1 − two-actor Polyak | first two-actor deploy | -7.35 (se 2.43, -3.03; 0+/10-) | -65.10 (se 4.98, -13.07; 0+/10-) | n/a |
| 2 | full MART μ2 − full two-actor deploy | -13.45 (se 21.09, -0.64; 3+/7-) | -154.65 (se 20.25, -7.64; 0+/10-) | n/a |
| 3 | full MART μ3 − full two-actor deploy | -22.84 (se 20.73, -1.10; 3+/7-) | -141.84 (se 20.48, -6.93; 0+/10-) | n/a |

## Hop 3 start-state ΔQ vs discounted ΔG

Same discount 0.99. MART μ3 vs μ2, both followed by MART Polyak μ1.
ΔQ(s0) mean 0.541 (se 0.018, mean/se 29.23, 10/10 positive).
Discounted ΔG mean -3.766 (se 1.517, mean/se -2.48, 3+/7-).
A consistent Q rise with a smaller, noisier G drop is start-state
misranking under this continuation. It is not a claim about later states.

## Readout constraints

- First-action findings apply to these start states only.
- Hybrid minus full is not a share of return explained by continuation.
- Extra hop-actor optimization is not measured here.
- This cell is tail90 Walker; do not pool with first90 Hopper scores.

Start-state action swaps under a conservative target continuation did
not reproduce the early terminations seen when rolling out the later
actor for the whole episode. First-action value at s0 does not explain
deployment return of the later actor.

