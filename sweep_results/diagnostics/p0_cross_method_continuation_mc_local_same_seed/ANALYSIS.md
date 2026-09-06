# Cross-method continuation MC (Walker-medium T=20 seed 0)

Same eval seeds `1000–1009`. First action and continuation
may come from different checkpoints; each actor uses its own mean/std.
Q(s0,a0) is the continuation critic. Start-state only: later visited
states are not restored here. Hybrid vs full rollout is not a percent
decomposition of return, because later actions and visited states both
change.

MART `/home/ext_csv/mpi_sweep_lab/results/mpi4_norm/walker2d-medium-v2_tau20_mpi4_seed0/params_1000000.pkl`
sha256 `6afcce90cb8bbcf963876013da788714096f4e5c896d36e68fee793e3c19ff48`
two-actor `/home/ext_csv/MPI_sweep/results/mcep_p3/walker2d-medium-v2_tau20_mcep3_seed0/params_1000000.pkl`
sha256 `15a9b3e6298b836170f2bce03d00c8be76c21fcfeff07972d2d389e1562ee503`; discount `0.99`.

## Full rollouts

| Protocol | Score | Mean L | Timeouts | Q(s0,a0) | Disc. G |
| --- | ---: | ---: | ---: | ---: | ---: |
| MART μ2 | 1.83 | 199 | 0/10 | 1521946394624.00 | 60.05 |
| MART μ3 | 1.83 | 199 | 0/10 | 1521946394624.00 | 60.05 |
| MART μ4 | 1.83 | 199 | 0/10 | 1521946394624.00 | 60.05 |
| two-actor deployment | 1.83 | 199 | 0/10 | 1300524354764.80 | 60.05 |
| MART Polyak μ1 | 1.83 | 199 | 0/10 | 1521946394624.00 | 60.05 |
| two-actor Polyak target | 1.83 | 199 | 0/10 | 1300524354764.80 | 60.05 |

## 2×2 hybrid (first action × continuation)

### Hop 2: MART μ2 vs two-actor deployment

| First \\ rest | MART Polyak μ1 | two-actor Polyak target |
| --- | ---: | ---: |
| MART hop | J=1.83, L=199 (0/10), G=60.05 | J=1.83, L=199 (0/10), G=60.05 |
| two-actor deploy | J=1.83, L=199 (0/10), G=60.05 | J=1.83, L=199 (0/10), G=60.05 |

### Hop 3: MART μ3 vs two-actor deployment

| First \\ rest | MART Polyak μ1 | two-actor Polyak target |
| --- | ---: | ---: |
| MART hop | J=1.83, L=199 (0/10), G=60.05 | J=1.83, L=199 (0/10), G=60.05 |
| two-actor deploy | J=1.83, L=199 (0/10), G=60.05 | J=1.83, L=199 (0/10), G=60.05 |

### Hop 4: MART μ4 vs two-actor deployment

| First \\ rest | MART Polyak μ1 | two-actor Polyak target |
| --- | ---: | ---: |
| MART hop | J=1.83, L=199 (0/10), G=60.05 | J=1.83, L=199 (0/10), G=60.05 |
| two-actor deploy | J=1.83, L=199 (0/10), G=60.05 | J=1.83, L=199 (0/10), G=60.05 |

## Paired contrasts (new − ref, 10 episodes)

Q deltas are reported only when both arms use the same continuation
critic. Continuation swaps change the critic, so those rows leave ΔQ blank.

| Hop | Contrast | ΔJ mean (se, mean/se) | ΔG disc. mean (se, mean/se) | ΔQ(s0) |
| ---: | --- | --- | --- | --- |
| 3 | MART μ3−μ2 | MART Polyak μ1 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 3 | MART μ3−μ2 | two-actor Polyak target | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 2 | MART μ2 − two-actor deploy | MART Polyak μ1 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 2 | MART μ2 − two-actor deploy | two-actor Polyak target | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 3 | MART μ3 − two-actor deploy | MART Polyak μ1 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 3 | MART μ3 − two-actor deploy | two-actor Polyak target | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 4 | MART μ4−μ3 | MART Polyak μ1 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 4 | MART μ4 − two-actor deploy | MART Polyak μ1 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 4 | MART μ4 − two-actor deploy | two-actor Polyak target | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) |
| 4 | full MART μ4 − full two-actor deploy | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | n/a |
| 2 | MART Polyak μ1 − two-actor Polyak | first MART μ2 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | n/a |
| 3 | MART Polyak μ1 − two-actor Polyak | first MART μ3 | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | n/a |
| 0 | MART Polyak μ1 − two-actor Polyak | first two-actor deploy | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | n/a |
| 2 | full MART μ2 − full two-actor deploy | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | n/a |
| 3 | full MART μ3 − full two-actor deploy | 0.00 (se 0.00, nan; 0+/0-) | 0.00 (se 0.00, nan; 0+/0-) | n/a |

## Hop 3 start-state ΔQ vs discounted ΔG

Same discount 0.99. MART μ3 vs μ2, both followed by MART Polyak μ1.
ΔQ(s0) mean 0.000 (se 0.000, mean/se nan, 0/10 positive).
Discounted ΔG mean 0.000 (se 0.000, mean/se nan, 0+/0-).
A consistent Q rise with a smaller, noisier G drop is start-state
misranking under this continuation. It is not a claim about later states.

## Readout constraints

- First-action findings apply to these start states only.
- Hybrid minus full is not a share of return explained by continuation.
- Extra hop-actor optimization is not measured here.
- Shard `local_mpi4_norm_vs_mcep3_same_seed`. Do not pool with a different MART/two-actor tree.

Start-state action swaps under a conservative target continuation did
not reproduce the early terminations seen when rolling out the later
actor for the whole episode. First-action value at s0 does not explain
deployment return of the later actor.

