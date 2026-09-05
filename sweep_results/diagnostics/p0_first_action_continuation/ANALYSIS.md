# First-action × Polyak continuation (Hopper T=10)

Built: 2026-09-06T01:09:43+09:00

Understood as: same Gymnasium reset seeds as the hop-actor audit
(10000–10009), first action from
MART μ2/μ3 or two-actor deployment, then continuation from MART Polyak μ1
or two-actor Polyak target. CPU only. First90 checkpoints on ext_csv.
Walker / HalfCheetah-expert tail90 checkpoints are not on this host.

## Checkpoints

- Paired cells run: 6 (3 Hopper tasks × seeds {0,1}).
- MART and two-actor 1M files exist locally; `target_actor_params` present.

## Protocol

- Continuation is the Polyak target actor, not the online first actor.
- Each policy uses its own checkpoint mean/std.
- Discount for ΔG is 0.99, matching `train_td3bc --discount`.
- Two-actor first-action rows are shared across hop-2 and hop-3 tables.
- Full-episode μk / deployment numbers come from `p0_hop_actor_audit`.

## hopper-medium

### Hop 2

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ2 | 48.04 | 51.64 |
| two-actor deploy | 50.79 | 55.30 |

- First-action swap μ2 vs Polyak μ1, same MART continuation: mean Δ score -4.30 (std 10.33, 11/20 negative).
- Hybrid mean length change -41.5; hybrid timeouts 0/20.

- Full-episode MART μ2 (existing hop audit): scores 63.80/70.07, lengths 626/680.

### Hop 3

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ3 | 49.68 | 49.95 |
| two-actor deploy | 50.79 | 55.30 |

- First-action swap μ3 vs Polyak μ1, same MART continuation: mean Δ score -2.65 (std 12.03, 8/20 negative).
- Hybrid mean length change -26.4; hybrid timeouts 0/20.

- Full-episode MART μ3 (existing hop audit): scores 89.51/98.54, lengths 879/979.

Hop-3 ranking on the same discount 0.99:
- mean ΔQ(s0, μ3−Polyak μ1) +0.298.
- mean discounted ΔG -0.002 (std 1.124).
- sign agree 13/20; Q↑ G↓ 7; Q↓ G↑ 0.

First-action contrast (MART hop minus two-actor deploy, same continuation) and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first action) are in `contrasts.csv`.

## hopper-medium-replay

### Hop 2

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ2 | 85.50 | 89.16 |
| two-actor deploy | 83.85 | 96.55 |

- First-action swap μ2 vs Polyak μ1, same MART continuation: mean Δ score +0.06 (std 25.22, 11/20 negative).
- Hybrid mean length change -8.2; hybrid timeouts 7/20.

- Full-episode MART μ2 (existing hop audit): scores 99.43/100.36, lengths 1000/1000.

### Hop 3

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ3 | 84.39 | 92.32 |
| two-actor deploy | 83.85 | 96.55 |

- First-action swap μ3 vs Polyak μ1, same MART continuation: mean Δ score -1.05 (std 31.60, 10/20 negative).
- Hybrid mean length change -9.9; hybrid timeouts 11/20.

- Full-episode MART μ3 (existing hop audit): scores 100.36/98.94, lengths 1000/1000.

Hop-3 ranking on the same discount 0.99:
- mean ΔQ(s0, μ3−Polyak μ1) +1.436.
- mean discounted ΔG -0.555 (std 1.467).
- sign agree 8/20; Q↑ G↓ 12; Q↓ G↑ 0.

First-action contrast (MART hop minus two-actor deploy, same continuation) and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first action) are in `contrasts.csv`.

## hopper-expert

### Hop 2

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ2 | 106.29 | 105.24 |
| two-actor deploy | 105.76 | 108.51 |

- First-action swap μ2 vs Polyak μ1, same MART continuation: mean Δ score +5.22 (std 20.74, 8/20 negative).
- Hybrid mean length change +50.2; hybrid timeouts 14/20.

- Full-episode MART μ2 (existing hop audit): scores 103.45/95.36, lengths 910/839.

### Hop 3

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ3 | 104.64 | 107.08 |
| two-actor deploy | 105.76 | 108.51 |

- First-action swap μ3 vs Polyak μ1, same MART continuation: mean Δ score +3.57 (std 22.94, 10/20 negative).
- Hybrid mean length change +26.1; hybrid timeouts 9/20.

- Full-episode MART μ3 (existing hop audit): scores 90.46/46.09, lengths 782/415.

Hop-3 ranking on the same discount 0.99:
- mean ΔQ(s0, μ3−Polyak μ1) +0.216.
- mean discounted ΔG +0.130 (std 0.981).
- sign agree 7/20; Q↑ G↓ 13; Q↓ G↑ 0.

First-action contrast (MART hop minus two-actor deploy, same continuation) and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first action) are in `contrasts.csv`.

## Read

Hopper-expert hop 3 is the clear continuation case on this host. Full-episode
μ3 scores 90.5/46.1 with lengths 782/415. The same μ3 used only as the first
action, then MART Polyak μ1, scores 104.6 with 9/20 timeouts and does not
reproduce that collapse. The other three 2×2 cells are 105.8–108.5. First
action μ3 vs two-actor deploy, holding MART continuation fixed, is −1.1 points.
So the MART μ3 vs two-actor deployment gap is not a first-action gap on these
initial states.

Hopper-medium goes the other way: full μ2/μ3 beat μ1, but swapping only the
first action onto Polyak μ1 does not recover that gain (Δ score −4.3 / −2.7,
0/20 hybrid timeouts). First-action value and full-actor value differ in both
directions.

Hop-3 ΔQ vs discounted ΔG (γ=0.99) on MART continuation: mean ΔG is small
relative to the paired episode spread (expert +0.13, std 0.98). Q at s0 is
usually positive for μ3 while discounted ΔG is often negative (expert 13/20
Q↑G↓). That is a ranking disagreement at the initial state, not a claim that
the mean −3.77-style gap is itself significant.

The 2×2 does not turn hybrid vs full-episode into a percentage split of the
return drop. Later actions and visited states change together. Walker-medium
1M checkpoints are not on ext_csv.

