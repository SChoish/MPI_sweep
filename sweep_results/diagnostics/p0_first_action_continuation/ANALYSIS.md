# First-action × Polyak continuation (Hopper T=10)

Built: 2026-09-06T11:16:36+09:00

Understood as: same Gymnasium reset seeds as the hop-actor audit
(10000–10009), first action from
MART μ2/μ3/μ4 or two-actor deployment, then continuation from MART Polyak μ1
or two-actor Polyak target. CPU, deterministic Polyak (no TD3 target noise).
First90 checkpoints on ext_csv. Walker tail90 is not on this host.

## Checkpoints

- Paired cells run: 6 (3 Hopper tasks × seeds {0,1}).
- MART and two-actor 1M files exist locally; `target_actor_params` present.

## Protocol

- Continuation is the Polyak target actor, not the online first actor.
- Each policy uses its own checkpoint mean/std.
- Discount for ΔG is 0.99, matching `train_td3bc --discount`.
- Two-actor first-action rows are shared across hop-2/3/4 tables.
- Adjacent-hop ranking is μ_k−μ_{k-1} under the same MART Polyak continuation.
- Timeout counts are per 2×2 cell; they are not shared across the table.
- Full-episode μk / deployment numbers come from `p0_hop_actor_audit`.

## hopper-medium

### Hop 2

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ2 | 48.04 (0/20 to) | 51.64 (0/20 to) |
| two-actor deploy | 50.79 (0/20 to) | 55.30 (0/20 to) |

- First action μ2 minus two-actor deploy, MART Polyak continuation: ΔJ -2.75 (std 8.66); discounted ΔG -0.08 (std 0.78). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ2 minus two-actor deploy, two-actor Polyak continuation: ΔJ -3.66 (std 12.85); discounted ΔG -0.03 (std 1.17). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ2 vs Polyak μ1, same MART continuation: mean Δ score -4.30 (std 10.33, 11/20 negative).
- Hybrid mean length change -41.5; hybrid timeouts 0/20.

- Full-episode MART μ2 (existing hop audit): scores 63.80/70.07, lengths 626/680.

### Hop 3

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ3 | 49.68 (0/20 to) | 49.95 (0/20 to) |
| two-actor deploy | 50.79 (0/20 to) | 55.30 (0/20 to) |

- First action μ3 minus two-actor deploy, MART Polyak continuation: ΔJ -1.11 (std 5.15); discounted ΔG +0.12 (std 0.43). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ3 minus two-actor deploy, two-actor Polyak continuation: ΔJ -5.34 (std 9.85); discounted ΔG -0.28 (std 0.59). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ3 vs Polyak μ1, same MART continuation: mean Δ score -2.65 (std 12.03, 8/20 negative).
- Hybrid mean length change -26.4; hybrid timeouts 0/20.

- Full-episode MART μ3 (existing hop audit): scores 89.51/98.54, lengths 879/979.

Adjacent μ3−μ2 under MART Polyak continuation (γ=0.99):
- mean ΔQ(s0) +0.153.
- mean discounted ΔG +0.200 (std 0.807).
- sign agree 13/20; Q↑ G↓ 7; Q↓ G↑ 0.

Same hop vs Polyak μ1 first action (not the adjacent-hop contrast):
- mean ΔQ(s0, μ3−Polyak μ1) +0.298.
- mean discounted ΔG -0.002 (std 1.124).

### Hop 4

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ4 | 50.63 (0/20 to) | 51.86 (0/20 to) |
| two-actor deploy | 50.79 (0/20 to) | 55.30 (0/20 to) |

- First action μ4 minus two-actor deploy, MART Polyak continuation: ΔJ -0.17 (std 8.46); discounted ΔG +0.01 (std 1.05). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ4 minus two-actor deploy, two-actor Polyak continuation: ΔJ -3.44 (std 13.71); discounted ΔG -0.19 (std 0.76). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ4 vs Polyak μ1, same MART continuation: mean Δ score -1.71 (std 8.98, 11/20 negative).
- Hybrid mean length change -15.6; hybrid timeouts 0/20.

- Full-episode MART μ4 (existing hop audit): scores 100.75/100.71, lengths 1000/1000.

Adjacent μ4−μ3 under MART Polyak continuation (γ=0.99):
- mean ΔQ(s0) +0.156.
- mean discounted ΔG -0.113 (std 0.991).
- sign agree 10/20; Q↑ G↓ 10; Q↓ G↑ 0.

First-action contrast (MART hop minus two-actor deploy, same continuation) and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first action) are in `contrasts.csv`.

## hopper-medium-replay

### Hop 2

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ2 | 85.50 (7/20 to) | 89.16 (14/20 to) |
| two-actor deploy | 83.85 (10/20 to) | 96.55 (16/20 to) |

- First action μ2 minus two-actor deploy, MART Polyak continuation: ΔJ +1.65 (std 30.82); discounted ΔG +0.07 (std 1.64). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ2 minus two-actor deploy, two-actor Polyak continuation: ΔJ -7.39 (std 24.41); discounted ΔG -0.47 (std 1.82). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ2 vs Polyak μ1, same MART continuation: mean Δ score +0.06 (std 25.22, 11/20 negative).
- Hybrid mean length change -8.2; hybrid timeouts 7/20.

- Full-episode MART μ2 (existing hop audit): scores 99.43/100.36, lengths 1000/1000.

### Hop 3

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ3 | 84.39 (11/20 to) | 92.32 (15/20 to) |
| two-actor deploy | 83.85 (10/20 to) | 96.55 (16/20 to) |

- First action μ3 minus two-actor deploy, MART Polyak continuation: ΔJ +0.54 (std 35.33); discounted ΔG +0.15 (std 1.14). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ3 minus two-actor deploy, two-actor Polyak continuation: ΔJ -4.24 (std 17.19); discounted ΔG -0.16 (std 1.98). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ3 vs Polyak μ1, same MART continuation: mean Δ score -1.05 (std 31.60, 10/20 negative).
- Hybrid mean length change -9.9; hybrid timeouts 11/20.

- Full-episode MART μ3 (existing hop audit): scores 100.36/98.94, lengths 1000/1000.

Adjacent μ3−μ2 under MART Polyak continuation (γ=0.99):
- mean ΔQ(s0) +0.638.
- mean discounted ΔG +0.081 (std 1.089).
- sign agree 13/20; Q↑ G↓ 7; Q↓ G↑ 0.

Same hop vs Polyak μ1 first action (not the adjacent-hop contrast):
- mean ΔQ(s0, μ3−Polyak μ1) +1.436.
- mean discounted ΔG -0.555 (std 1.467).

### Hop 4

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ4 | 85.28 (11/20 to) | 97.09 (18/20 to) |
| two-actor deploy | 83.85 (10/20 to) | 96.55 (16/20 to) |

- First action μ4 minus two-actor deploy, MART Polyak continuation: ΔJ +1.43 (std 27.50); discounted ΔG +0.20 (std 1.09). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ4 minus two-actor deploy, two-actor Polyak continuation: ΔJ +0.54 (std 14.84); discounted ΔG -0.36 (std 1.86). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ4 vs Polyak μ1, same MART continuation: mean Δ score -0.16 (std 26.56, 10/20 negative).
- Hybrid mean length change -4.2; hybrid timeouts 11/20.

- Full-episode MART μ4 (existing hop audit): scores 100.75/100.10, lengths 1000/1000.

Adjacent μ4−μ3 under MART Polyak continuation (γ=0.99):
- mean ΔQ(s0) +0.354.
- mean discounted ΔG +0.052 (std 0.778).
- sign agree 10/20; Q↑ G↓ 10; Q↓ G↑ 0.

First-action contrast (MART hop minus two-actor deploy, same continuation) and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first action) are in `contrasts.csv`.

## hopper-expert

### Hop 2

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ2 | 106.29 (14/20 to) | 105.24 (14/20 to) |
| two-actor deploy | 105.76 (15/20 to) | 108.51 (18/20 to) |

- First action μ2 minus two-actor deploy, MART Polyak continuation: ΔJ +0.52 (std 16.58); discounted ΔG +0.04 (std 1.09). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ2 minus two-actor deploy, two-actor Polyak continuation: ΔJ -3.27 (std 13.97); discounted ΔG +0.24 (std 0.49). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ2 vs Polyak μ1, same MART continuation: mean Δ score +5.22 (std 20.74, 8/20 negative).
- Hybrid mean length change +50.2; hybrid timeouts 14/20.

- Full-episode MART μ2 (existing hop audit): scores 103.45/95.36, lengths 910/839.

### Hop 3

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ3 | 104.64 (9/20 to) | 107.08 (16/20 to) |
| two-actor deploy | 105.76 (15/20 to) | 108.51 (18/20 to) |

- First action μ3 minus two-actor deploy, MART Polyak continuation: ΔJ -1.13 (std 16.83); discounted ΔG +0.28 (std 0.53). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ3 minus two-actor deploy, two-actor Polyak continuation: ΔJ -1.43 (std 15.22); discounted ΔG +0.20 (std 0.51). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ3 vs Polyak μ1, same MART continuation: mean Δ score +3.57 (std 22.94, 10/20 negative).
- Hybrid mean length change +26.1; hybrid timeouts 9/20.

- Full-episode MART μ3 (existing hop audit): scores 90.46/46.09, lengths 782/415.

Adjacent μ3−μ2 under MART Polyak continuation (γ=0.99):
- mean ΔQ(s0) +0.236.
- mean discounted ΔG +0.234 (std 1.025).
- sign agree 10/20; Q↑ G↓ 10; Q↓ G↑ 0.

Same hop vs Polyak μ1 first action (not the adjacent-hop contrast):
- mean ΔQ(s0, μ3−Polyak μ1) +0.216.
- mean discounted ΔG +0.130 (std 0.981).

### Hop 4

| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |
| --- | ---: | ---: |
| MART μ4 | 104.43 (12/20 to) | 104.76 (14/20 to) |
| two-actor deploy | 105.76 (15/20 to) | 108.51 (18/20 to) |

- First action μ4 minus two-actor deploy, MART Polyak continuation: ΔJ -1.34 (std 20.32); discounted ΔG +0.43 (std 0.82). ΔJ is normalized score; ΔG is discounted raw return.
- First action μ4 minus two-actor deploy, two-actor Polyak continuation: ΔJ -3.75 (std 10.62); discounted ΔG +0.11 (std 0.76). ΔJ is normalized score; ΔG is discounted raw return.

- First-action swap μ4 vs Polyak μ1, same MART continuation: mean Δ score +3.36 (std 22.61, 6/20 negative).
- Hybrid mean length change +30.7; hybrid timeouts 12/20.

- Full-episode MART μ4 (existing hop audit): scores 72.15/30.88, lengths 619/294.

Adjacent μ4−μ3 under MART Polyak continuation (γ=0.99):
- mean ΔQ(s0) +0.131.
- mean discounted ΔG +0.150 (std 0.749).
- sign agree 10/20; Q↑ G↓ 10; Q↓ G↑ 0.

First-action contrast (MART hop minus two-actor deploy, same continuation) and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first action) are in `contrasts.csv`.

## Read

On Hopper score, a small first-action gap with a larger full-episode μk gap
supports a continuation difference on the evaluated initial states. That is not
a claim that first actions are equivalent on every metric, nor that this
isolates the training Bellman target: continuation is deterministic Polyak,
without TD3 target noise. Timeout counts and ΔJ std are per cell; a small
mean ΔJ is not policy equality. MART μ4 is in the same 2×2 so the grid can
be read against the deployed MART4 actor, not only intermediate hops.

Walker-medium is not in this dump: those 1M checkpoints are not on ext_csv.

