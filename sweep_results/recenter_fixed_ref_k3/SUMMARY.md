# TD3+BC K=3 recenter versus fixed_ref

Updated 2026-09-25 01:59:58 UTC+09:00.

Final checkpoint only. `d4rl_pi3` is the deployment actor. `d4rl_score` is the first actor. Paired difference is recenter minus fixed_ref. Pilot scores are not included.

Scored runs 91/216. A blank cell has no 1M eval yet. Cell means use only seeds where both branches are scored.

## Deployment actor `d4rl_pi3`

| env | T | n | recenter | fixed_ref | delta |
|---|---:|---:|---:|---:|---:|
| hopper-medium-v2 | 0.4 | 4/4 | 49.5 | 49.5 | 0.0 |
| hopper-medium-v2 | 2.5 | 4/4 | 55.6 | 51.4 | 4.2 |
| hopper-medium-v2 | 10 | 4/4 | 75.3 | 67.9 | 7.5 |
| hopper-medium-replay-v2 | 0.4 | 4/4 | 41.2 | 41.6 | -0.4 |
| hopper-medium-replay-v2 | 2.5 | 4/4 | 88.0 | 76.2 | 11.8 |
| hopper-medium-replay-v2 | 10 | 4/4 | 99.9 | 99.8 | 0.1 |
| hopper-expert-v2 | 0.4 | 4/4 | 102.2 | 106.0 | -3.8 |
| hopper-expert-v2 | 2.5 | 4/4 | 103.3 | 104.4 | -1.1 |
| hopper-expert-v2 | 10 | 3/4 | 30.3 | 31.3 | -1.1 |
| halfcheetah-medium-v2 | 0.4 | 2/4 | 45.5 | 45.0 | 0.6 |
| halfcheetah-medium-v2 | 2.5 | 2/4 | 51.4 | 49.5 | 1.9 |
| halfcheetah-medium-v2 | 10 | 2/4 | 58.9 | 56.9 | 2.0 |
| halfcheetah-medium-replay-v2 | 0.4 | 2/4 | 43.8 | 42.9 | 0.9 |
| halfcheetah-medium-replay-v2 | 2.5 | 2/4 | 47.6 | 46.5 | 1.2 |
| halfcheetah-medium-replay-v2 | 10 | 0/4 | — | — | — |
| halfcheetah-expert-v2 | 0.4 | 0/4 | — | — | — |
| halfcheetah-expert-v2 | 2.5 | 0/4 | — | — | — |
| halfcheetah-expert-v2 | 10 | 0/4 | — | — | — |
| walker2d-medium-v2 | 0.4 | 0/4 | — | — | — |
| walker2d-medium-v2 | 2.5 | 0/4 | — | — | — |
| walker2d-medium-v2 | 10 | 0/4 | — | — | — |
| walker2d-medium-replay-v2 | 0.4 | 0/4 | — | — | — |
| walker2d-medium-replay-v2 | 2.5 | 0/4 | — | — | — |
| walker2d-medium-replay-v2 | 10 | 0/4 | — | — | — |
| walker2d-expert-v2 | 0.4 | 0/4 | — | — | — |
| walker2d-expert-v2 | 2.5 | 0/4 | — | — | — |
| walker2d-expert-v2 | 10 | 0/4 | — | — | — |

Equal-weight mean of scored cell deltas: 1.7 over 14/27 cells.

## First actor `d4rl_score`

| env | T | n | recenter | fixed_ref | delta |
|---|---:|---:|---:|---:|---:|
| hopper-medium-v2 | 0.4 | 4/4 | 45.1 | 45.1 | 0.0 |
| hopper-medium-v2 | 2.5 | 4/4 | 45.3 | 45.3 | 0.0 |
| hopper-medium-v2 | 10 | 4/4 | 45.5 | 45.5 | 0.0 |
| hopper-medium-replay-v2 | 0.4 | 4/4 | 35.4 | 35.4 | 0.0 |
| hopper-medium-replay-v2 | 2.5 | 4/4 | 42.7 | 42.7 | 0.0 |
| hopper-medium-replay-v2 | 10 | 4/4 | 60.1 | 60.1 | 0.0 |
| hopper-expert-v2 | 0.4 | 4/4 | 96.5 | 96.5 | 0.0 |
| hopper-expert-v2 | 2.5 | 4/4 | 94.7 | 94.7 | 0.0 |
| hopper-expert-v2 | 10 | 3/4 | 29.3 | 29.3 | 0.0 |
| halfcheetah-medium-v2 | 0.4 | 2/4 | 43.8 | 43.8 | 0.0 |
| halfcheetah-medium-v2 | 2.5 | 2/4 | 47.2 | 47.2 | 0.0 |
| halfcheetah-medium-v2 | 10 | 2/4 | 52.6 | 52.6 | 0.0 |
| halfcheetah-medium-replay-v2 | 0.4 | 2/4 | 40.9 | 40.9 | 0.0 |
| halfcheetah-medium-replay-v2 | 2.5 | 2/4 | 43.9 | 43.9 | 0.0 |
| halfcheetah-medium-replay-v2 | 10 | 0/4 | — | — | — |
| halfcheetah-expert-v2 | 0.4 | 0/4 | — | — | — |
| halfcheetah-expert-v2 | 2.5 | 0/4 | — | — | — |
| halfcheetah-expert-v2 | 10 | 0/4 | — | — | — |
| walker2d-medium-v2 | 0.4 | 0/4 | — | — | — |
| walker2d-medium-v2 | 2.5 | 0/4 | — | — | — |
| walker2d-medium-v2 | 10 | 0/4 | — | — | — |
| walker2d-medium-replay-v2 | 0.4 | 0/4 | — | — | — |
| walker2d-medium-replay-v2 | 2.5 | 0/4 | — | — | — |
| walker2d-medium-replay-v2 | 10 | 0/4 | — | — | — |
| walker2d-expert-v2 | 0.4 | 0/4 | — | — | — |
| walker2d-expert-v2 | 2.5 | 0/4 | — | — | — |
| walker2d-expert-v2 | 10 | 0/4 | — | — | — |

Equal-weight mean of scored first-actor cell deltas: 0.0 over 14/27 cells.

## Seeds, deployment `d4rl_pi3`

Each seed column is `recenter / fixed_ref / delta`.

| env | T | s0 | s1 | s2 | s3 |
|---|---:|---|---|---|---|
| hopper-medium-v2 | 0.4 | 50.7 / 50.4 / 0.3 | 50.9 / 45.9 / 5.0 | 42.7 / 49.4 / -6.7 | 53.7 / 52.2 / 1.4 |
| hopper-medium-v2 | 2.5 | 59.8 / 46.3 / 13.5 | 58.2 / 48.9 / 9.3 | 53.9 / 51.4 / 2.5 | 50.5 / 59.2 / -8.6 |
| hopper-medium-v2 | 10 | 100.6 / 100.3 / 0.2 | 100.8 / 82.0 / 18.8 | 0.7 / 15.0 / -14.3 | 99.3 / 74.1 / 25.2 |
| hopper-medium-replay-v2 | 0.4 | 31.7 / 37.3 / -5.6 | 56.9 / 56.2 / 0.7 | 44.3 / 39.7 / 4.6 | 31.9 / 33.2 / -1.4 |
| hopper-medium-replay-v2 | 2.5 | 65.6 / 68.4 / -2.7 | 99.9 / 98.1 / 1.8 | 98.7 / 99.2 / -0.5 | 87.7 / 39.1 / 48.6 |
| hopper-medium-replay-v2 | 10 | 99.9 / 100.7 / -0.8 | 99.6 / 97.9 / 1.7 | 98.6 / 99.6 / -1.0 | 101.4 / 100.9 / 0.5 |
| hopper-expert-v2 | 0.4 | 85.0 / 107.7 / -22.6 | 109.9 / 101.8 / 8.0 | 106.8 / 107.8 / -1.0 | 107.2 / 106.9 / 0.3 |
| hopper-expert-v2 | 2.5 | 89.9 / 100.9 / -11.0 | 110.4 / 109.9 / 0.5 | 102.9 / 96.4 / 6.6 | 109.9 / 110.6 / -0.7 |
| hopper-expert-v2 | 10 | 1.8 / 2.1 / -0.4 | 84.8 / 84.2 / 0.6 | 4.2 / 7.7 / -3.4 | 2.4 / — / — |
| halfcheetah-medium-v2 | 0.4 | 45.3 / 45.4 / 0.0 | — / — / — | 45.7 / 44.5 / 1.2 | — / — / — |
| halfcheetah-medium-v2 | 2.5 | 51.0 / 49.3 / 1.8 | — / — / — | 51.8 / 49.7 / 2.1 | — / — / — |
| halfcheetah-medium-v2 | 10 | 59.3 / 57.3 / 2.0 | — / — / — | 58.6 / 56.6 / 2.0 | — / — / — |
| halfcheetah-medium-replay-v2 | 0.4 | 43.4 / 42.6 / 0.8 | — / — / — | 44.3 / 43.2 / 1.1 | — / — / — |
| halfcheetah-medium-replay-v2 | 2.5 | 47.2 / 46.3 / 0.9 | — / — / — | 48.0 / 46.6 / 1.4 | — / — / — |
| halfcheetah-medium-replay-v2 | 10 | — / — / — | — / — / — | — / — / — | — / — / — |
| halfcheetah-expert-v2 | 0.4 | — / — / — | — / — / — | — / — / — | — / — / — |
| halfcheetah-expert-v2 | 2.5 | — / — / — | — / — / — | — / — / — | — / — / — |
| halfcheetah-expert-v2 | 10 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-medium-v2 | 0.4 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-medium-v2 | 2.5 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-medium-v2 | 10 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-medium-replay-v2 | 0.4 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-medium-replay-v2 | 2.5 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-medium-replay-v2 | 10 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-expert-v2 | 0.4 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-expert-v2 | 2.5 | — / — / — | — / — / — | — / — / — | — / — / — |
| walker2d-expert-v2 | 10 | — / — / — | — / — / — | — / — / — | — / — / — |
