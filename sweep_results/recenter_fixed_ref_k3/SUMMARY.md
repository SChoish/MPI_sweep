# TD3+BC K=3 recenter versus fixed_ref

Updated 2026-09-25 01:09:29 UTC+09:00.

Final checkpoint only. `d4rl_pi3` is the deployment actor. `d4rl_score` is the first actor. Paired difference is recenter minus fixed_ref. Pilot scores are not included.

Scored runs 72/216. A blank cell has no 1M eval yet. Cell means use only seeds where both branches are scored.

## Deployment actor `d4rl_pi3`

| env | T | n | recenter | fixed_ref | delta |
|---|---:|---:|---:|---:|---:|
| hopper-medium-v2 | 0.4 | 4/4 | 49.5 | 49.5 | 0.0 |
| hopper-medium-v2 | 2.5 | 4/4 | 55.6 | 51.4 | 4.2 |
| hopper-medium-v2 | 10 | 4/4 | 75.3 | 67.9 | 7.5 |
| hopper-medium-replay-v2 | 0.4 | 4/4 | 41.2 | 41.6 | -0.4 |
| hopper-medium-replay-v2 | 2.5 | 4/4 | 88.0 | 76.2 | 11.8 |
| hopper-medium-replay-v2 | 10 | 4/4 | 99.9 | 99.8 | 0.1 |
| hopper-expert-v2 | 0.4 | 2/4 | 95.9 | 107.7 | -11.8 |
| hopper-expert-v2 | 2.5 | 2/4 | 96.4 | 98.6 | -2.2 |
| hopper-expert-v2 | 10 | 2/4 | 3.0 | 4.9 | -1.9 |
| halfcheetah-medium-v2 | 0.4 | 2/4 | 45.5 | 45.0 | 0.6 |
| halfcheetah-medium-v2 | 2.5 | 2/4 | 51.4 | 49.5 | 1.9 |
| halfcheetah-medium-v2 | 10 | 1/4 | 59.3 | 57.3 | 2.0 |
| halfcheetah-medium-replay-v2 | 0.4 | 0/4 | — | — | — |
| halfcheetah-medium-replay-v2 | 2.5 | 0/4 | — | — | — |
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

Equal-weight mean of scored cell deltas: 1.0 over 12/27 cells.

## First actor `d4rl_score`

| env | T | n | recenter | fixed_ref | delta |
|---|---:|---:|---:|---:|---:|
| hopper-medium-v2 | 0.4 | 4/4 | 45.1 | 45.1 | 0.0 |
| hopper-medium-v2 | 2.5 | 4/4 | 45.3 | 45.3 | 0.0 |
| hopper-medium-v2 | 10 | 4/4 | 45.5 | 45.5 | 0.0 |
| hopper-medium-replay-v2 | 0.4 | 4/4 | 35.4 | 35.4 | 0.0 |
| hopper-medium-replay-v2 | 2.5 | 4/4 | 42.7 | 42.7 | 0.0 |
| hopper-medium-replay-v2 | 10 | 4/4 | 60.1 | 60.1 | 0.0 |
| hopper-expert-v2 | 0.4 | 2/4 | 100.7 | 100.7 | 0.0 |
| hopper-expert-v2 | 2.5 | 2/4 | 109.6 | 109.6 | 0.0 |
| hopper-expert-v2 | 10 | 2/4 | 18.9 | 18.9 | 0.0 |
| halfcheetah-medium-v2 | 0.4 | 2/4 | 43.8 | 43.8 | 0.0 |
| halfcheetah-medium-v2 | 2.5 | 2/4 | 47.2 | 47.2 | 0.0 |
| halfcheetah-medium-v2 | 10 | 1/4 | 52.5 | 52.5 | 0.0 |
| halfcheetah-medium-replay-v2 | 0.4 | 0/4 | — | — | — |
| halfcheetah-medium-replay-v2 | 2.5 | 0/4 | — | — | — |
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

Equal-weight mean of scored first-actor cell deltas: 0.0 over 12/27 cells.

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
| hopper-expert-v2 | 0.4 | 85.0 / 107.7 / -22.6 | 109.9 / — / — | 106.8 / 107.8 / -1.0 | — / — / — |
| hopper-expert-v2 | 2.5 | 89.9 / 100.9 / -11.0 | — / — / — | 102.9 / 96.4 / 6.6 | — / — / — |
| hopper-expert-v2 | 10 | 1.8 / 2.1 / -0.4 | — / — / — | 4.2 / 7.7 / -3.4 | — / — / — |
| halfcheetah-medium-v2 | 0.4 | 45.3 / 45.4 / 0.0 | — / — / — | 45.7 / 44.5 / 1.2 | — / — / — |
| halfcheetah-medium-v2 | 2.5 | 51.0 / 49.3 / 1.8 | — / — / — | 51.8 / 49.7 / 2.1 | — / — / — |
| halfcheetah-medium-v2 | 10 | 59.3 / 57.3 / 2.0 | — / — / — | 58.6 / — / — | — / — / — |
| halfcheetah-medium-replay-v2 | 0.4 | — / — / — | — / — / — | — / — / — | — / — / — |
| halfcheetah-medium-replay-v2 | 2.5 | — / — / — | — / — / — | — / — / — | — / — / — |
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
