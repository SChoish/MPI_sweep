# TD3+BC K=4 recenter versus fixed_ref

Updated 2026-09-26 09:09:12 UTC+09:00.

Host `svcho` only. Seeds follow `(env_index + seed) % 2 == 0`. choi seeds are blank. T=10 is still training and is not in this file.

Final checkpoint only. `d4rl_pi4` is the deployment actor. `d4rl_score` is the first actor. Paired difference is recenter minus fixed_ref. Pilot scores are not included. Pair verification did not pass bitwise pi_2 equality on GPU; scores are still the 1M eval rows.

Scored runs 72/144 primary. A blank cell has no 1M eval yet. Cell means use only seeds where both branches are scored.

## Deployment actor `d4rl_pi4`

| env | T | n | recenter | fixed_ref | delta |
|---|---:|---:|---:|---:|---:|
| hopper-medium-v2 | 4 | 2/2 | 62.0 | 46.6 | 15.4 |
| hopper-medium-v2 | 7 | 2/2 | 84.5 | 53.2 | 31.3 |
| hopper-medium-replay-v2 | 4 | 2/2 | 98.1 | 92.2 | 5.9 |
| hopper-medium-replay-v2 | 7 | 2/2 | 98.7 | 98.4 | 0.3 |
| hopper-expert-v2 | 4 | 2/2 | 107.2 | 105.5 | 1.7 |
| hopper-expert-v2 | 7 | 2/2 | 108.2 | 105.4 | 2.8 |
| halfcheetah-medium-v2 | 4 | 2/2 | 52.8 | 50.7 | 2.1 |
| halfcheetah-medium-v2 | 7 | 2/2 | 56.0 | 53.2 | 2.8 |
| halfcheetah-medium-replay-v2 | 4 | 2/2 | 50.2 | 46.9 | 3.4 |
| halfcheetah-medium-replay-v2 | 7 | 2/2 | 53.3 | 49.0 | 4.4 |
| halfcheetah-expert-v2 | 4 | 2/2 | 71.8 | 97.4 | -25.7 |
| halfcheetah-expert-v2 | 7 | 2/2 | 51.1 | 75.4 | -24.3 |
| walker2d-medium-v2 | 4 | 2/2 | 86.9 | 85.4 | 1.5 |
| walker2d-medium-v2 | 7 | 2/2 | 90.8 | 89.5 | 1.2 |
| walker2d-medium-replay-v2 | 4 | 2/2 | 85.4 | 84.1 | 1.3 |
| walker2d-medium-replay-v2 | 7 | 2/2 | 88.3 | 91.1 | -2.8 |
| walker2d-expert-v2 | 4 | 2/2 | 111.8 | 110.5 | 1.2 |
| walker2d-expert-v2 | 7 | 2/2 | 112.8 | 111.4 | 1.5 |

Equal-weight mean of scored cell deltas: 1.3 over 18/18 svcho cells.

## First actor `d4rl_score`

| env | T | n | recenter | fixed_ref | delta |
|---|---:|---:|---:|---:|---:|
| hopper-medium-v2 | 4 | 2/2 | 50.0 | 50.0 | 0.0 |
| hopper-medium-v2 | 7 | 2/2 | 52.1 | 52.1 | 0.0 |
| hopper-medium-replay-v2 | 4 | 2/2 | 37.0 | 37.0 | 0.0 |
| hopper-medium-replay-v2 | 7 | 2/2 | 45.5 | 45.5 | 0.0 |
| hopper-expert-v2 | 4 | 2/2 | 104.3 | 104.3 | 0.0 |
| hopper-expert-v2 | 7 | 2/2 | 99.6 | 99.6 | 0.0 |
| halfcheetah-medium-v2 | 4 | 2/2 | 47.2 | 47.2 | 0.0 |
| halfcheetah-medium-v2 | 7 | 2/2 | 49.2 | 49.2 | 0.0 |
| halfcheetah-medium-replay-v2 | 4 | 2/2 | 44.3 | 44.3 | 0.0 |
| halfcheetah-medium-replay-v2 | 7 | 2/2 | 45.2 | 45.2 | 0.0 |
| halfcheetah-expert-v2 | 4 | 2/2 | 96.2 | 96.2 | 0.0 |
| halfcheetah-expert-v2 | 7 | 2/2 | 91.6 | 91.6 | 0.0 |
| walker2d-medium-v2 | 4 | 2/2 | 83.7 | 83.7 | 0.0 |
| walker2d-medium-v2 | 7 | 2/2 | 86.5 | 86.5 | 0.0 |
| walker2d-medium-replay-v2 | 4 | 2/2 | 69.4 | 69.4 | 0.0 |
| walker2d-medium-replay-v2 | 7 | 2/2 | 82.1 | 82.1 | 0.0 |
| walker2d-expert-v2 | 4 | 2/2 | 109.3 | 109.3 | 0.0 |
| walker2d-expert-v2 | 7 | 2/2 | 110.0 | 110.0 | 0.0 |

Equal-weight mean of scored first-actor cell deltas: 0.0 over 18/18 svcho cells.

## Seeds, deployment `d4rl_pi4`

Each seed column is `recenter / fixed_ref / delta`.

| env | T | s0 | s1 | s2 | s3 |
|---|---:|---|---|---|---|
| hopper-medium-v2 | 4 | 75.3 / 48.8 / 26.5 | — / — / — | 48.8 / 44.5 / 4.3 | — / — / — |
| hopper-medium-v2 | 7 | 90.2 / 57.6 / 32.6 | — / — / — | 78.9 / 48.8 / 30.1 | — / — / — |
| hopper-medium-replay-v2 | 4 | — / — / — | 98.6 / 99.8 / -1.3 | — / — / — | 97.6 / 84.5 / 13.1 |
| hopper-medium-replay-v2 | 7 | — / — / — | 99.1 / 99.5 / -0.4 | — / — / — | 98.4 / 97.4 / 1.0 |
| hopper-expert-v2 | 4 | 105.4 / 109.0 / -3.6 | — / — / — | 109.0 / 102.0 / 7.0 | — / — / — |
| hopper-expert-v2 | 7 | 106.4 / 110.1 / -3.7 | — / — / — | 109.9 / 100.6 / 9.3 | — / — / — |
| halfcheetah-medium-v2 | 4 | — / — / — | 52.5 / 50.9 / 1.7 | — / — / — | 53.0 / 50.6 / 2.4 |
| halfcheetah-medium-v2 | 7 | — / — / — | 56.2 / 53.2 / 3.0 | — / — / — | 55.8 / 53.1 / 2.7 |
| halfcheetah-medium-replay-v2 | 4 | 49.4 / 46.1 / 3.3 | — / — / — | 51.1 / 47.6 / 3.5 | — / — / — |
| halfcheetah-medium-replay-v2 | 7 | 52.3 / 48.9 / 3.3 | — / — / — | 54.4 / 49.0 / 5.4 | — / — / — |
| halfcheetah-expert-v2 | 4 | — / — / — | 77.8 / 96.8 / -19.0 | — / — / — | 65.8 / 98.1 / -32.3 |
| halfcheetah-expert-v2 | 7 | — / — / — | 55.9 / 62.2 / -6.3 | — / — / — | 46.3 / 88.7 / -42.4 |
| walker2d-medium-v2 | 4 | 86.7 / 84.9 / 1.8 | — / — / — | 87.1 / 85.9 / 1.1 | — / — / — |
| walker2d-medium-v2 | 7 | 90.5 / 88.9 / 1.6 | — / — / — | 91.1 / 90.2 / 0.9 | — / — / — |
| walker2d-medium-replay-v2 | 4 | — / — / — | 87.5 / 82.7 / 4.8 | — / — / — | 83.4 / 85.6 / -2.2 |
| walker2d-medium-replay-v2 | 7 | — / — / — | 85.1 / 92.1 / -7.0 | — / — / — | 91.5 / 90.2 / 1.3 |
| walker2d-expert-v2 | 4 | 112.2 / 110.5 / 1.7 | — / — / — | 111.4 / 110.6 / 0.8 | — / — / — |
| walker2d-expert-v2 | 7 | 113.3 / 111.3 / 2.0 | — / — / — | 112.4 / 111.5 / 1.0 | — / — / — |
