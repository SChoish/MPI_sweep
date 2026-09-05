# P0 shared-driver protocol

Status: **protocol frozen, no launch**.

The existing BAR versus two-actor comparison changes first-actor work,
anchoring, and the critic trajectory. This experiment keeps those objects
shared and branches only how the deployment policy is built.

The frozen design is
[`P0_SHARED_DRIVER_PROTOCOL.json`](P0_SHARED_DRIVER_PROTOCOL.json).
This runbook does not override that file.

## Shared objects

One driver updates, in order:

1. critic, using the Polyak first actor for Bellman targets;
2. the online first actor at budget `T/K`, then Polyak targets;
3. deployment actors only.

Minibatch indices and the JAX RNG stream are identical across branches.
Downstream actors never enter backups.

## Deployment branches

| Branch | Construction | Actors |
| --- | --- | --- |
| `recenter` | hop `k>=2` is proximal to `actor_{k-1}` | `K` |
| `data_anchor` | one extra actor is dataset-anchored at `T` | 2 |
| `fixed_ref` | hop `k>=2` is proximal to the just-updated first actor | `K` |
| `data_anchor_matched` | repeat the dataset-anchored update `K-1` times | 2; secondary |

`recenter` must reproduce current implicit BAR. `data_anchor` must reproduce
the current two-actor control. `tests/test_shared_driver.py` checks both, and
checks that first-actor / critic / target parameters stay identical.

## Primary analysis, when launched

Nine paper tasks, `T={4,7,10,14,20}`, `K=4`, seeds `{0,1}`.
Primary signed contrast: shared-`recenter` endpoint minus shared-`data_anchor`
endpoint, task-equal mean, 100,000-draw task bootstrap, author band `±3`.
Record first-actor returns on every branch. Do not splice historical BAR or
mcep scores into those pairs.

## CLI, not a launch

```bash
python train_td3bc.py --method shared --deployment-branch recenter --mpi-steps 4
python train_td3bc.py --method shared --deployment-branch data_anchor --mpi-steps 4
python train_td3bc.py --method shared --deployment-branch fixed_ref --mpi-steps 4
```

Do not start a queue from this runbook. A later host freeze must write an
environment-locked `FROZEN_MANIFEST.json` first, on one resolved stack.
