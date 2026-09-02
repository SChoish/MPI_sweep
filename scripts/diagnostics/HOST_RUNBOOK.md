# Host runbook

## Prerequisites

1. `MPI_sweep` repo (this package) on the machine.
2. `LAB_ROOT`: checkout whose `train_td3bc.py` matches your checkpoints
   (lab layout: `actor_params` / `actor2_params` / …).
3. Local `results_*` trees under `RESULTS_ROOT`.
4. D4RL HDF5 under `DATA_DIR`.
5. Python env with JAX + the same deps used for training (`offrl` / equivalent).

Release `train_td3bc.py` uses `actors_params` tuples and is **not** compatible
with lab matched-sweep checkpoints. Always set `LAB_ROOT`.
`RESULTS_ROOT` may be a different directory; it only needs to contain the
expected local `results_*` trees.

## Seed ownership (example)

| Host | Seeds |
| --- | --- |
| example host A | 0 1 |
| example host B | 2 3 |

This table is illustrative, not an assignment. Only dump seeds that exist
**locally**, and use a stable unique host label for the output folder.

## Jobs

| `--job` | What runs | Needs |
| --- | --- | --- |
| `ckpt` (default) | geometry + target-policy exposure | finished 1M ckpts in `results_mpi*` / `results_expl*_matched` / `results_qnorm` |
| `geometry` | `dump_mpi_frontier_geometry.py` + summarize | same |
| `exposure` | `dump_target_policy_exposure.py` | same |
| `frozen` | `run_frozen_critic_small_step.py` | TD3+BC (`results_qnorm`) ckpts; set `FROZEN_RESULTS_DIR` if needed |
| `route` | `run_route_shadow_mc.py` | `results_route_shadow` runs; set `ROUTE_RESULTS_DIR` if needed |
| `all` | ckpt + frozen + route | all of the above |

The standalone [P1 target-value audit](P1_TARGET_VALUE_RUNBOOK.md) and
[actor-cost profiler](ACTOR_COST_RUNBOOK.md) are deliberately absent from this
table and from `--job all`. P1 has a stricter exact-270-checkpoint gate than
the host-local exposure dump; actor cost is a fresh hardware benchmark rather
than a checkpoint dump.

## Commands

```bash
# seeds on this host
bash scripts/diagnostics/run_host.sh --host offrl --seeds 2 3 --job ckpt

# geometry only, custom taus/methods
bash scripts/diagnostics/run_host.sh --host offrl --seeds 2 3 \
  --methods expl2 expl3 mpi3 --taus 4 7 10 14 20 --job geometry
```

## After the dump

```bash
cd MPI_sweep
git add sweep_results/diagnostics/hosts/<host>
git status --short
git commit -m "Add <host> checkpoint diagnostics"
git push
```

The repository allowlist under `hosts/.gitignore` stages only CSV, JSON, and
Markdown summaries. Generated `raw/*.npz`, `common_batches/*.npy`, and logs
stay local. Do not use `git add -f` for them, and do not edit another host's
directory.

## Manual script calls

```bash
export HOST="$(hostname -s)"
export PYTHONPATH=$LAB_ROOT:$PYTHONPATH

$PY -u scripts/diagnostics/dump_mpi_frontier_geometry.py \
  --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
  --out-dir sweep_results/diagnostics/hosts/$HOST/matched_geometry \
  --seeds 2 3 --methods mpi2 mpi3 expl2 expl3 --taus 4 7 10 14 20

$PY -u scripts/diagnostics/summarize_mpi_frontier_geometry.py \
  sweep_results/diagnostics/hosts/$HOST/matched_geometry

$PY -u scripts/diagnostics/dump_target_policy_exposure.py \
  --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
  --out-dir sweep_results/diagnostics/hosts/$HOST/target_policy_exposure \
  --seeds 2 3 --methods mpi2 mpi3 expl2 expl3 --taus 4 7 10 14 20
```
