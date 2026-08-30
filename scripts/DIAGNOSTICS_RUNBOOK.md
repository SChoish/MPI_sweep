# Host-local checkpoint diagnostics

Run these on the machine that owns the checkpoints. Do **not** copy large
`params_*.pkl` trees across hosts. Push only the compact CSV/JSON outputs
under `sweep_results/diagnostics/hosts/<host>/`.

## Why `LAB_ROOT`?

Finished matched sweeps on the lab machines use the **lab checkpoint layout**
(`actor_params`, `actor2_params`, …). The release `train_td3bc.py` uses a
different `actors_params` tuple. Diagnostic scripts therefore import
`train_td3bc` from `LAB_ROOT` (your `mpi_sweep_lab` / equivalent checkout).

```bash
export LAB_ROOT=/home/$USER/mpi_sweep_lab
export RESULTS_ROOT=$LAB_ROOT          # dirs like results_mpi3, results_expl3_matched
export DATA_DIR=/raid/$USER/datasets/d4rl
export PY=/home/$USER/miniconda3/envs/offrl/bin/python   # or your jax env
```

## One-shot runner

From the `MPI_sweep` repo root (after `git pull`):

```bash
# this host owns seeds 0 and 1
bash scripts/run_host_diagnostics.sh --host "$(hostname -s)" --seeds 0 1

# another host owns seeds 2 and 3
bash scripts/run_host_diagnostics.sh --host offrl --seeds 2 3 \
  --methods mpi2 mpi3 expl2 expl3 --taus 4 7 10 14 20
```

Jobs:

| `--job` | Script |
| --- | --- |
| `geometry` (default via `all`) | `dump_mpi_frontier_geometry.py` + summarize |
| `exposure` | `dump_target_policy_exposure.py` |
| `all` | both |

Methods understood by the dumpers: `td3`, `mpi2`, `mpi3`, `expl2`, `expl3`
(matched explicit trees).

## Manual equivalents

```bash
export PYTHONPATH=$LAB_ROOT:$PYTHONPATH

$PY -u scripts/dump_mpi_frontier_geometry.py \
  --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
  --out-dir sweep_results/diagnostics/hosts/$HOST/matched_geometry \
  --seeds 2 3 --methods expl2 expl3 --taus 4 7 10 14 20

$PY -u scripts/dump_target_policy_exposure.py \
  --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
  --out-dir sweep_results/diagnostics/hosts/$HOST/target_policy_exposure \
  --seeds 2 3 --methods expl2 expl3 --taus 4 7 10 14 20
```

Other scripts in `scripts/` (frozen-critic small-step, route-shadow MC, MC
value audit) are available the same way; they are not wired into the one-shot
runner because they need narrower cell lists.

## Return path

1. Inspect CSVs locally.
2. `git add sweep_results/diagnostics/hosts/<host> && git commit && git push`
3. Aggregator / manuscript machine pulls; do not rewrite another host's folder.

## Checkpoint contract

- Prefer `params_1000000.pkl` (+ final `eval.csv` row at step 1M).
- Skip missing cells (dumpers log `[missing]` unless `--strict-missing`).
- Keep GPU free or force CPU (`dump_mpi_frontier_geometry.py` already defaults
  to CPU via `JAX_PLATFORMS`).
