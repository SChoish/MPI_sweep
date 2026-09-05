# Host checkpoint diagnostics

Package root: `scripts/diagnostics/`

| File | Role |
| --- | --- |
| [HOST_RUNBOOK.md](HOST_RUNBOOK.md) | How to run on each machine |
| [ARTIFACT_MAP.md](ARTIFACT_MAP.md) | Maps `sweep_results/diagnostics/*` → scripts / status |
| [P1_TARGET_VALUE_RUNBOOK.md](P1_TARGET_VALUE_RUNBOOK.md) | Exact 270-checkpoint P1 audit and verifier |
| [P2_RELU_RESIDENCE_RUNBOOK.md](P2_RELU_RESIDENCE_RUNBOOK.md) | Development-only ReLU activation-region pilot and final-audit contract |
| [ACTOR_COST_RUNBOOK.md](ACTOR_COST_RUNBOOK.md) | Released K=1--4 H200 timing/memory snapshot and rerun protocol |
| [P0_INTERMEDIATE_ACTORS_RUNBOOK.md](P0_INTERMEDIATE_ACTORS_RUNBOOK.md) | CPU all-actor eval and action distances on frozen P0 K=4 checkpoints |
| `verify_p0_score_merge.py` | Independently rebuilds the 180-row/90-pair P0 compact merge and keeps integrity separate from scientific admissibility |
| `run_host.sh` | One-shot wrapper for this host's seeds |
| `dump_*.py` / `run_*.py` / `summarize_*.py` | Implementations |

## Quick start

```bash
cd MPI_sweep && git pull

export LAB_ROOT=/path/to/mpi_sweep_lab   # must match ckpt format (actor_params)
export RESULTS_ROOT=$LAB_ROOT
export DATA_DIR=/raid/$USER/datasets/d4rl
export PY=/path/to/offrl/bin/python

bash scripts/diagnostics/run_host.sh \
  --host "$(hostname -s)" \
  --seeds 0 1 \
  --job ckpt
```

Replace `0 1` with the seeds stored on this machine. Use a stable, unique
`--host` label; the short hostname is the default. `RESULTS_ROOT` may differ
from `LAB_ROOT` as long as it contains the local `results_*` directories.

Writes diagnostics to:

```text
sweep_results/diagnostics/hosts/<host>/
```

The host output policy tracks only `*.csv`, `*.json`, and `*.md`.
Per-state `raw/*.npz`, `common_batches/*.npy`, and logs remain local and are
excluded by `sweep_results/diagnostics/hosts/.gitignore`.

Then commit **only that host folder** and push:

```bash
git add sweep_results/diagnostics/hosts/<host>
git status --short
git commit -m "Add <host> checkpoint diagnostics"
git push
```

Review `git status` before committing. Do not move checkpoints or force-add
ignored raw artifacts.

## Important

`run_host.sh --job ckpt` does **not** recreate the entire archived
`sweep_results/diagnostics/` tree. See [ARTIFACT_MAP.md](ARTIFACT_MAP.md).
The P1 target-value audit and actor-cost profiler are standalone locked
protocols and are not included in any `run_host.sh --job` choice, including
`all`.

The compact `sweep_results/diagnostics/actor_cost/` archive is a measured,
hash-pinned one-stack result, not a host checkpoint dump. The release verifier
recomputes its timing and memory summaries without requiring the local dataset;
the standalone verifier additionally rehashes the recorded source and dataset
when those raw dependencies are available.

The older target-policy exposure dump may operate on a host-local partial
inventory. It does not satisfy P1's exact 270-checkpoint inclusion gate and
must not be substituted for the standalone P1 audit.
