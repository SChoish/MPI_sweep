# Host checkpoint diagnostics

Package root: `scripts/diagnostics/`

| File | Role |
| --- | --- |
| [HOST_RUNBOOK.md](HOST_RUNBOOK.md) | How to run on each machine |
| [ARTIFACT_MAP.md](ARTIFACT_MAP.md) | Maps `sweep_results/diagnostics/*` → scripts / status |
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
