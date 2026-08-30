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
  --seeds 2 3 \
  --job ckpt
```

Writes compact CSVs to:

```text
sweep_results/diagnostics/hosts/<host>/
```

Then commit **only that host folder** and push. Do not move large `params_*.pkl`.

## Important

`run_host.sh --job ckpt` does **not** recreate the entire archived
`sweep_results/diagnostics/` tree. See [ARTIFACT_MAP.md](ARTIFACT_MAP.md).
