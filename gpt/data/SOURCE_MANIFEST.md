# Source manifest

This GPT manuscript was constructed from the following repository material.

## Direct score matrices

- `sweep_results/K=1/Imp/seed0.csv`
- `sweep_results/K=1/Imp/seed1.csv`
- `sweep_results/K=1/Imp/seed2.csv`
- `sweep_results/K=1/Imp/seed3.csv`
- `sweep_results/K=2/Imp/seed0.csv`
- `sweep_results/K=2/Imp/seed1.csv`
- `sweep_results/K=3/Imp/seed0.csv`
- `sweep_results/K=3/Imp/seed1.csv`
- `sweep_results/K=4/Imp/seed0.csv`
- `sweep_results/K=4/Imp/seed1.csv`
- `sweep_results/K=2/Exp/seed0.csv`
- `sweep_results/K=2/Exp/seed1.csv`
- `sweep_results/K=3/Exp/seed0.csv`
- `sweep_results/K=3/Exp/seed1.csv`

The `gpt/data/raw` directory freezes the raw matrices needed for the new K=4 comparison and the independent K=1 replication. Aggregate arrays for all complete methods are stored in `budget_means.csv` and `aggregate_summary.csv`.

## Manuscript and code audit

- `aistats27_manuscript/`
- `train_td3bc.py`
- `launch_mpi_sweep.py`
- `tau_grids.py`
- root `.gitignore`

## Unavailable raw material

The reviewed repository snapshot does not include raw training logs, checkpoints, transition-level diagnostic outputs, or scripts sufficient to reconstruct the target-exposure, simulator-reference, shadow-critic, fixed-reference, target-lag, or frozen-critic audits. Those claims are treated as inherited secondary evidence.
