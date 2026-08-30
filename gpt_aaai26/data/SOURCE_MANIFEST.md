# Data source manifest

## Complete sweep summaries

`aggregate_summary.csv`, `budget_means.csv`, `collapse_sensitivity.csv`, and `environment_high_means.csv` summarize the complete two-seed, fourteen-budget grids for:

- TD3+BC (K=1)
- BAR-Prox K=2, K=3, K=4
- BAR-Lin K=2, K=3

## Frozen raw matrices used by this bundle

- `raw/td3_seed0.csv`, `raw/td3_seed1.csv`
- `raw/prox3_seed0.csv`, `raw/prox3_seed1.csv`
- `raw/prox4_seed0.csv`, `raw/prox4_seed1.csv`
- `raw/k1_rep_seed2.csv`, `raw/k1_rep_seed3.csv`

The first six matrices support direct recomputation of K4-vs-K1 and K4-vs-K3 dataset-cluster uncertainty. The latter two support the independent one-hop seed-pair replication.

## Derived output

`task_cluster_uncertainty.csv` uses 100,000 dataset-level bootstrap resamples with seed 20260830. Each resampled unit retains both seeds and all seven high-budget values T in {4,7,10,12,14,17,20}.

## Diagnostic audit

See `../audit_pack/`. Those files are sanitized compact outputs from the larger repository audit and do not duplicate raw checkpoints or NPZ files.
