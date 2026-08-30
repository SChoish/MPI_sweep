# Source and Claim Manifest

## Manuscript basis

- Current AISTATS-26 manuscript source: `../aistats26_manuscript/`
- Trainer and dataset implementation: `../train_td3bc.py`, `../d4rl_data.py`
- Complete sweep matrices: `../sweep_results/K=*/.../seed{0,1}.csv`
- Compact diagnostic audit: `../sweep_results/diagnostics/audit_pack/`
- Targeted controls: `../sweep_results/diagnostics/control_*.csv`
- Frozen-critic and route/simulator summaries: `../sweep_results/diagnostics/`

## Bundled compact data

- `data/aggregate_summary.csv`: low/high/tail means and collapse counts.
- `data/budget_means.csv`: mean score at each of fourteen nominal coefficient budgets.
- `data/environment_high_means.csv`: per-task high-budget means with consistent E labels.
- `data/task_cluster_uncertainty.csv`: descriptive task-resampling summaries supplied by the audit.
- `data/audit_claims.json`: claim-level target, failure, simulator, route, and control statistics.
- `data/config_summary.csv`: implementation and archived-run configuration ledger.

## Claim provenance

- `25.97 -> 63.61`, `35/63 -> 9/63`: recomputed from complete seed-0/1 score matrices.
- K4-K1 and K4-K3 task intervals: descriptive task-cluster resampling over nine per-task high-budget means.
- `88/90`, `84/90`, `36/90`: recomputed by the compact audit from matched run/checkpoint outputs.
- Route `8/8` primary and `5/8` final-checkpoint sensitivity: `route_intervention_run.csv` and `sensitivity.csv`.
- Simulator `-41.4` vs `1.70e12`: medians of the checkpoint-level signed-error medians.
- Failure medians/common-critic sign reversals: `failure_diagnostics_run.csv` and `claim_reproduction.json`.

## Evidence not duplicated

The original multi-gigabyte NPZ/per-state raw diagnostic inputs remain on the authors' machines. Their checksums, row contracts, and re-aggregated compact outputs are documented in the audit pack. The GPT bundle does not claim to independently rerun training or simulator rollouts.
