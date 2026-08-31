# Source and Claim Manifest

## Manuscript basis

- Prior scientific manuscript source: `../aistats26_manuscript/`
- Trainer and dataset implementation: `../train_td3bc.py`, `../d4rl_data.py`
- Complete proximal sweep matrices: `../sweep_results/K={1,2,3,4}/Imp/seed{0,1,2,3}.csv`
- Complete linearized sweep matrices: `../sweep_results/K={2,3}/Exp/seed{0,1}.csv`
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

- `25.38 -> 63.97`, `34/63 -> 8/63`, and the rescued-cell gain share `54.7%`: recomputed from the pooled four-seed proximal matrices.
- Disjoint host-separated pair depth sequences (`25.97,42.02,52.78,63.61` and `24.79,40.38,51.05,64.33`): recomputed separately from seeds 0--1 and 2--3.
- K4-K1 and K4-K3 score and collapse-risk intervals: descriptive resampling over nine task-level four-seed high-budget summaries.
- `88/90`, `84/90`, `36/90`: recomputed by the compact audit from matched run/checkpoint outputs.
- Route `8/8` primary and `5/8` final-checkpoint sensitivity: `route_intervention_run.csv` and `sensitivity.csv`.
- Simulator `-41.4` vs `1.70e12`: medians of the checkpoint-level signed-error medians.
- Failure medians/common-critic sign reversals: `failure_diagnostics_run.csv` and `claim_reproduction.json`.

## Packaged evidence

`package_bundle.py` includes the source score matrices and the four compact audit CSVs read by `verify_bundle.py`. It replaces absolute checkpoint fields with deterministic `artifact://checkpoint/...` identifiers and rejects local workspace identifiers before writing the ZIP.

## Evidence not duplicated

The original multi-gigabyte NPZ/per-state raw diagnostic inputs remain on author-held machines. Their checksums, row contracts, and re-aggregated compact outputs are documented in the audit pack. Exact per-run configuration manifests and code hashes for proximal seeds 2--3 are also not retained in the repository; their score matrices and host-separated replication summaries are retained. The GPT bundle does not claim to independently rerun training or simulator rollouts.
