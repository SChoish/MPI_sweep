# Recomputed and Sanitized Diagnostics Audit

## Result

All deterministic reproduction checks used by the revised manuscript pass. The pack is a second aggregation path over preserved local diagnostic outputs; it is not an independent rerun of policy training or simulator rollouts.

## Integrity

- Target exposure: 270 rows; exact state pairing; no NaN or Inf.
- Failure diagnostics: 180 rows; exact fixed-state pairing; no NaN or Inf.
- Simulator screen: 60 checkpoints and 720 states.
- Route intervention: 8 training runs and 32 exact checkpoint-arm pairings.
- Original audit hashed 2,135 inputs totaling 3,180,555,260 bytes.

## Substantive audit conclusions

1. The target-facing displacement result is robust to cluster resampling: BAR-P3 is below TD3+BC in 88/90 matched cells, with median ratio 0.6577.
2. Final-policy movement is heterogeneous. It is lower in 36/90 current-state proxy comparisons; the median ratio is 1.057, while a few large contractions make the arithmetic mean difference negative.
3. Failure signatures reproduce, but 90 TD-error rows rely on an archived run-level diagnostic dump because the source NPZs did not retain per-state residual arrays.
4. The old simulator values mixed aggregation language. The revision uses the median of checkpoint-level medians: -41.37 for 55 stable checkpoints and 1.697e12 for five collapsed checkpoints.
5. The route result is 8/8 only for an average across four checkpoints (and for state-mean/trimmed variants). It is 5/8 at the final checkpoint alone. The four cells were selected post hoc, and the estimand is alignment with the common first-route continuation.
6. Identical critic initialization across K is not a validity requirement for marginal seed-averaged method comparisons. It is an optional common-random-number design that can reduce variance; the revised critique no longer treats its absence as a confound.

## Files

- `claim_summary.json`: machine-readable headline audit.
- `task_cluster_uncertainty.csv`: K4 score and collapse-risk task bootstrap.
- `target_exposure_summary.csv`: matched target/final movement summaries.
- `failure_summary.csv`: stable/collapsed diagnostic medians.
- `simulator_aggregation.csv`: pooled-state versus checkpoint-level aggregation.
- `route_sensitivity.csv`: aggregation sensitivity.
- `claim_ledger.csv`: claim-by-claim evidential status.
- `verify_audit_pack.py`: deterministic consistency checks over this compact pack.
