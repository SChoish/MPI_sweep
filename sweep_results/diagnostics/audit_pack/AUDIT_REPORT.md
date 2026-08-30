# Independent diagnostics audit

Overall deterministic claim reproduction: **PASS**

This pack was rebuilt from existing local NPZ/per-state diagnostic inputs.
No policy training or simulator rollout was rerun.

## Integrity

- Target exposure: 270/270 rows; exact state pairing; NaN=0, inf=0.
- Failure diagnostics: 180/180 rows; exact validation-state pairing; NaN=0, inf=0.
- Simulator reference: 60/60 checkpoints and 720/720 states.
- Route intervention: 8/8 runs; 32 checkpoint arm-pair checks passed.
- Checksums: 2135 inputs, 3180555266 bytes.

## Claim reproduction

- PASS `prox3_vs_td3_target_lower`: observed=88, expected=88
- PASS `prox3_vs_prox2_target_lower`: observed=84, expected=84
- PASS `prox3_vs_td3_final_lower`: observed=36, expected=36
- PASS `simulator_stable`: observed=55, expected=55
- PASS `simulator_collapsed`: observed=5, expected=5
- PASS `simulator_rank_eligible`: observed=69, expected=69
- PASS `simulator_rank_disagreement`: observed=29, expected=29
- PASS `route_positive`: observed=8, expected=8
- PASS `prox3_vs_td3_target_median_ratio`: observed=0.6576946519209848, expected=0.6576946525
- PASS `prox3_vs_prox2_target_median_ratio`: observed=0.9030389773245951, expected=0.9030389793
- PASS `prox3_vs_td3_final_median_ratio`: observed=1.0574135361848942, expected=1.057
- PASS `proxy_target_pearson`: observed=0.9999114535124949, expected=0.9999114535
- PASS `proxy_smoothed_pearson`: observed=0.9992799627436579, expected=0.9992799627
- PASS `route_mean_delta`: observed=0.12351472249786054, expected=0.1235147225
- PASS `route_median_delta`: observed=0.013267972954500764, expected=0.013267973
- PASS `route_sign_test_p`: observed=0.0078125, expected=0.0078125

## Findings and limitations

- Target comparisons use 18-cluster `(env, seed)` bootstrap; T values remain grouped.
- Simulator aggregation was recomputed in both forms. The manuscript values `-38.7` and `1.70e12` correspond to pooled-state medians, not medians of checkpoint-level medians.
- The route result is alignment against the common `Q_MC^{rho_1}` first-route estimand, not own-continuation calibration.
- The K=2 source NPZ files do not retain per-state TD-residual arrays. `td_error_max` is therefore unavailable for 90 rows; TD quantiles are independently checked for count/finiteness but read from the prior run-level diagnostic dump.
- Last-three-checkpoint failure sensitivity is unavailable because the source audit retained final-checkpoint diagnostics only.

## Leakage audit

- Failure signatures are descriptive, not causal.
- Simulator state rows are not treated as independent observations.
- Route inference resamples the eight training runs, not checkpoints or states.
- No manuscript aggregate table is used as an input to claim recomputation.
