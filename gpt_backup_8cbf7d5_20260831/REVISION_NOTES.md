# Revision Map

| Audit item | AAAI-26 revision |
|---|---|
| `C_k` did not match implementation | Separate equations for P1, Pk, L1, Lk in main and supplement |
| Linearized first hop not fully scale matched | Renamed action-metric-matched projected linearization |
| Intermediate cadence overgeneralized | Reports run-family-specific cadence; all headline scores use the common 10-episode final-1M checkpoint |
| `C^2` insufficient for `O(beta^3)` | Added locally Lipschitz Hessian; stated `C^2` gives `o(beta^2)` |
| E/ME/expert inconsistency | Uses E = exact `expert-v2`; exact nine IDs in supplement |
| Ideal proximal analysis overstated | Explicitly states one-Adam-step persistent chain is not certified |
| Proximal grid expanded to four seeds | Reports the exact equally weighted `T>=4` aggregate, both disjoint host-separated depth replications, and task-resampling intervals |
| Target metric named too strongly | Uses sample-anchored next-state target-action displacement |
| Target and final metrics conflated | States different states, anchors, and Polyak status; scopes the 270-checkpoint mechanism audit to TD3+BC/P2/P3 |
| Route `8/8` lacked selection/sensitivity | States post-hoc selection and final-checkpoint `5/8` |
| Simulator aggregation ambiguous | Uses median of checkpoint medians: `-41.4` vs `1.70e12` |
| Newer seed-pair provenance incomplete | Uses the full four-seed score grid but discloses that exact newer-pair run manifests/code hashes are unavailable |
| Stability envelope undefined | Operational definition appears in the introduction |
| `T` mapping was ambiguous | Defines `T=tau=alpha/2`, exact `K=1` TD3+BC mapping, and `h=T/K`; distinguishes it from compute and realized distance |
| Missing pseudocode/configuration | Added Algorithm 1 and compact main/supplement configuration tables |
| Checklist was abbreviated and overstated reproducibility | Inputs all 31 official questions; uses `Partial`/`No` where exact lockfiles, manifests, hardware records, or wrappers are unavailable |
| WPO missing | Added 2025 ICML WPO and clarified novelty |
| Earlier GPT seed-confound criticism | Withdrawn; supplement explains marginal randomized-procedure estimand |
