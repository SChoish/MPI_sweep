# Revision Map

| Audit item | AAAI-26 revision |
|---|---|
| `C_k` did not match implementation | Separate equations for P1, Pk, L1, Lk in main and supplement |
| Linearized first hop not fully scale matched | Renamed action-metric-matched projected linearization |
| Archived evaluation frequency stated as 5,000 | Corrected to 50,000; final score defined as 10-episode mean at 1M |
| `C^2` insufficient for `O(beta^3)` | Added locally Lipschitz Hessian; stated `C^2` gives `o(beta^2)` |
| E/ME/expert inconsistency | Uses E = exact `expert-v2`; exact nine IDs in supplement |
| Ideal proximal analysis overstated | Explicitly states one-Adam-step persistent chain is not certified |
| Two-seed headline lacked uncertainty | Added descriptive task-cluster intervals and collapse-risk nuance |
| Target metric named too strongly | Uses sample-anchored next-state target-action displacement |
| Target and final metrics conflated | States different states, anchors, and Polyak status |
| Route `8/8` lacked selection/sensitivity | States post-hoc selection and final-checkpoint `5/8` |
| Simulator aggregation ambiguous | Uses median of checkpoint medians: `-41.4` vs `1.70e12` |
| K4 four-seed claim risk | Described only as targeted K4 replication, not four-seed cross-K confirmation |
| Stability envelope undefined | Operational definition appears in the introduction |
| `T` could be mistaken for compute | Defined as nominal actor-loss coefficient budget |
| Missing pseudocode/configuration | Added Algorithm 1 and compact main/supplement configuration tables |
| Checklist overstated reproducibility | Uses `Partial` where exact lockfiles/wrappers are unavailable |
| WPO missing | Added 2025 ICML WPO and clarified novelty |
| Earlier GPT seed-confound criticism | Withdrawn; supplement explains marginal randomized-procedure estimand |
