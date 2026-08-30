# offrl host diagnostics

Generated on `offrl` from local release checkpoints in `results/mpi4_norm`.

| Job | Status |
| --- | --- |
| matched geometry (`mpi4`, taus 4/7/10, frontier envs) | done (own-critic only; no local `results_qnorm` for canonical TD3@τ=1) |
| target-policy exposure (`mpi4`) | done |
| frozen critic / route shadow | skipped (no `results_qnorm` / `results_route_shadow` on this host) |

Seeds 2–3 are incomplete for walker frontier cells; those rows are omitted as missing.
