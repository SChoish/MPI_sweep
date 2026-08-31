# offrl host diagnostics

Generated on `offrl` from local release checkpoints in `results/mpi4_norm`.

| Job | Status |
| --- | --- |
| matched geometry (`mpi4`, taus 4/7/10, frontier envs) | done (own-critic only; no local `results_qnorm` for canonical TD3@τ=1) |
| target-policy exposure (`mpi4`, τ∈{4,7,10,14,20}, seeds 0–3, 9 env = 180) | done → [`target_policy_exposure/`](target_policy_exposure/) |
| actor-path semigroup (`mpi1`/`mpi2` unnorm + `mpi4`/`mpi8` norm, available 1M only) | done → [`actor_path_semigroup/`](actor_path_semigroup/) |
| frozen critic / route shadow | skipped on offrl (route-shadow K=4 runs on **ext_csv**) |

`mpi4_norm` 1M matrix is complete (504/504), including walker frontier seeds 2–3.
