# offrl host diagnostics

Generated on `offrl` from local release checkpoints in `results/mpi4_norm`.

| Job | Status |
| --- | --- |
| matched geometry (`mpi4`, frontier) | done → [`matched_geometry/`](matched_geometry/) |
| target-policy exposure (`mpi4`, 9 env × τ∈{4,7,10,14,20} × seed 0–3 = 180) | done → [`target_policy_exposure/`](target_policy_exposure/) |
| actor-path semigroup (available 1M) | done → [`actor_path_semigroup/`](actor_path_semigroup/) |
| `mpi4_norm` import checksums (504×1M) | done → [`mpi4_norm_import/`](mpi4_norm_import/) |
| frozen critic / route-shadow K=4 | not on offrl (ext_csv) |

`mpi4_norm` 1M: **504/504**.
