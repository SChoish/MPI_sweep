# ext_csh host diagnostics

Post-hoc dumps generated from local P2/P3 checkpoints for four
stability-frontier environments, five budgets, and seeds 2--3.

| Job | Scope |
| --- | --- |
| `matched_geometry/` | 80 P2/P3 runs with own-critic and common-reference geometry |
| `target_policy_exposure/` | 80 P2/P3 target-branch checkpoints under the direct next-state protocol |

The matched-geometry manifest is authoritative about the reference critic:
it uses the environment- and seed-matched TD3+BC \(T=.05\) checkpoint. The
`canonical_td3_tau1` value retained in the CSV `critic_scope` column is a
legacy schema label, not the actual coefficient.

Own-critic run-total gain is positive in all 80 runs. Under the \(T=.05\)
reference critic it is positive in all 45 stable runs and negative in all 35
collapsed runs. This is a restricted-environment, post-hoc sensitivity audit,
not an independent implementation or a simulator-ground-truth evaluation.

The target-exposure dump measures the target branch and its current-state
first-branch proxy. P3 has lower deterministic target exposure than P2 in
36/40 matched cells (median ratio `.905`). This is a restricted, post-hoc
sensitivity check. Final-actor displacement, where used, comes from
`matched_geometry/run_diagnostics.csv`, not from
`current_sample_anchor_proxy_*`.
