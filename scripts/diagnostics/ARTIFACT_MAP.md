# Artifact map: `sweep_results/diagnostics/`

What is already in the release tree, what this package can regenerate from
**local checkpoints**, and what needs a different pipeline.

Legend:

- **ckpt**: post-hoc from finished `params_1000000.pkl` (+ eval)
- **special-ckpt**: needs a specific results tree (route-shadow / qnorm cells)
- **retrain**: must launch control training, not dump-only
- **aggregate**: derived from score tables / multiple dumps
- **archive**: frozen manuscript snapshot; not the host-dump schema
- **benchmark**: fresh measurement on one locked hardware/software stack

“Partial” means the packaged script can regenerate the core measurements, but
not necessarily the archive's curated summary, exact filenames, or frozen
selection of runs.

| Path under `sweep_results/diagnostics/` | Kind | Regenerable with this package? | Script / note |
| --- | --- | --- | --- |
| `hosts/<host>/matched_geometry/` | ckpt | **yes** (new layout) | `dump_mpi_frontier_geometry.py` → `hop_geometry.csv`, `run_diagnostics.csv` + summarize |
| `hosts/<host>/target_policy_exposure/` | ckpt | **yes** | `dump_target_policy_exposure.py` → `target_policy_exposure_summary.csv`, `MANIFEST.json` |
| `hosts/<host>/frozen_critic_small_step/` | special-ckpt | **yes** | `run_frozen_critic_small_step.py` |
| `hosts/<host>/route_shadow/` | special-ckpt | **yes** | `run_route_shadow_mc.py` → `checkpoint_pairs.csv`, `run_level.csv`, `SUMMARY.json` |
| `target_policy_exposure/` (top-level archive) | archive | partial | Same dumper; archive was a curated 270-run snapshot with SUMMARY |
| `matched_geometry/prox2_*`, `lin2_*`, `td3_prox3_*` | archive | **no (different schema)** | Older matched-geometry pipeline / renamed outputs; host dumps use `hop_geometry.csv` |
| `route_shadow/` (top-level archive) | archive | partial | Use `run_route_shadow_mc.py` on `results_route_shadow` |
| `route_shadow_k4/` | archive | partial | Compact copy of `audit_report_20260901/route_shadow_mc_k4`; excluded from manuscript |
| `bar_mcep_p3_paired/` | aggregate/archive | partial | `audit_bar_mcep_p3_paired.py` regenerates `AUDIT.json` only; the paired CSV, summary, and source manifest are frozen and verifier-gated |
| `p0_bar_p4_vs_two_actor_p4/` | retrain + aggregate | **180-cell score integrity only**; numeric `unresolved`; **P0-inadmissible** | `verify_p0_score_merge.py` independently checks the compact merge; host dependency stacks differ, five optimizer states are nonfinite, and no full ckpt-bound verify exists |
| `p0_intermediate_actors/` | ckpt | **yes** (local tail90 only) | `run_p0_intermediate_actors.py` + `analyze_p0_intermediate_actors.py`; all-actor env eval, action distances, frozen-critic hop ΔL. Does not train. Head90 stays missing. See `P0_INTERMEDIATE_ACTORS_RUNBOOK.md` |
| `p0_episode_survival/` | aggregate | **yes** from stored episode CSVs | `analyze_p0_episode_survival.py`; Hopper T=10 survival from first90 checkpoints; tail90 hop-objective join from a different checkpoint shard. Do not pool across shards. |
| `p0_hop_continuation_mc/` | ckpt | **yes** (one tail90 cell) | `run_p0_hop_continuation_mc.py`; Walker-medium T=20 seed 0 hops 2–3, first action then Polyak μ1 |
| `p0_cross_method_continuation_mc/` | ckpt | **yes** (same tail90 pair) | `run_p0_cross_method_continuation_mc.py`; MART μ2/μ3 vs two-actor deployment, 2×2 first-action × Polyak continuation |
| `p0_frozen_hop_extra_opt/` | ckpt + extra-opt | **yes** on a host with first90 Hopper 1M MART ckpts | `run_p0_frozen_hop_extra_opt.py`; freeze critic + μ_{k-1}, extra-optimize hop actor 2–4. Hopper-medium/expert T=10 seeds {0,1}. Hash-pinned; no mpi1/exp3 substitution |
| `p0_hop_actor_audit/` | archive + ckpt CPU re-eval | **yes** on hosts with local P0 1M ckpts | First90 Hopper eval, survival, and hop ΔL live here (`0c8c50cd` / `ab0f233`). Regenerators: `inventory_p0_hop_actors.py`, `eval_p0_hop_actors.py`, `diagnose_p0_hop_actions.py`, `analyze_p0_hop_actors.py`, `diagnose_p0_hop_objective.py`. Do not splice other sweeps. |
| `p1_target_value_audit/` | archive | **artifact-arithmetic verified**; raw NPZ external | Common-critic paired ratios and full geometry are promoted; outlier-dominated absolute means and comparator residual are not headline evidence |
| `actor_cost/` | benchmark/archive | **verified measured snapshot**; rerun needs local data/GPU | Hash-pinned K=1--4 H200 profile plus profiler and independent verifier; see `ACTOR_COST_RUNBOOK.md` |
| `frozen_critic_small_step/` (top-level archive) | archive | partial | Use `run_frozen_critic_small_step.py` |
| `fixed_operator_order/` | archive | retired preflight scaffold; **no learned result** | Frozen 0/18 snapshot retained as verifier-gated provenance; superseded by `P2_RELU_RESIDENCE_RUNBOOK.md` |
| `p2_relu_residence_final/` | archive | **clean-source final-v2 verified**; raw NPZ stay external | Hash-pinned 12-run Hopper/Walker cell tables + task/task-equal/category summaries; full bundle `/home/ext_csv/mpi_sweep_lab/p2-relu-final-v2-06d64b3/` |
| `actor_path_semigroup/` (top-level archive) | archive | **yes** | `run_actor_semigroup.py`; full local `mpi1/2/3/exp3` seeds 2/3 grid (incl. tau 0.7 and 12); provenance only, not manuscript error evidence |
| `simulator_calibration/` | archive | **no in this package** | Exploratory screening; not wired here |
| `control_final_scores.csv`, `control_summary.csv` | retrain | **no** | From compute-matched / target-lag **training** (`launch_causal_controls.py` in lab) |
| `compute_matched_MANIFEST.json`, `target_lag_MANIFEST.json` | retrain | **no** | Launch manifests for those controls |
| `environment_t_sensitivity.csv` | aggregate | **no** | Built from return tables, not raw ckpt dumps |
| `i234_first_endpoint/` | aggregate (eval.csv) + CPU ckpt re-eval | **yes** on the owning host | `recover_i234_first_endpoint.py --host {s23,ext_csv}` for eval.csv tables; `eval_i4_historical_first_endpoint.py` for historical I4 first/endpoint CPU re-eval; s01 and s23 stay in separate tables |

Raw checkpoint derivatives, common batches, per-state arrays, and benchmark work
files are not release artifacts and must not be staged merely because a run
completed.

## Recommended workflow

1. Each host runs `run_host.sh --job ckpt` for its seeds.
2. Push `hosts/<host>/` only. The host `.gitignore` allowlists CSV/JSON/MD;
   per-state NPZ/NPY data and logs remain local.
3. Keep top-level archive folders as the manuscript-frozen copies unless you
   explicitly regenerate and replace them with a reviewed merge.

## Optional extras in this folder

| Script | Use |
| --- | --- |
| `dump_td3bc_geometry.py` | TD3+BC-only geometry grid |
| `dump_explicit_effective_step.py` | Explicit effective-step audit |
| `dump_checkpoint_dynamics.py` | Checkpoint dynamics dump |
| `mc_value_audit.py` | MC value audit |
| `compare_mpi_geometry_sets.py` | Compare two geometry dump directories |

These are not in `run_host.sh --job ckpt`; call them manually when needed.

## Per-run / per-sweep provenance (new runs)

New training runs write `PROVENANCE.json` next to `config.json`, and sweeps
write `SWEEP_PROVENANCE.json` into the log-dir (git revision + dirty flag,
source/dataset/normalization digests, package versions, host/accelerator
inventory, final-checkpoint hash). See [`../PROVENANCE.md`](../PROVENANCE.md)
for the full contract. Historical runs are not backfilled.
