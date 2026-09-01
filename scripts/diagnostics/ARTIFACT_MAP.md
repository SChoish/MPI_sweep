# Artifact map: `sweep_results/diagnostics/`

What is already in the release tree, what this package can regenerate from
**local checkpoints**, and what needs a different pipeline.

Legend:

- **ckpt**: post-hoc from finished `params_1000000.pkl` (+ eval)
- **special-ckpt**: needs a specific results tree (route-shadow / qnorm cells)
- **retrain**: must launch control training, not dump-only
- **aggregate**: derived from score tables / multiple dumps
- **archive**: frozen manuscript snapshot; not the host-dump schema

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
| `frozen_critic_small_step/` (top-level archive) | archive | partial | Use `run_frozen_critic_small_step.py` |
| `actor_path_semigroup/` (top-level archive) | archive | **yes** | `run_actor_semigroup.py`; full local `mpi1/2/3/exp3` seeds 2/3 grid (incl. tau 0.7 and 12); provenance only, not manuscript error evidence |
| `simulator_calibration/` | archive | **no in this package** | Exploratory screening; not wired here |
| `control_final_scores.csv`, `control_summary.csv` | retrain | **no** | From compute-matched / target-lag **training** (`launch_causal_controls.py` in lab) |
| `compute_matched_MANIFEST.json`, `target_lag_MANIFEST.json` | retrain | **no** | Launch manifests for those controls |
| `environment_t_sensitivity.csv` | aggregate | **no** | Built from return tables, not raw ckpt dumps |

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
