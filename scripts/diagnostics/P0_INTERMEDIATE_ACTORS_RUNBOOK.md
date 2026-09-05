# P0 intermediate-actor evaluation (CPU, no training)

Diagnose why MART4 and two-actor-P4 can finish close in return by scoring
**every stored actor** on the frozen P0 K=4 grid and measuring action movement
on shared states. Do not launch training. Do not splice other sweeps.

`eval_i4_historical_first_endpoint.py` is not in this tree; first/endpoint
recovery remains `recover_i234_first_endpoint.py` / `recover_p0_tail90_first_actor.py`.
This protocol re-evaluates all six comparison policies and does **not** reuse
those archived scores.

## Interpreter

Use the tail-shard JAX/Flax stack (Python 3.10, JAX 0.4.38, Flax 0.10.4):

```bash
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
cd /home/ext_csh/MPI_sweep
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
```

Default `--workers 4`. Do not raise this while a GPU trainer is live.

## Commands

```bash
"$PY" -u scripts/diagnostics/run_p0_intermediate_actors.py inventory
"$PY" -u scripts/diagnostics/run_p0_intermediate_actors.py eval --pilot
"$PY" -u scripts/diagnostics/run_p0_intermediate_actors.py eval --workers 4
"$PY" -u scripts/diagnostics/run_p0_intermediate_actors.py actions
"$PY" -u scripts/diagnostics/run_p0_intermediate_actors.py objectives
"$PY" -u scripts/diagnostics/analyze_p0_intermediate_actors.py
PYTHONPATH=/home/ext_csh/MPI_sweep "$PY" -u scripts/diagnostics/analyze_p0_episode_survival.py
PYTHONPATH=/home/ext_csh/MPI_sweep "$PY" -u scripts/diagnostics/run_p0_hop_continuation_mc.py
PYTHONPATH=/home/ext_csh/MPI_sweep "$PY" -u scripts/diagnostics/run_p0_cross_method_continuation_mc.py
```

`all` runs inventory, eval, then actions (not objectives). Eval and actions resume from existing
per-policy JSONL / NPZ. Objectives need the dataset-state NPZ from `actions`.
Survival curves use stored lengths only; Hopper T=10 plots come from
`p0_hop_actor_audit/episode_scores.csv`.

## Protocol

- Grid: nine D4RL tasks, `T={4,7,10,14,20}`, seeds `{0,1}`, MART4 and two_actor_p4.
- Local checkpoints: frozen tail90 `CHECKPOINTS.json` only.
- Missing head90 cells stay missing (ext_csv). No `results/mpi4*` substitution.
- 10 episodes, seeds `1000..1009`, shared across compared policies of a task.
  This dump is the tail90 checkpoint shard only. First90 Hopper checkpoints
  are not on this host; do not splice the two shards.
- Gymnasium v4 eval envs from `train_td3bc.EVAL_ENV`; per-checkpoint mean/std.
- Dataset states: 2048 HDF5 transitions, seed `20260905`, per task.
- Rollout common set B: 256 states from MART μ4 plus 256 from two-actor deploy.
- RMS = `sqrt(mean_action_dim((a_next-a_prev)^2))`. Near-zero hops excluded from cosine.

## Outputs

`sweep_results/diagnostics/p0_intermediate_actors/`

| File | Role |
| --- | --- |
| `INVENTORY.json`, `COVERAGE.json` | Checkpoint presence vs 180-run plan |
| `episodes.csv` | Episode-level returns |
| `action_distances.csv`, `hop_q_scale.csv` | Common-state action diagnostics |
| `hop_objectives.csv` | Frozen-critic hop ΔL on dataset and rollout states |
| `ANALYSIS.md`, `figures/` | Profiles, heatmap, scatter |
| `raw/` | Gitignored NPZ (dataset + rollout states) |

Survival / return decomposition: `sweep_results/diagnostics/p0_episode_survival/`
(`hopper_t10_survival.png`, `hop_objective_vs_return.csv`, `SURVIVAL_ANALYSIS.md`).
