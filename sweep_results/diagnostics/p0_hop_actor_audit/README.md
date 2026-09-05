# P0 hop-actor audit (ext_csv)

CPU-only diagnostic of MART actors 1–4 and the two-actor target/deployment
policies on existing P0 K=4 checkpoints. No training. Archived training
`d4rl_score` / `d4rl_pi4` / `d4rl_eval` values are not mixed into these tables.

## Coverage

This host has the first90 shard only (90/180 runs, 45 paired cells). Walker and
halfcheetah-expert 1M checkpoints remain on `ext_csh` and were **not** replaced
from other sweeps.

## Reproduce

```bash
cd /home/ext_csv/MPI_sweep
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
PY=/home/ext_csv/miniconda3/envs/offrl/bin/python3.12

$PY scripts/diagnostics/inventory_p0_hop_actors.py
$PY -u scripts/diagnostics/eval_p0_hop_actors.py --workers 4 --cpu0 96 --episodes 10
$PY scripts/diagnostics/diagnose_p0_hop_actions.py --workers 4 --cpu0 96
$PY scripts/diagnostics/analyze_p0_hop_actors.py
$PY scripts/diagnostics/diagnose_p0_hop_objective.py --T 10
```

`--smoke` evaluates hopper-medium T=4 seed=0 only. The evaluator resumes from
`episode_scores.csv`.

## Outputs

| File | Role |
| --- | --- |
| `INVENTORY.json` / `inventory.csv` | checkpoint paths, hashes, actors, missing reasons |
| `COVERAGE.json` | 180-run denominator vs local evalable/paired counts |
| `episode_scores.csv` | one row per episode × actor |
| `cell_means.csv` | 10-episode means (not extra training seeds) |
| `return_decomposition.csv` | length, timeout count, pooled reward/step |
| `survival_hopper_T10.csv` | `S_k(t)` at t=100,200,…,1000 |
| `hop_objective.csv` | frozen-critic JKO `ΔL_k` on dataset states (Hopper T=10) |
| `action_diagnostics.csv` | common-state action distances |
| `ANALYSIS.md` | short report |
| `fig_*.png` | profiles, heatmap, scatter, Hopper T=10 survival, return split, hop ΔL vs ΔJ |
