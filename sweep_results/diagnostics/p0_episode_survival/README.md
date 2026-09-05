# Episode survival and hop objectives

Return decomposed into episode lifetime vs reward per step, plus
\(S_k(t)=\Pr(L_k\ge t)\) from stored lengths. Hopper T=10 curves use the
ext_csv first90 episode CSV (`0c8c50cd`); they are not a new rollout.

See [SURVIVAL_ANALYSIS.md](SURVIVAL_ANALYSIS.md).

Hopper checkpoints are not on this host. Local tail90 hop ΔL is written by

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
  /home/ext_csh/miniconda3/envs/capo_jax/bin/python -u \
  scripts/diagnostics/run_p0_intermediate_actors.py objectives
PYTHONPATH=/home/ext_csh/MPI_sweep \
  /home/ext_csh/miniconda3/envs/capo_jax/bin/python -u \
  scripts/diagnostics/analyze_p0_episode_survival.py
```
