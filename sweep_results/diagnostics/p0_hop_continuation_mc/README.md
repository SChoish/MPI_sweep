# Continuation-matched MC (case B)

Walker-medium T=20 seed 0, MART hops 2 and 3: first action from the hop
actor, remaining steps from Polyak μ1. No training.

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=/home/ext_csh/MPI_sweep \
  /home/ext_csh/miniconda3/envs/capo_jax/bin/python -u \
  scripts/diagnostics/run_p0_hop_continuation_mc.py
```

See [ANALYSIS.md](ANALYSIS.md).
