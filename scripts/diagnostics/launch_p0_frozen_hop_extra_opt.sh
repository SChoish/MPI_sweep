#!/usr/bin/env bash
# One extra-opt process per GPU. Does not pack 12 jobs.
# GPU 0: hopper-medium seed 0 hops 2-3-4
# GPU 1: hopper-expert seed 0 hops 2-3-4
set -uo pipefail
ROOT=/home/ext_csh/MPI_sweep
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
OUT=$ROOT/sweep_results/diagnostics/p0_frozen_hop_extra_opt
CKPT_ROOT=$OUT/source_ckpts
RUNNER=$ROOT/scripts/diagnostics/run_p0_frozen_hop_extra_opt.py
mkdir -p "$OUT/logs" "$CKPT_ROOT"

need=(
  "$CKPT_ROOT/hopper-medium-v2_tau10_mpi4_seed0/params_1000000.pkl"
  "$CKPT_ROOT/hopper-expert-v2_tau10_mpi4_seed0/params_1000000.pkl"
)
missing=0
for path in "${need[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing $path" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "refusing to launch: first90 Hopper 1M ckpts are not on this host" >&2
  exit 2
fi

launch_one() {
  local gpu=$1 task=$2 seed=$3
  local log=$OUT/logs/${task}_seed${seed}_gpu${gpu}.log
  local pidfile=$OUT/logs/${task}_seed${seed}_gpu${gpu}.pid
  unset JAX_PLATFORMS JAX_PLATFORM_NAME
  CUDA_VISIBLE_DEVICES=$gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  nohup "$PY" -u "$RUNNER" \
    --gpu "$gpu" \
    --tasks "$task" \
    --seeds "$seed" \
    --hops 2 3 4 \
    --data-dir "$ROOT/data" \
    --out-dir "$OUT" \
    --ckpt-root "$CKPT_ROOT" \
    >"$log" 2>&1 &
  echo $! >"$pidfile"
  echo "launched gpu=$gpu task=$task seed=$seed pid=$(cat "$pidfile") log=$log"
}

launch_one 0 hopper-medium-v2 0
launch_one 1 hopper-expert-v2 0
