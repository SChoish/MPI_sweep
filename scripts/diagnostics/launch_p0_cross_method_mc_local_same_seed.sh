#!/usr/bin/env bash
# CPU Walker T=20 seed 0: local mpi4_norm MART4 vs local same-seed mcep.
# Does not overwrite the published P0 tail90 dump. Does not use GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/sweep_results/diagnostics/p0_cross_method_continuation_mc_local_same_seed"
LOG="$OUT/logs"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python3.12}"
SCRIPT="$ROOT/scripts/diagnostics/run_p0_cross_method_continuation_mc.py"
MART="/home/ext_csv/mpi_sweep_lab/results/mpi4_norm/walker2d-medium-v2_tau20_mpi4_seed0/params_1000000.pkl"
TWO="/home/ext_csv/MPI_sweep/results/mcep_p3/walker2d-medium-v2_tau20_mcep3_seed0/params_1000000.pkl"
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

log="$LOG/walker_t20_s0.log"
pidf="$LOG/walker_t20_s0.pid"
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  taskset -c 200-203 \
  "$PY" -u "$SCRIPT" \
    --mart-checkpoint "$MART" \
    --two-actor-checkpoint "$TWO" \
    --out-dir "$OUT" \
    --shard local_mpi4_norm_vs_mcep3_same_seed \
  </dev/null >"$log" 2>&1 &
pid=$!
disown "$pid" 2>/dev/null || true
echo "$pid" >"$pidf"
echo "cpu pid=$pid log=$log"
