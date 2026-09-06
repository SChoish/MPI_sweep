#!/usr/bin/env bash
# CPU-only first-action × Polyak continuation. Resumes episodes.csv; adds MART μ4.
# Does not use GPU. Does not stop AMO.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/sweep_results/diagnostics/p0_first_action_continuation"
LOG="$OUT/logs"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python3.12}"
SCRIPT="$ROOT/scripts/diagnostics/run_p0_first_action_continuation.py"
TAG="${1:-mu4_resume}"
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

log="$LOG/${TAG}.log"
pidf="$LOG/${TAG}.pid"
cd "$ROOT"
nohup env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  taskset -c 200-203 \
  "$PY" -u "$SCRIPT" \
    --T 10 \
    --seeds 0 1 \
    --episodes 10 \
    --workers 4 \
    --cpu0 200 \
  </dev/null >"$log" 2>&1 &
pid=$!
disown "$pid" 2>/dev/null || true
echo "$pid" >"$pidf"
echo "cpu pid=$pid tag=$TAG log=$log"
