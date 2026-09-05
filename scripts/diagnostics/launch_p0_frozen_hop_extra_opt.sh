#!/usr/bin/env bash
# One frozen extra-opt process per GPU. Does not stop AMO.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/sweep_results/diagnostics/p0_frozen_hop_extra_opt"
SRC="$OUT/source_ckpts"
LOG="$OUT/logs"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python3.12}"
SCRIPT="$ROOT/scripts/diagnostics/run_p0_frozen_hop_extra_opt.py"

MEDIUM_SRC="/home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94/runs/bar_p4/hopper-medium-v2_tau10_mpi4_seed0/params_1000000.pkl"
EXPERT_SRC="/home/ext_csv/mpi_sweep_lab/p0_bar_two_actor_p4_29fea94_cont2_gpu_d923/runs/bar_p4/hopper-expert-v2_tau10_mpi4_seed0/params_1000000.pkl"
MEDIUM_DST="$SRC/hopper-medium-v2_tau10_mpi4_seed0/params_1000000.pkl"
EXPERT_DST="$SRC/hopper-expert-v2_tau10_mpi4_seed0/params_1000000.pkl"

stage() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ ! -f "$src" ]]; then
    echo "missing source checkpoint: $src" >&2
    exit 1
  fi
  if [[ ! -f "$dest" ]]; then
    cp -a "$src" "$dest"
  fi
}

stage "$MEDIUM_SRC" "$MEDIUM_DST"
stage "$EXPERT_SRC" "$EXPERT_DST"
mkdir -p "$LOG"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0"
export D4RL_SUPPRESS_IMPORT_ERROR=1

launch_one() {
  local gpu="$1" cpus="$2" task="$3" tag="$4"
  local log="$LOG/${tag}.log"
  local pidf="$LOG/${tag}.pid"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    taskset -c "$cpus" \
    "$PY" -u "$SCRIPT" \
      --device cuda \
      --tasks "$task" \
      --seeds 0 \
      --T 10 \
      --extra-steps 10000 \
      --skip-report \
      --out-csv "$OUT/extra_opt_curves.${tag}.csv" \
    </dev/null >"$log" 2>&1 &
  local pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" >"$pidf"
  echo "gpu=$gpu pid=$pid tag=$tag log=$log"
}

cd "$ROOT"
launch_one 0 200-207 hopper-medium-v2 medium_s0
sleep 5
launch_one 1 208-215 hopper-expert-v2 expert_s0
echo "AMO left running. extra-opt: one process on GPU 0, one on GPU 1."
