#!/usr/bin/env bash
# Run post-hoc diagnostics on THIS machine's local checkpoints.
# See README.md and HOST_RUNBOOK.md in this directory.
set -euo pipefail

PKG=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$PKG/../.." && pwd)
LAB_ROOT=${LAB_ROOT:-$REPO/../mpi_sweep_lab}
RESULTS_ROOT=${RESULTS_ROOT:-$LAB_ROOT}
DATA_DIR=${DATA_DIR:-/raid/${USER}/datasets/d4rl}
HOST=${HOST:-$(hostname -s)}
OUT_ROOT=${OUT_ROOT:-$REPO/sweep_results/diagnostics/hosts/$HOST}
PY=${PY:-python3}

SEEDS=(0 1)
METHODS=(mpi2 mpi3 expl2 expl3)
TAUS=(4 7 10 14 20)
JOB=ckpt   # ckpt | geometry | exposure | frozen | route | all

usage() {
  cat <<EOF
Usage: LAB_ROOT=... RESULTS_ROOT=... DATA_DIR=... bash run_host.sh [flags]

Flags:
  --host NAME
  --seeds N [N...]
  --methods M [M...]     td3 mpi2 mpi3 expl2 expl3
  --taus T [T...]
  --job ckpt|geometry|exposure|frozen|route|all
  --lab-root DIR         tree with train_td3bc.py matching ckpt format
  --results-root DIR     contains results_* run folders
  --data-dir DIR
  --out-root DIR

Default --job ckpt = geometry + exposure (pure checkpoint dumps).
frozen/route need extra paths; see HOST_RUNBOOK.md.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST=$2; OUT_ROOT=$REPO/sweep_results/diagnostics/hosts/$HOST; shift 2 ;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do SEEDS+=("$1"); shift; done ;;
    --methods) shift; METHODS=(); while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do METHODS+=("$1"); shift; done ;;
    --taus) shift; TAUS=(); while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do TAUS+=("$1"); shift; done ;;
    --job) JOB=$2; shift 2 ;;
    --lab-root) LAB_ROOT=$2; shift 2 ;;
    --results-root) RESULTS_ROOT=$2; shift 2 ;;
    --data-dir) DATA_DIR=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -f "$LAB_ROOT/train_td3bc.py" ]]; then
  echo "LAB_ROOT has no train_td3bc.py: $LAB_ROOT"
  exit 1
fi

export PYTHONPATH="$LAB_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
echo "[diag] host=$HOST job=$JOB"
echo "[diag] lab=$LAB_ROOT results=$RESULTS_ROOT out=$OUT_ROOT"
echo "[diag] seeds=${SEEDS[*]} methods=${METHODS[*]} taus=${TAUS[*]}"

run_geometry() {
  mkdir -p "$OUT_ROOT/matched_geometry"
  "$PY" -u "$PKG/dump_mpi_frontier_geometry.py" \
    --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
    --out-dir "$OUT_ROOT/matched_geometry" \
    --seeds "${SEEDS[@]}" --methods "${METHODS[@]}" --taus "${TAUS[@]}"
  "$PY" -u "$PKG/summarize_mpi_frontier_geometry.py" "$OUT_ROOT/matched_geometry"
}

run_exposure() {
  mkdir -p "$OUT_ROOT/target_policy_exposure"
  "$PY" -u "$PKG/dump_target_policy_exposure.py" \
    --root "$RESULTS_ROOT" --data-dir "$DATA_DIR" \
    --out-dir "$OUT_ROOT/target_policy_exposure" \
    --seeds "${SEEDS[@]}" --methods "${METHODS[@]}" --taus "${TAUS[@]}"
}

run_frozen() {
  mkdir -p "$OUT_ROOT/frozen_critic_small_step"
  local seed_str="${SEEDS[*]}"
  "$PY" -u "$PKG/run_frozen_critic_small_step.py" \
    --results-dir "${FROZEN_RESULTS_DIR:-$RESULTS_ROOT/results_qnorm}" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_ROOT/frozen_critic_small_step" \
    --seeds "$seed_str"
}

run_route() {
  mkdir -p "$OUT_ROOT/route_shadow"
  local seed_str="${SEEDS[*]}"
  "$PY" -u "$PKG/run_route_shadow_mc.py" \
    --results-dir "${ROUTE_RESULTS_DIR:-$RESULTS_ROOT/results_route_shadow}" \
    --out-dir "$OUT_ROOT/route_shadow" \
    --seeds "$seed_str"
}

case "$JOB" in
  geometry) run_geometry ;;
  exposure) run_exposure ;;
  frozen) run_frozen ;;
  route) run_route ;;
  ckpt) run_geometry; run_exposure ;;
  all) run_geometry; run_exposure; run_frozen; run_route ;;
  *) echo "bad --job $JOB"; exit 1 ;;
esac

echo "[diag] done → $OUT_ROOT"
echo "[diag] publishable: CSV/JSON/MD only (raw NPZ/NPY and logs stay local)"
echo "[diag] next: git add sweep_results/diagnostics/hosts/$HOST && git status --short"
