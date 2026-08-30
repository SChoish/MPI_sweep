#!/usr/bin/env bash
# Run post-hoc checkpoint diagnostics on THIS machine's local result trees.
#
# Requires a lab-format train_td3bc.py on PYTHONPATH (actor_params / actor2_params
# checkpoints). Point LAB_ROOT at your mpi_sweep_lab (or equivalent) checkout.
#
# Example (seeds owned by this host):
#   LAB_ROOT=/home/YOU/mpi_sweep_lab \
#   RESULTS_ROOT=/home/YOU/mpi_sweep_lab \
#   DATA_DIR=/raid/YOU/datasets/d4rl \
#   bash scripts/run_host_diagnostics.sh \
#     --host offrl --seeds 2 3 --methods mpi2 mpi3 expl2 expl3
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
LAB_ROOT=${LAB_ROOT:-$REPO/../mpi_sweep_lab}
RESULTS_ROOT=${RESULTS_ROOT:-$LAB_ROOT}
DATA_DIR=${DATA_DIR:-/raid/${USER}/datasets/d4rl}
HOST=${HOST:-$(hostname -s)}
OUT_ROOT=${OUT_ROOT:-$REPO/sweep_results/diagnostics/hosts/$HOST}
PY=${PY:-python3}

SEEDS=(0 1)
METHODS=(mpi2 mpi3 expl2 expl3)
TAUS=(4 7 10 14 20)
JOB=all

usage() {
  sed -n '2,16p' "$0"
  echo "Flags: --host NAME --seeds N... --methods M... --taus T... --job geometry|exposure|all"
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
    *) echo "unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -f "$LAB_ROOT/train_td3bc.py" ]]; then
  echo "LAB_ROOT missing train_td3bc.py: $LAB_ROOT"
  echo "Set LAB_ROOT to the checkout that produced your checkpoints."
  exit 1
fi

export PYTHONPATH="$LAB_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"
echo "[host-diag] host=$HOST lab=$LAB_ROOT results=$RESULTS_ROOT out=$OUT_ROOT"
echo "[host-diag] seeds=${SEEDS[*]} methods=${METHODS[*]} taus=${TAUS[*]} job=$JOB"

run_geometry() {
  "$PY" -u "$REPO/scripts/dump_mpi_frontier_geometry.py" \
    --root "$RESULTS_ROOT" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_ROOT/matched_geometry" \
    --seeds "${SEEDS[@]}" \
    --methods "${METHODS[@]}" \
    --taus "${TAUS[@]}"
  "$PY" -u "$REPO/scripts/summarize_mpi_frontier_geometry.py" \
    "$OUT_ROOT/matched_geometry"
}

run_exposure() {
  "$PY" -u "$REPO/scripts/dump_target_policy_exposure.py" \
    --root "$RESULTS_ROOT" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_ROOT/target_policy_exposure" \
    --seeds "${SEEDS[@]}" \
    --methods "${METHODS[@]}" \
    --taus "${TAUS[@]}"
}

case "$JOB" in
  geometry) run_geometry ;;
  exposure) run_exposure ;;
  all) run_geometry; run_exposure ;;
  *) echo "bad --job $JOB"; exit 1 ;;
esac

echo "[host-diag] done → $OUT_ROOT"
echo "[host-diag] commit CSVs under sweep_results/diagnostics/hosts/$HOST and push when ready"
