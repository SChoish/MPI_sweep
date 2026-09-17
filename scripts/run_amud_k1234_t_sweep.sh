#!/usr/bin/env bash
# Sequential K=1..4 MPI/BAR T-sweep on antmaze-umaze-diverse-v2.
# Default grid: tau∈MPI_TAU_GRID (14), seeds 0–3, eval@1M × 100 episodes.
set -uo pipefail

ROOT=/home/choi/MPI_sweep
PY="$ROOT/.venv/bin/python"
LOGDIR=/home/choi/logs
OUT="$ROOT/results/antmaze_umaze_diverse_t_sweep"
LOG="$LOGDIR/mpi_amud_k1234_t_sweep.log"

mkdir -p "$LOGDIR" "$OUT" "$ROOT/logs/antmaze_umaze_diverse"
exec >>"$LOG" 2>&1

log() { printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON=(
  --domains antmaze
  --datasets umaze-diverse
  --seeds "0 1 2 3"
  --gpus 0
  --slots-per-gpu 2
  --method bar
  --integrator implicit
  --eval-episodes 100
  --eval-freq 1000000
  --max-timesteps 1000000
  --data-dir "$ROOT/data"
  --save-dir "$OUT"
  --log-dir "$ROOT/logs/antmaze_umaze_diverse"
  --python "$PY"
  --q-scale-norm
)

log "start antmaze-umaze-diverse T sweep K=1..4 out=$OUT"

for K in 1 2 3 4; do
  log "=== hops=K=$K dry-run count ==="
  "$PY" "$ROOT/launch_mpi_sweep.py" --hops "$K" "${COMMON[@]}" --dry-run \
    | tee "$ROOT/logs/antmaze_umaze_diverse/dryrun_K${K}.txt" \
    | tail -3
  n=$("$PY" "$ROOT/launch_mpi_sweep.py" --hops "$K" "${COMMON[@]}" --dry-run 2>/dev/null | rg -c '^\[' || true)
  log "K=$K pending_lines≈$n — launching"
  "$PY" "$ROOT/launch_mpi_sweep.py" --hops "$K" "${COMMON[@]}"
  rc=$?
  log "K=$K finished rc=$rc"
  if [[ $rc -ne 0 ]]; then
    log "ABORT after K=$K failure"
    exit "$rc"
  fi
done

log "all K=1..4 complete"
