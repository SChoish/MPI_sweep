#!/usr/bin/env bash
# Periodic: rewrite Exp/MPI CSVs → push sweep_results → refresh canvas.
# Focus: Exp K=2/3 high-τ {24,28,34,40} on ext_csh.
set -uo pipefail

ROOT=/home/ext_csh/MPI_sweep
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
LOG_DIR=/home/ext_csh/logs/mpi_exp_tau40_periodic
LOG="$LOG_DIR/periodic.log"
PIDFILE="$LOG_DIR/periodic.pid"
INTERVAL="${MPI_EXP_TAU40_PERIOD_SEC:-600}"

mkdir -p "$LOG_DIR"
echo $$ >"$PIDFILE"
cd "$ROOT" || exit 1

# Prefer MPI_sweep deploy key / Host alias.
export GIT_SSH_COMMAND="ssh -i /home/ext_csh/.ssh/mpi_sweep_deploy -o IdentitiesOnly=yes"

tick() {
  local ts
  ts=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')
  echo "=== mpi exp τ40 tick @ ${ts} ==="

  # 1) Refresh seed2/3 matrices (includes high-τ rows when present).
  "$PY" "$ROOT/scripts/write_sweep_results_csv.py" || echo "[warn] write_sweep_results_csv rc=$?"

  # 2) Pull then push sweep_results only (no diagnostics — keep tick short).
  set +e
  git pull --ff-only origin main
  pull_rc=$?
  set -e
  if [[ $pull_rc -ne 0 ]]; then
    echo "[warn] git pull --ff-only failed rc=$pull_rc (continue canvas)"
  else
    git add sweep_results/
    if git diff --cached --quiet; then
      echo "no sweep_results changes; skip commit/push"
    else
      export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-ext_csh}"
      export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-ext_csh@users.noreply.github.com}"
      export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
      export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
      # Do not touch git config; identity via env only.
      git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" \
        commit -m "$(cat <<EOF
Update ext_csh Exp/MPI sweep_results (incl. high-τ).

EOF
)" && git push origin HEAD:main && echo "pushed ok" || echo "[warn] commit/push failed"
    fi
  fi

  # 3) Canvas rewrite + agent wake line.
  "$PY" "$ROOT/scripts/rewrite_mpi_k123_s23_canvas.py" || echo "[warn] canvas rewrite rc=$?"
  echo "AGENT_LOOP_TICK_mpi_exp_tau40 {\"prompt\":\"Refresh mpi-k123-s23 canvas for Exp K=2/3 high-τ {24,28,34,40}. Report exp2/exp3 1M progress, live trains, queue ok/fail, ETA.\"}"
}

echo "==== $(TZ=Asia/Seoul date -Is) loop start pid=$$ interval=${INTERVAL}s ====" | tee -a "$LOG"
while true; do
  tick >>"$LOG" 2>&1
  # Also mirror AGENT_LOOP_TICK to stdout for monitored-shell wake.
  rg 'AGENT_LOOP_TICK_mpi_exp_tau40' "$LOG" | tail -1 || true
  sleep "$INTERVAL"
done
