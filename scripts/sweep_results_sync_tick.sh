#!/usr/bin/env bash
# One tick: pull → write seed2/3 CSVs → run diagnostics → commit+push sweep_results only.
# No --autostash: leftover dump-script edits caused unresolved conflicts (2026-08-30).
set -euo pipefail

ROOT=/home/ext_csh/MPI_sweep
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
KEY=/home/ext_csh/.ssh/deploy_key_ext_csh_20260829
export GIT_SSH_COMMAND="ssh -i ${KEY} -o IdentitiesOnly=yes"
cd "$ROOT"

TS=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')
echo "=== sweep_results sync+diag @ ${TS} ==="

# Keep local operational edits (append-mode worker logs) off the rebase path.
LAUNCH_BAK=""
if ! git diff --quiet -- launch_mpi_sweep.py; then
  LAUNCH_BAK=$(mktemp /tmp/launch_mpi_sweep.XXXXXX.py)
  cp -a launch_mpi_sweep.py "$LAUNCH_BAK"
  git checkout -- launch_mpi_sweep.py
  echo "[sync] parked dirty launch_mpi_sweep.py for pull"
fi

set +e
git pull --rebase origin main
pull_rc=$?
set -e
if [[ -n "$LAUNCH_BAK" ]]; then
  cp -a "$LAUNCH_BAK" launch_mpi_sweep.py
  rm -f "$LAUNCH_BAK"
  echo "[sync] restored launch_mpi_sweep.py local patch"
fi
if [[ $pull_rc -ne 0 ]]; then
  echo "fatal: git pull --rebase failed rc=$pull_rc"
  exit "$pull_rc"
fi

"$PY" "$ROOT/scripts/write_sweep_results_csv.py"

# Diagnostics may fail independently; still push CSV progress if any.
set +e
bash "$ROOT/scripts/run_ext_csh_diagnostics_ckpt.sh"
diag_rc=$?
set -e
echo "[diag] rc=$diag_rc"

# Stage only sweep_results (matrices + hosts/ext_csh publishable files).
git add sweep_results/
if git diff --cached --quiet; then
  echo "no sweep_results changes; skip commit/push"
  exit 0
fi

export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-ext_csh}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-ext_csh@users.noreply.github.com}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

git commit -m "$(cat <<EOF
Update ext_csh sweep_results matrices and diagnostics.

EOF
)"

git push origin HEAD:main
echo "pushed ok"
