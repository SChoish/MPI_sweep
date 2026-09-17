#!/usr/bin/env bash
# choi: every-10m refresh+push of antmaze-umaze-diverse sweep_results to origin/main.
set -euo pipefail

ROOT=/home/choi/MPI_sweep
PY="$ROOT/.venv/bin/python"
LOG=/home/choi/logs/mpi_amud_auto_push.log
LOCK=/home/choi/logs/mpi_amud_auto_push.lock

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
exec 9>"$LOCK"
flock -n 9 || { echo "[$(TZ=Asia/Seoul date '+%F %T %Z')] SKIP: already running"; exit 0; }

TS=$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')
echo "=== amud auto_push @ ${TS} ==="
cd "$ROOT"

# Park local antmaze publish artifacts so pull --rebase can run cleanly.
PARK=""
if ! git diff --quiet -- sweep_results/antmaze-umaze-diverse 2>/dev/null \
  || [[ -n "$(git ls-files --others --exclude-standard sweep_results/antmaze-umaze-diverse)" ]]; then
  PARK=$(mktemp -d /tmp/amud_sweep_park.XXXXXX)
  mkdir -p "$PARK"
  cp -a sweep_results/antmaze-umaze-diverse "$PARK/" 2>/dev/null || true
  git checkout -- sweep_results/antmaze-umaze-diverse 2>/dev/null || true
  # leave untracked alone; publish will rewrite anyway
  echo "[sync] parked dirty antmaze-umaze-diverse for pull"
fi

set +e
git fetch origin main
git pull --rebase origin main
pull_rc=$?
set -e
if [[ $pull_rc -ne 0 ]]; then
  echo "fatal: pull --rebase failed rc=$pull_rc"
  exit "$pull_rc"
fi

"$PY" "$ROOT/scripts/publish_amud_sweep_results.py"

git add sweep_results/antmaze-umaze-diverse/
if git diff --cached --quiet; then
  echo "no antmaze-umaze-diverse changes; skip commit/push"
  [[ -n "$PARK" ]] && rm -rf "$PARK"
  exit 0
fi

git commit -m "$(cat <<EOF
Update choi antmaze-umaze-diverse T-sweep results.

EOF
)"

git push origin HEAD:main
echo "pushed ok $(git rev-parse --short HEAD)"
[[ -n "$PARK" ]] && rm -rf "$PARK"
