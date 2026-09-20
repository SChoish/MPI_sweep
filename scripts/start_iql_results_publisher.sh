#!/usr/bin/env bash
# Install a score publisher without pulling code into the active training checkout.
set -euo pipefail
PROFILE=${1:?Usage: start_iql_results_publisher.sh ext_csv|ext_csh /path/to/MPI_sweep}
REPO=${2:?Pass the existing MPI_sweep checkout}
case "$PROFILE" in ext_csv|ext_csh) ;; *) echo 'Unknown machine profile' >&2; exit 2 ;; esac
REPO=$(git -C "$REPO" rev-parse --show-toplevel)
STATE=$(git -C "$REPO" rev-parse --absolute-git-dir)/score-publisher-$PROFILE
mkdir -p "$STATE"
SCRIPT=$STATE/publish_iql_results.py
if [[ -f "$STATE/pid" ]]; then
  PID=$(cat "$STATE/pid")
  if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    COMMAND=$(ps -p "$PID" -o args=)
    if [[ "$COMMAND" == *"$SCRIPT"* ]]; then
      echo "Publisher already running (pid $PID); log: $STATE/publisher.log"
      exit 0
    fi
  fi
fi
git -C "$REPO" fetch origin main
git -C "$REPO" show origin/main:scripts/publish_iql_results.py > "$SCRIPT.tmp"
mv "$SCRIPT.tmp" "$SCRIPT"
# First verify and publish once. A missing source/authentication stops installation.
python3 "$SCRIPT" --profile "$PROFILE" --repo "$REPO" --push
nohup python3 -u "$SCRIPT" --profile "$PROFILE" --repo "$REPO" --push --watch \
  >> "$STATE/publisher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$STATE/pid"
sleep 1
kill -0 "$PID"
echo "Score publisher started (pid $PID), every five minutes; log: $STATE/publisher.log"
