#!/usr/bin/env bash
set -euo pipefail

MANUSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${TECTONIC_BIN:-}" ]]; then
  TECTONIC_CMD="$TECTONIC_BIN"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC_CMD="$(command -v tectonic)"
elif [[ -x "${HOME}/.local/bin/tectonic" ]]; then
  TECTONIC_CMD="${HOME}/.local/bin/tectonic"
else
  echo "error: tectonic not found; see BUILD.md" >&2
  exit 1
fi

cd "$MANUSCRIPT_DIR"
"$PYTHON_BIN" make_figures.py
"$TECTONIC_CMD" paper.tex --keep-logs --keep-intermediates
cp paper.pdf mpi_aistats27_manuscript.pdf

if grep -Eq 'Overfull \\[hv]box|Undefined control sequence|Emergency stop|Fatal error' paper.log; then
  echo "error: paper.log contains a blocking typesetting warning" >&2
  grep -En 'Overfull \\[hv]box|Undefined control sequence|Emergency stop|Fatal error' paper.log >&2
  exit 1
fi

echo "built: $MANUSCRIPT_DIR/mpi_aistats27_manuscript.pdf"
