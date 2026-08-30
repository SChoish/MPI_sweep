#!/usr/bin/env bash
set -euo pipefail

MANUSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="${PYTHON_BIN:-python3}"

if [[ -n "${SWEEP_RESULTS_DIR:-}" ]]; then
  RESULTS_DIR="$SWEEP_RESULTS_DIR"
elif [[ -d "$MANUSCRIPT_DIR/../sweep_results/diagnostics" ]]; then
  RESULTS_DIR="$MANUSCRIPT_DIR/../sweep_results"
else
  echo "error: sweep_results not found; set SWEEP_RESULTS_DIR" >&2
  exit 1
fi

if [[ -n "${TECTONIC_BIN:-}" ]]; then
  TECTONIC_CMD="$TECTONIC_BIN"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC_CMD="$(command -v tectonic)"
else
  echo "error: tectonic not found; set TECTONIC_BIN or add it to PATH" >&2
  exit 1
fi

cd "$MANUSCRIPT_DIR"
"$PYTHON_CMD" make_figures.py --results-dir "$RESULTS_DIR"

"$TECTONIC_CMD" paper.tex --keep-logs --keep-intermediates
"$TECTONIC_CMD" supplement.tex --keep-logs --keep-intermediates
"$TECTONIC_CMD" reproducibility_checklist.tex --keep-logs --keep-intermediates

cp paper.pdf tdrr_aaai26_manuscript.pdf
cp supplement.pdf tdrr_aaai26_supplement.pdf

for log_file in paper.log supplement.log reproducibility_checklist.log; do
  if grep -Eq 'Overfull \\[hv]box|Undefined control sequence|Emergency stop|Fatal error' "$log_file"; then
    echo "error: $log_file contains a blocking typesetting warning" >&2
    grep -En 'Overfull \\[hv]box|Undefined control sequence|Emergency stop|Fatal error' "$log_file" >&2
    exit 1
  fi
done

conclusion_page="$(sed -n 's/.*newlabel{sec:conclusion}{{[^}]*}{\([0-9][0-9]*\)}.*/\1/p' paper.aux | head -n 1)"
if [[ -z "$conclusion_page" || "$conclusion_page" -gt 7 ]]; then
  echo "error: conclusion must appear by technical page 7; found '${conclusion_page:-unknown}'" >&2
  exit 1
fi

echo "built: $MANUSCRIPT_DIR/tdrr_aaai26_manuscript.pdf"
echo "built: $MANUSCRIPT_DIR/tdrr_aaai26_supplement.pdf"
