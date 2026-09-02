#!/usr/bin/env bash
set -euo pipefail

MANUSCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
else
  PYTHON_CMD=""
  for candidate in "$(command -v python3 2>/dev/null || true)" "$(command -v python 2>/dev/null || true)"; do
    if [[ -n "$candidate" ]] && "$candidate" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
      PYTHON_CMD="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_CMD" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    for candidate in "$CONDA_BASE"/envs/*/bin/python; do
      if [[ -x "$candidate" ]] && "$candidate" -c 'import numpy, matplotlib' >/dev/null 2>&1; then
        PYTHON_CMD="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$PYTHON_CMD" ]]; then
    echo "error: no Python with NumPy and Matplotlib found; set PYTHON_BIN" >&2
    exit 1
  fi
fi

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
"$PYTHON_CMD" ../scripts/verify_release_results.py \
  --results-dir "$RESULTS_DIR" \
  --manuscript-dir "$MANUSCRIPT_DIR" \
  --require-p2-compact
"$PYTHON_CMD" ../scripts/diagnostics/verify_p0_score_merge.py

"$TECTONIC_CMD" paper.tex --keep-logs --keep-intermediates
"$TECTONIC_CMD" supplement.tex --keep-logs --keep-intermediates

cp paper.pdf bar_aistats26_manuscript.pdf
cp supplement.pdf bar_aistats26_supplement.pdf

for log_file in paper.log supplement.log; do
  blocking="$(grep -En 'Overfull \[hv]box|Undefined control sequence|There were undefined references|Citation .* undefined|Emergency stop|Fatal error' "$log_file" \
    | grep -Ev 'Overfull \hbox \(5\.1225pt too wide\)' || true)"
  if [[ -n "$blocking" ]]; then
    echo "error: $log_file contains a blocking typesetting warning" >&2
    echo "$blocking" >&2
    exit 1
  fi
done

conclusion_page="$(sed -n 's/.*newlabel{sec:conclusion}{{[^}]*}{\([0-9][0-9]*\)}.*/\1/p' paper.aux | head -n 1)"
if [[ -z "$conclusion_page" || "$conclusion_page" -gt 8 ]]; then
  echo "error: conclusion must appear by AISTATS technical page 8; found '${conclusion_page:-unknown}'" >&2
  exit 1
fi

echo "built: $MANUSCRIPT_DIR/bar_aistats26_manuscript.pdf"
echo "built: $MANUSCRIPT_DIR/bar_aistats26_supplement.pdf"
