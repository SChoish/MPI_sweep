#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PDFLATEX_BIN="${PDFLATEX_BIN:-pdflatex}"
RESULTS_DIR="${SWEEP_RESULTS_DIR:-$ROOT/../sweep_results}"

cd "$ROOT"
"$PYTHON_BIN" verify_bundle.py --results-dir "$RESULTS_DIR" --prebuild
"$PYTHON_BIN" make_figures.py

for source in paper supplement; do
  "$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error "$source.tex" >"$source.build1.log"
  "$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error "$source.tex" >"$source.build2.log"
done

cp paper.pdf gpt_aaai26_manuscript.pdf
cp supplement.pdf gpt_aaai26_supplement.pdf

"$PYTHON_BIN" package_bundle.py
"$PYTHON_BIN" verify_bundle.py --results-dir "$RESULTS_DIR" --postbuild

echo "built: $ROOT/gpt_aaai26_manuscript.pdf"
echo "built: $ROOT/gpt_aaai26_supplement.pdf"
echo "built: $ROOT/gpt_aaai26_bundle.zip"
