#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PDFLATEX_BIN="${PDFLATEX_BIN:-pdflatex}"
RESULTS_DIR="${SWEEP_RESULTS_DIR:-$ROOT/../sweep_results}"
STYLE_SRC="$ROOT/../aistats26_manuscript/aistats2026.sty"
FANCY_SRC="$ROOT/../aistats26_manuscript/fancyhdr.sty"

cd "$ROOT"

# Materialize the official AISTATS style inside this directory before building
# so the packaged gpt/ bundle is standalone after one build.
if [[ ! -f aistats2026.sty ]]; then
  cp "$STYLE_SRC" aistats2026.sty
fi
if [[ ! -f fancyhdr.sty && -f "$FANCY_SRC" ]]; then
  cp "$FANCY_SRC" fancyhdr.sty
fi

if [[ -f verify_bundle.py ]]; then
  "$PYTHON_BIN" verify_bundle.py --results-dir "$RESULTS_DIR" --prebuild
fi
if [[ -f make_figures.py ]]; then
  "$PYTHON_BIN" make_figures.py
fi

# paper.aux is intentionally retained for supplement cross-references.
"$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error paper.tex > paper.build1.log
bibtex paper >/dev/null 2>&1 || true
"$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error paper.tex > paper.build2.log
"$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error paper.tex > paper.build3.log

"$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error supplement.tex > supplement.build1.log
bibtex supplement >/dev/null 2>&1 || true
"$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error supplement.tex > supplement.build2.log
"$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error supplement.tex > supplement.build3.log

cp paper.pdf gpt_aistats26_manuscript.pdf
cp supplement.pdf gpt_aistats26_supplement.pdf

if [[ -f package_bundle.py ]]; then
  "$PYTHON_BIN" package_bundle.py
fi
if [[ -f verify_bundle.py ]]; then
  "$PYTHON_BIN" verify_bundle.py --results-dir "$RESULTS_DIR" --postbuild
fi

echo "built: $ROOT/gpt_aistats26_manuscript.pdf"
echo "built: $ROOT/gpt_aistats26_supplement.pdf"
echo "AISTATS style materialized locally: $ROOT/aistats2026.sty"
