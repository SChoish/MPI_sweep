#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
python3 make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
cp paper.pdf mpi_gpt_aistats27_manuscript.pdf
if grep -Eq 'Overfull \\[hv]box|Undefined control sequence|Emergency stop|Fatal error' paper.log; then
  echo "blocking TeX warning detected" >&2
  grep -En 'Overfull \\[hv]box|Undefined control sequence|Emergency stop|Fatal error' paper.log >&2
  exit 1
fi
printf 'built: %s\n' "$DIR/mpi_gpt_aistats27_manuscript.pdf"
