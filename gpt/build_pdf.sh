#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDFLATEX_BIN="${PDFLATEX_BIN:-pdflatex}"
BIBTEX_BIN="${BIBTEX_BIN:-bibtex}"

cd "$ROOT"

build_one() {
  local doc="$1"
  "$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error "$doc.tex" >"$doc.build1.log"
  "$BIBTEX_BIN" "$doc" >"$doc.bibtex.log" 2>&1 || true
  "$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error "$doc.tex" >"$doc.build2.log"
  "$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error "$doc.tex" >"$doc.build3.log"
}

build_one paper
build_one supplement

cp paper.pdf gpt_aistats26_manuscript.pdf
cp supplement.pdf gpt_aistats26_supplement.pdf

echo "built: $ROOT/gpt_aistats26_manuscript.pdf"
echo "built: $ROOT/gpt_aistats26_supplement.pdf"
echo "note: aistats2026.sty and fancyhdr.sty are resolved from ../aistats26_manuscript via input@path"
