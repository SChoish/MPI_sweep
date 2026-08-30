#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LATEXMK_BIN="${LATEXMK_BIN:-latexmk}"

"$PYTHON_BIN" make_figures.py
"$PYTHON_BIN" audit_pack/verify_audit_pack.py
"$PYTHON_BIN" verify_bundle.py

"$LATEXMK_BIN" -pdf -interaction=nonstopmode -halt-on-error paper.tex
"$LATEXMK_BIN" -pdf -interaction=nonstopmode -halt-on-error supplement.tex
cp paper.pdf bar_gpt_aaai26_manuscript.pdf
cp supplement.pdf bar_gpt_aaai26_supplement.pdf

for log in paper.log supplement.log; do
  if grep -Eq 'Undefined control sequence|Emergency stop|Fatal error' "$log"; then
    echo "blocking TeX error in $log" >&2
    grep -En 'Undefined control sequence|Emergency stop|Fatal error' "$log" >&2
    exit 1
  fi
done

technical_page="$(sed -n 's/.*newlabel{sec:technical_end}{{[^}]*}{\([0-9][0-9]*\)}.*/\1/p' paper.aux | head -n1)"
if [[ -z "$technical_page" || "$technical_page" -gt 7 ]]; then
  echo "technical content must end by AAAI page 7; found '${technical_page:-unknown}'" >&2
  exit 1
fi

if command -v pdffonts >/dev/null 2>&1; then
  if pdffonts bar_gpt_aaai26_manuscript.pdf | awk 'NR>2 && $5=="Type 3" {bad=1} END{exit bad?0:1}'; then
    echo "Type 3 font detected" >&2
    exit 1
  fi
fi

echo "built: $ROOT/bar_gpt_aaai26_manuscript.pdf (technical end page $technical_page)"
echo "built: $ROOT/bar_gpt_aaai26_supplement.pdf"
