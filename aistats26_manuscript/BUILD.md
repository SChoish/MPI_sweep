# Building the AISTATS 2026 Submission

Run the canonical entry point from this directory:

```bash
./build_pdf.sh
```

It regenerates figures from released CSV data, verifies the manuscript-facing
numbers and the released P0/P1/P2 compact audits, builds the anonymous main paper with the official AISTATS 2026
checklist, builds the single-column technical supplement, and rejects blocking
TeX warnings or a conclusion that spills beyond technical page 8.

## Requirements

- Python 3 with NumPy and Matplotlib
- Tectonic available on `PATH`

Executables can be selected without changing the script:

```bash
PYTHON_BIN=/path/to/python \
TECTONIC_BIN=/path/to/tectonic \
./build_pdf.sh
```

## Results directory

The default input is the repository's sibling `../sweep_results`. For another
checkout layout, pass the canonical results directory explicitly:

```bash
SWEEP_RESULTS_DIR=/path/to/sweep_results ./build_pdf.sh
```

The figure script contains no fallback result arrays:

```bash
python3 make_figures.py --results-dir /path/to/sweep_results
```

## Outputs

A successful build refreshes:

```text
part_aistats26_manuscript.pdf
part_aistats26_supplement.pdf
```

Intermediate `paper.pdf` and `supplement.pdf` files are retained for debugging.

## Independent checks

```bash
python3 ../scripts/verify_release_results.py \
  --results-dir ../sweep_results \
  --manuscript-dir . \
  --require-p2-compact
python3 ../scripts/diagnostics/verify_p0_score_merge.py
rg 'newlabel\{sec:conclusion\}' paper.aux
rg 'Overfull|Undefined control sequence|Emergency stop|Fatal error' \
  paper.log supplement.log
```

Expected results:

- all numerical and compact-audit checks pass (P0 remains explicitly
  `NOT_ADMISSIBLE` despite passing compact integrity);
- `sec:conclusion` is on page 8 or earlier;
- the warning scan returns no output;
- `paper.tex` uses the official unmodified `aistats2026.sty` submission style;
- the main PDF orders technical content, references, and the official checklist;
- the supplementary PDF is single-column as required by the paper pack.
