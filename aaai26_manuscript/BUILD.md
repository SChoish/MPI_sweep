# Building the AAAI-26 Submission

Run the canonical entry point from this directory:

```bash
./build_pdf.sh
```

It regenerates figures from CSV data, builds the main paper with its official reproducibility checklist, builds the technical supplement, and rejects blocking TeX warnings or a main conclusion that spills beyond technical page 7.

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

The default input is the repository's sibling `../sweep_results`. For any other checkout layout, pass the canonical results directory explicitly:

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
tdrr_aaai26_manuscript.pdf
tdrr_aaai26_supplement.pdf
```

The intermediate `paper.pdf`, `supplement.pdf`, and `reproducibility_checklist.pdf` files are retained for debugging.

## Independent verification

```bash
rg 'newlabel\{sec:conclusion\}' paper.aux
rg 'Overfull|Undefined control sequence|Emergency stop|Fatal error' \
  paper.log supplement.log reproducibility_checklist.log
```

Expected results:

- `sec:conclusion` is on page 7 or earlier.
- The warning scan returns no output.
- The main source uses the official `aaai2026` submission style.
- Tables use captions below the tabular material and at least 9-point text.
- The main PDF contains the checklist after the references.
- The main manuscript is self-contained; the optional technical supplement is not required to understand its central claims.

Underfull-box and font-substitution notices from Tectonic are non-blocking when the checks above pass.
