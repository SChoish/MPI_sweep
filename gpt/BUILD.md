# Build and Validation

Run from this directory:

```bash
./build_pdf.sh
```

Optional overrides:

```bash
PYTHON_BIN=/path/to/python3 \
PDFLATEX_BIN=/path/to/pdflatex \
SWEEP_RESULTS_DIR=/path/to/sweep_results \
./build_pdf.sh
```

The script fails on:

- a modified or unexpected `aaai2026.sty` hash;
- forbidden AAAI packages such as `hyperref`, `geometry`, or `balance`;
- stale claims such as a 5,000-update archived evaluation frequency;
- `medium-expert` / `-ME` dataset labels;
- missing implementation-specific normalization definitions;
- disagreement with complete score matrices or compact audit outputs;
- unresolved references, undefined control sequences, overfull boxes, or fatal LaTeX errors;
- technical content extending beyond AAAI page 7;
- Type 3 fonts in the generated PDFs.

Generated outputs:

```text
gpt_aaai26_manuscript.pdf
gpt_aaai26_supplement.pdf
```

Intermediate `.aux`, `.log`, and temporary PDFs are ignored by Git.
