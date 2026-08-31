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
- stale blanket cadence claims (including universal 5,000- or 50,000-update wording);
- `medium-expert` / `-ME` dataset labels;
- missing implementation-specific normalization definitions or the exact `T=tau=alpha/2` mapping;
- a missing source-results directory or disagreement with the four-seed proximal, two-seed linearized, or compact-audit outputs;
- unresolved references or citations, undefined control sequences, overfull boxes, or fatal LaTeX errors;
- technical content extending beyond AAAI page 7 or a checklist that does not follow the technical content;
- a missing/incomplete official 31-question reproducibility checklist;
- a missing, corrupt, incomplete, path-unsafe, or non-anonymous bundle ZIP;
- missing Poppler inspection tools or Type 3 fonts in the generated PDFs.

Generated outputs:

```text
gpt_aaai26_manuscript.pdf
gpt_aaai26_supplement.pdf
gpt_aaai26_bundle.zip
```

Intermediate `.aux`, `.log`, and temporary PDFs are ignored by Git.
