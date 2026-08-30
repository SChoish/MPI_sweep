# BAR — AISTATS 2026 Submission Package

This directory contains the anonymous submission package for:

> **BAR: Budgeted Actor Refinement Broadens the Stability Envelope of Offline TD3+BC**

Budgeted Actor Refinement (BAR) splits a fixed nominal actor-improvement budget
across a persistent TD3+BC actor chain. The first branch supplies target-policy
actions and the final branch is evaluated. The empirical claim remains at the
implemented-bundle level: the results do not causally isolate budget splitting,
routing, re-centering, or compute.

## Files

- `paper.tex`: eight-page-limit main manuscript; inputs the checklist after references
- `supplement.tex`: single-column technical supplement
- `reproducibility_checklist.tex`: official AISTATS 2026 questions with answers
- `references.bib`: bibliography
- `aistats2026.sty`, `fancyhdr.sty`: unmodified official paper-pack files
- `make_figures.py`: regenerates empirical figures from released CSV files
- `build_pdf.sh`: canonical build and validation entry point

The build produces:

- `bar_aistats26_manuscript.pdf`
- `bar_aistats26_supplement.pdf`

## Build

```bash
./build_pdf.sh
```

The script uses `../sweep_results` by default. Set `SWEEP_RESULTS_DIR`,
`PYTHON_BIN`, or `TECTONIC_BIN` only when the checkout layout or executable
locations differ. See [BUILD.md](BUILD.md) for all checks.

## Formatting basis

The package uses the official AISTATS 2026 paper pack. The anonymous submission
limit is eight technical pages, excluding references and the reproducibility
checklist. The optional supplement uses the required single-column format.

## Empirical scope

- Complete nine-task, fourteen-budget, two-seed TD3+BC phase diagram
- Proximal BAR variants with `K={2,3,4}` and projected-linearized variants with `K={2,3}`
- Independent `K=1` replication on seeds 2–3
- Direct next-state Polyak-target exposure and final-policy displacement diagnostics
- Route-only shadows, compute-matched refinement, actor-target lag, simulator-reference screening, and a frozen-critic small-step audit
- Targeted four-seed Hopper-medium `K=4` score summary

