# BAR — AISTATS 2026 Submission Package

This directory contains the anonymous submission package for:

> **BAR: Budgeted Actor Refinement Decouples Bootstrap Exposure from Policy Reach**

Budgeted Actor Refinement (BAR) splits a nominal actor-improvement coefficient
across a persistent TD3+BC actor chain. The first branch supplies Bellman-target
actions and the final branch is deployed, structurally separating bootstrap
exposure from policy reach. Across four proximal seeds, the equally weighted
high-budget aggregate is strictly ordered by depth, and two disjoint
host-separated seed pairs reproduce that ordering. Targeted controls narrow the
mechanism without claiming a component-level causal decomposition.

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

- Complete nine-task, fourteen-budget, four-seed proximal grid for `K={1,2,3,4}`
- Complete two-seed action-metric-matched projected-linearized grid for `K={2,3}`
- Full-depth replication across disjoint host-separated seed pairs `{0,1}` and `{2,3}`
- A 270-checkpoint P3 target-action exposure audit, explicitly separate from P4
- Route shadows, compute-matched refinement, actor-target lag, simulator screening, and a frozen-critic local audit
- Task-resampling uncertainty, collapse-risk sensitivity, and rescued-cell gain attribution

