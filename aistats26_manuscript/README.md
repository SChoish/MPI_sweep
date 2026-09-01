# BAR — AISTATS 2026 Submission Package

This directory contains the anonymous submission package for:

> **BAR: Budgeted Actor Refinement Decouples Bootstrap Exposure from Policy Reach**

Budgeted Actor Refinement (BAR) splits a nominal actor-improvement coefficient
across a persistent TD3+BC actor chain. The first branch supplies Bellman-target
actions and the final branch is deployed, structurally separating bootstrap
exposure from policy reach. Across four proximal seeds, the equally weighted
high-budget aggregate is strictly ordered by depth, and two disjoint
host-separated seed pairs reproduce that ordering. The four-seed
projected-linearized grid also reproduces the `K=2` to `K=3` high-budget
gain, raising the mean from 38.43 to 46.44. Targeted controls probe competing
routing, optimizer-count, target-lag, and critic-calibration explanations. A
90-pair MCEP-inspired two-actor control shows that sequential re-centering is
not a prerequisite for the observed mean and seed-mean collapse outcomes on
this grid. As a bundled procedure comparison, it does not isolate which
simpler-procedure element is responsible.

## Files

- `paper.tex`: eight-page-limit main manuscript; inputs the checklist after references
- `supplement.tex`: single-column technical supplement
- `reproducibility_checklist.tex`: official AISTATS 2026 questions with answers
- `references.bib`: bibliography
- `aistats2026.sty`, `fancyhdr.sty`: unmodified official paper-pack files
- `make_figures.py`: regenerates empirical figures from released CSV files
- `build_pdf.sh`: canonical build and validation entry point

Experiment-inclusion decisions and the controlled operator-error rerun are
tracked in [`../TODO.md`](../TODO.md).

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
- Complete nine-task, fourteen-budget, four-seed action-metric-matched projected-linearized grid for `K={2,3}`
- Proximal full-depth replication across disjoint host-separated seed pairs `{0,1}` and `{2,3}`
- Linearized `K=2` to `K=3` improvement reproduced in both disjoint seed pairs
- A 270-checkpoint TD3/P2/P3 exposure audit plus a separate 180-checkpoint
  four-seed P4 target-branch audit
- A restricted four-environment seed-2/3 P3/P2 exposure sensitivity reproducing
  the direction in 36/40 cells
- A 24-cell P4 final-policy frontier panel, kept distinct from the full target-branch audit
- Route shadows, compute-matched refinement, actor-target lag, simulator
  screening, and a frozen-critic local-consistency audit
- A 90-pair, two-seed MCEP-inspired policy-separation control with a
  contemporaneous BAR-P3 rerun and documented CPU-recovery sensitivity
- Task-resampling uncertainty, dynamics-family sensitivity, collapse-risk sensitivity, and rescued-cell gain attribution

