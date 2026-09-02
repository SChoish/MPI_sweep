# BAR — AISTATS 2026 Submission Package

This directory contains the anonymous submission package for:

> **BAR: Budgeted Actor Refinement Decouples Bootstrap Exposure from Policy Reach**

Budgeted Actor Refinement (BAR) splits a nominal actor-improvement coefficient
across a persistent TD3+BC actor chain. The paper follows one regime-based
argument: ideal JKO steps explain the common small-$T$ flow; as TD3+BC leaves
its high-return plateau, BAR limits the target-routed first hop while retaining
final-policy reach; at extreme $T$, all methods are nonlocal and multi-hop
shifts rather than removes the collapse boundary. At $T=4,7$, TD3+BC scores
45.93/30.36 while BAR-P4 remains at 82.06/81.79. Across four proximal seeds,
the equally weighted high-budget aggregate is strictly ordered by depth, and
two disjoint host-separated seed pairs reproduce that ordering. Targeted
controls probe routing, optimizer-count, target-lag, and critic-calibration. A
90-pair MCEP-inspired two-actor control shows that sequential re-centering is
not a prerequisite for the observed P3 mean and seed-mean collapse outcomes.
As a bundled procedure comparison, it does not isolate the chain's incremental
value. A completed P4 cross-host score sensitivity is numerically unresolved
but fails the locked same-runtime-stack gate, so a protocol-admissible P4
comparison remains open.

## Files

- `paper.tex`: eight-page-limit main manuscript; inputs the checklist after references
- `supplement.tex`: single-column technical supplement
- `reproducibility_checklist.tex`: official AISTATS 2026 questions with answers
- `references.bib`: bibliography
- `aistats2026.sty`, `fancyhdr.sty`: unmodified official paper-pack files
- `make_figures.py`: regenerates empirical figures from released CSV files
- `build_pdf.sh`: canonical build and validation entry point

Experiment-inclusion decisions and the ReLU activation-region follow-up are
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
- A 270-checkpoint TD3/P2/P3 target-action-displacement audit, a separate
  180-checkpoint four-seed P4 target-branch audit, and full same-next-state
  target/final geometry for 90 matched TD3/P3/P4 cells
- A matched-common-critic target-value perturbation audit: P3/P4 lower than
  TD3+BC in 86/90 and 88/90 cells, with all nine task medians below one
- A restricted four-environment seed-2/3 P3/P2 target-action-displacement
  sensitivity reproducing the direction in 36/40 cells
- Route shadows, compute-matched refinement, actor-target lag, simulator
  screening, and a clean-source six-task ReLU-region residence audit
- A 90-pair, two-seed MCEP-inspired policy-separation control with a
  contemporaneous BAR-P3 rerun and documented CPU-recovery sensitivity
- A completed 90-pair P4 two-actor score sensitivity retained only as
  protocol-inadmissible descriptive evidence because its host stacks differ
- A verified K=1--4 H200/JAX actor-phase profile with hash-pinned raw timing
  and scoped backend-memory records
- Task-resampling uncertainty, dynamics-family sensitivity, collapse-risk sensitivity, and rescued-cell gain attribution
