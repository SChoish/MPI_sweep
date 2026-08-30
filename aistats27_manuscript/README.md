# Multi-Step Proximal Policy Improvement — AISTATS 2027 Manuscript Package

This package contains an anonymous, submission-style manuscript built from the MPI paper and the matched fixed-budget analyses completed through 2026-08-30.

## Files

- `paper.tex`: main manuscript
- `sections/`: section sources, proofs, appendix, and manual bibliography
- `figures/`: vector PDF figures used by the manuscript
- `make_figures.py`: script that regenerates the figures from the audited aggregate values
- `aistats2027_provisional.sty`: provisional AISTATS-like layout
- `mpi_aistats27_manuscript.pdf`: compiled manuscript

## Build

```bash
./build_pdf.sh
```

See [`BUILD.md`](BUILD.md) for the exact current-workstation command,
first-time Tectonic installation, output path, and verification checks. The
bibliography is embedded in `sections/references.tex`; BibTeX is not used.

## Formatting status

The official AISTATS 2027 paper pack was not publicly available when this package was prepared. The manuscript therefore uses a provisional US-letter, two-column AISTATS-like style and keeps the main text within the current eight-page AISTATS submission convention. Replace `aistats2027_provisional.sty` with the official 2027 style once released and recheck pagination.

## Empirical scope represented in the manuscript

- Five-seed plug-in extension results for IQL, ReBRAC, and TD3+BC.
- Complete nine-environment, fourteen-budget fixed-budget results for TD3+BC and proximal and scale-matched projected linearized MPI with two and three total actor hops.
- Matched geometry and failure diagnostics for the two-hop proximal and linearized variants.
- Direct next-state Polyak-target exposure for 270 audited checkpoints and an exploratory 60-checkpoint CPU simulator-reference screening.
- Paired route-only shadow critics, compute-matched refinement controls, actor-target-lag controls, and a frozen-critic action-level small-step audit.
- Four transferred per-seed scores from a separate-machine targeted Hopper-medium proximal `K=4` run. They characterize the nonmonotone stability window but remain outside the complete-grid aggregate because raw evaluation files, configurations, and checkpoints have not been transferred.
