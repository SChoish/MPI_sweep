# TDRR — AAAI-26 Submission Package

This directory contains the anonymous submission package for:

> **Target–Deployment Routed Refinement: Broadening the Stability Envelope of Offline TD3+BC**

Target–Deployment Routed Refinement (TDRR) is the paper's staged TD3+BC actor architecture. The empirical claim is deliberately limited to TD3+BC: the main evidence is a complete fixed-budget sweep, direct target-policy measurements, and targeted routing and compute controls.

## Files

- `paper.tex`: main manuscript source; it includes the official checklist at build time
- `supplement.tex`: separate technical supplementary source
- `reproducibility_checklist.tex`: official AAAI-26 checklist, included after the references in the main PDF
- `references.bib`: bibliography used by the main paper and supplement
- `aaai2026.sty`, `aaai2026.bst`: official AAAI-26 style files
- `make_figures.py`: regenerates every empirical figure from released CSV files
- `figures/`: generated vector and raster figures
- `build_pdf.sh`: canonical build and validation entry point

The build produces:

- `tdrr_aaai26_manuscript.pdf` (technical paper, references, and checklist)
- `tdrr_aaai26_supplement.pdf`

## Build

From this directory:

```bash
./build_pdf.sh
```

The script uses the sibling `../sweep_results` directory by default. For another layout:

```bash
SWEEP_RESULTS_DIR=/path/to/sweep_results ./build_pdf.sh
```

Set `PYTHON_BIN` or `TECTONIC_BIN` only when those executables are not available as `python3` and `tectonic` on `PATH`. See [BUILD.md](BUILD.md) for verification details.

## Formatting status

The package uses the official AAAI-26 author-kit style (`aaai2026.sty`, template version 2026.1). The main manuscript has seven pages of technical content followed by references and the required reproducibility checklist. Proofs and additional analyses are in a separate supplementary PDF.

## Empirical scope

- Complete nine-task, fourteen-budget TD3+BC sweep with two paired seeds per cell
- Two- and three-hop proximal and scale-matched projected-linearized TDRR variants
- Direct next-state Polyak-target exposure and current-state final-policy displacement diagnostics
- Paired route-only shadow critics, compute-matched refinement controls, actor-target-lag controls, and a frozen-critic small-step audit
- A separate-machine, four-seed Hopper-medium four-hop study reported only as a targeted supplementary analysis; score CSVs are released, while matched checkpoints and configuration manifests are unavailable
