# GPT AISTATS-26 Manuscript Bundle

This directory is the GPT-authored AISTATS bundle for the current BAR study. It has been rewritten against the evidence available on `main` through 2026-09-02, including the completed four-seed phase diagrams, P3 and P4 direct policy-separation controls, the P1 target-value audit, the learned-ReLU local-scope audit, and the measured actor-cost snapshot.

## Current scientific framing

The manuscript no longer treats sequential re-centering as the established mechanism. The strongest result is **target--deployment separation**:

- BAR's bundled depth intervention strongly broadens the TD3+BC large-budget stability envelope.
- P3 target/action audits show that the critic-facing branch can contract without typical final-policy contraction.
- A directly data-anchored two-actor P3 control reproduces the aggregate P3 regime.
- The locked P4 comparison is unresolved: BAR minus two-actor has task-equal mean `-2.31`, 95% task-resampling interval `[-7.25, 2.41]`, paired median `+0.68`, and 55/90 BAR wins.
- Therefore the paper presents BAR as a strong depth-indexed routed procedure and probe, while acknowledging that the sequential chain has not shown an incremental return advantage over direct separation.

The broad target/evaluation-policy separation principle is not claimed as novel; MCEP is discussed explicitly. BAR's contribution is the depth intervention, stability phase diagram, direct movement/value diagnostics, and controlled comparison.

## Canonical source files

- `paper.tex` — AISTATS main manuscript.
- `supplement.tex` — proofs, full score table, P0/P1/P2/P3 details, and provenance notes.
- `ReproducibilityChecklist.tex` — checklist included by the main paper.
- `references.bib` — AISTATS bibliography.
- `build_pdf.sh` — local manuscript/supplement builder.
- `REVISION_NOTES.md` — evidence-driven changes from the previous framing.
- `SOURCE_MANIFEST.md` — scientific source map.

`paper.tex` and `supplement.tex` use the official `aistats2026.sty` and `fancyhdr.sty` from the sibling `../aistats26_manuscript/` directory via LaTeX's input path. The old `aaai2026.sty` file in this directory is legacy and is not used by the current manuscript.

## Build

From the repository root:

```bash
cd gpt
./build_pdf.sh
```

Expected outputs:

```text
gpt_aistats26_manuscript.pdf
gpt_aistats26_supplement.pdf
```

The supplement uses `xr` to read `paper.aux`, so the build script compiles the main paper first.

## Evidence caveats retained in the manuscript

The P0 180-final comparison is complete at score level and was assembled from frozen `ext_csv` and `ext_csh` shards, but it does not have the originally specified single-artifact-tree checkpoint-bound analyze/verify receipt. Some historical proximal seed-2/3 per-run manifests were not retained, and the exact historical Git revision of the earlier P3 two-actor control was never persisted. These limitations are stated rather than reconstructed post hoc.
