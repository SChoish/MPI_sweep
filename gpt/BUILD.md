# Build and Validation

Run from this directory:

```bash
./build_pdf.sh
```

Optional overrides:

```bash
PDFLATEX_BIN=/path/to/pdflatex \
BIBTEX_BIN=/path/to/bibtex \
./build_pdf.sh
```

The current sources are AISTATS sources. They resolve `aistats2026.sty` and its `fancyhdr.sty` dependency from the sibling directory `../aistats26_manuscript/` using `\input@path`. Do not build this directory after separating it from the repository unless those style files are copied alongside the manuscript and the input path is adjusted.

Build order matters because `supplement.tex` uses `xr` and reads `paper.aux`. `build_pdf.sh` compiles `paper.tex` first and then `supplement.tex`, with BibTeX and two final LaTeX passes for each document.

Expected outputs:

```text
gpt_aistats26_manuscript.pdf
gpt_aistats26_supplement.pdf
```

Before treating a compiled PDF as a submission candidate, check:

1. no undefined citations or references remain in `paper.build3.log` and `supplement.build3.log`;
2. the main paper uses the AISTATS title/author block rather than the legacy AAAI title macros;
3. the P0 table states the locked result as `unresolved` rather than as equivalence or two-actor superiority;
4. the P0 provenance limitation (cross-host score-level merge without one checkpoint-bound receipt) remains visible;
5. the P1 target-value result is described as a diagnostic, not a return or critic-accuracy certificate;
6. no legacy `gpt_aaai26_*` output is packaged as the current manuscript.
