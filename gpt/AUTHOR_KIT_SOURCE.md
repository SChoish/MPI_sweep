# AISTATS-26 Author-Kit Provenance

The current GPT manuscript uses the same AISTATS-26 LaTeX style files already tracked by the canonical sibling bundle:

- `../aistats26_manuscript/aistats2026.sty`
- `../aistats26_manuscript/fancyhdr.sty`

`paper.tex` and `supplement.tex` add `../aistats26_manuscript/` to LaTeX's input path before loading `aistats2026`. The local legacy `aaai2026.sty` is not used by the current build.

The main paper follows the canonical anonymous AISTATS title block used in `../aistats26_manuscript/paper.tex`:

```tex
\twocolumn[
\aistatstitle{...}
\aistatsauthor{Anonymous Authors}
\aistatsaddress{Anonymous Institution}
]
```

Bibliography entries are kept in `references.bib` and rendered with `apalike` through Natbib. The reproducibility checklist is included after the bibliography by `\input{ReproducibilityChecklist}`.

This directory is intentionally repository-relative rather than a standalone author-kit mirror. For a standalone submission archive, copy the AISTATS style dependencies into the archive and remove or adjust the parent-directory input path before packaging.
