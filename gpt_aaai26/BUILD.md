# Build and validation

Run:

```bash
./build_pdf.sh
```

The canonical process is:

1. regenerate PDF/PNG figures from `data/`;
2. validate the sanitized audit pack;
3. check manuscript-facing values and stale wording;
4. compile the AAAI-26 main paper and supplement;
5. assert `sec:technical_end` is on page 7 or earlier;
6. reject fatal TeX errors and Type-3 fonts.

Individual checks:

```bash
python3 make_figures.py
python3 audit_pack/verify_audit_pack.py
python3 verify_bundle.py
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The source uses no `hyperref`, `geometry`, `fontenc`, or other packages prohibited by the AAAI style. Bibliography entries are included directly in `sections/references.tex`, so BibTeX is not required for the canonical build.
