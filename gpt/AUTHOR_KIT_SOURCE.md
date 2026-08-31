# AAAI-26 Author-Kit Provenance

The manuscript uses `aaai2026.sty` with the `submission` option and follows the AAAI-26 main-track layout.

- Official author-kit endpoint: `https://aaai.org/authorkit26-1/`
- Official submission instructions: `https://aaai.org/conference/aaai/aaai-26/submission-instructions/`
- Official checklist source: `AuthorKit26/ReproducibilityChecklist/LaTeX/ReproducibilityChecklist.tex` inside `AuthorKit26-1.zip`; all 31 questions are unchanged and only response slots are filled.
- Style source used for the bundle: `AuthorKit26/AnonymousSubmission/LaTeX/aaai2026.sty` from the official `AuthorKit26-1.zip`.
- `aaai2026.sty` Git-blob SHA-1: `989b761198b6d142da3372a06e032fd8979972e3`
- Package declaration: `\ProvidesPackage{aaai2026}[2026/06/17 AAAI 2026 Submission format]`

The style file is included without modification. The main source uses a manual `thebibliography` block, so the bundle does not depend on a separate `.bst` file. The structured citation ledger is retained in `references.bib` for future editing.

AAAI-26 regular papers permit seven technical-content pages plus references, and require the reproducibility-checklist answers after the references. The build script checks that the technical-content boundary is no later than page 7, that all 31 official checklist questions are answered, and that the checklist begins afterward.
