# AAAI-26 Author-Kit Provenance

The manuscript uses `aaai2026.sty` with the `submission` option and follows the AAAI-26 main-track layout.

- Official author-kit endpoint: `https://aaai.org/authorkit26-1/`
- Official submission instructions: `https://aaai.org/conference/aaai/aaai-26/submission-instructions/`
- Style source used for the bundle: the public `AAAI-2026-Latex-Unified` mirror of the AAAI-26 author kit.
- `aaai2026.sty` Git-blob SHA-1: `1c587a54d5613355974d8ac25ebb7d5d741c84e0`
- Package declaration: `\ProvidesPackage{aaai2026}[2026/04/29 AAAI 2026 Submission format]`

The style file is included without modification. The main source uses a manual `thebibliography` block, so the bundle does not depend on a separate `.bst` file. The structured citation ledger is retained in `references.bib` for future editing.

AAAI-26 regular papers permit seven technical-content pages plus references, and require the reproducibility-checklist answers after the references. The build script checks that the technical-content boundary is no later than page 7.
