# GPT revision - AAAI 2026 BAR bundle

This directory is an independently revised AAAI-26 manuscript package for the working method name **BAR (Budgeted Actor Refinement)**.

The revision preserves refinement depth `K` as an empirical design variable. It only retracts the earlier, incorrect criticism that cross-`K` comparisons require identical critic initialization. Independent randomized seeds are valid for marginal seed-average comparisons; shared initialization is optional variance reduction.

## Main corrections

- exact implementation-specific `C_k` definitions for proximal and linearized paths;
- evaluation every 50,000 updates and final 1M/10-episode aggregation;
- locally Lipschitz Hessian assumption for the cubic Taylor remainder;
- code-grounded exact D4RL identifiers (`*-expert-v2`);
- explicit separation between the ideal proximal operator and the live one-Adam-step persistent actors;
- dataset-cluster uncertainty for K4-vs-K1 and K4-vs-K3;
- precise target/final displacement estimands;
- post-hoc route wording and final-checkpoint 5/8 sensitivity;
- checkpoint-level simulator aggregation;
- one-page pseudocode/configuration disclosure in the supplement;
- corrected related-work positioning, including Wasserstein Policy Optimization;
- sanitized compact audit pack without local absolute paths.

## Build

```bash
cd gpt_aaai26
./build_pdf.sh
```

Requirements: Python 3 with NumPy and Matplotlib, `latexmk`, and a LaTeX distribution.

Outputs:

- `bar_gpt_aaai26_manuscript.pdf`
- `bar_gpt_aaai26_supplement.pdf`

The build regenerates figures, validates all headline numbers including the complete K=4 grid, runs the compact audit checks, compiles both PDFs, verifies that technical content ends by AAAI page 7, and rejects Type-3 fonts.

## Review and audit

- `CRITICAL_REVIEW.md`: consolidated critical review.
- `REVISION_LOG.md`: item-by-item change map.
- `audit_pack/`: sanitized compact diagnostic audit.
- `SOURCE_MANIFEST.md`: source and evidence provenance.

`BAR` remains a working name; no replacement acronym is asserted in this revision.
