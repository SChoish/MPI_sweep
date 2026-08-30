# GPT AAAI-26 Manuscript Bundle

This directory is the revised GPT manuscript bundle for the offline TD3+BC actor-refinement study. It replaces the earlier provisional AISTATS-2027 bundle with an AAAI-26 submission-format version and incorporates:

- the current `aistats26_manuscript/` scientific narrative and implementation audit;
- the compact re-audit under `sweep_results/diagnostics/audit_pack/`;
- the external critical report supplied by the authors;
- a second source/code audit of the released trainer, score matrices, controls, and audit outputs.

The working method name remains **BAR: Budgeted Actor Refinement**. The name is deliberately isolated in two LaTeX macros in `paper.tex` and `supplement.tex`:

```tex
\newcommand{\method}{BAR}
\newcommand{\methodlong}{Budgeted Actor Refinement}
```

Changing those two definitions updates the manuscript without changing the scientific content. No replacement acronym has been silently imposed.

## Important correction retained in this revision

Refinement depth `K` remains a central experimental design variable. The earlier GPT criticism that cross-`K` comparisons require identical critic initializations was withdrawn. The complete sweep estimates the marginal performance of each randomized procedure; common initialization would be a variance-reduction device, not a validity condition. The supplement states this explicitly and treats paired transitions as descriptive under the implemented seed coupling.

## AAAI-26 format

- `paper.tex` uses the unmodified `aaai2026.sty` submission option.
- The main PDF contains five technical-content pages, one reference page, and the required reproducibility checklist after the references.
- `supplement.tex` is a separate anonymous technical appendix in the same AAAI-26 style.
- `aaai2026.sty` has Git-blob SHA-1 `1c587a54d5613355974d8ac25ebb7d5d741c84e0`; see `AUTHOR_KIT_SOURCE.md`.

## Main scientific revisions

1. The normalization factors `C_k` are defined separately for the first proximal hop, later proximal hops, the first linearized hop, and later linearized hops.
2. The linearized realization is described as **action-metric-matched**, not fully scale-matched, because its first-hop normalization point differs.
3. Archived evaluation is correctly reported as every 50,000 critic updates, with the headline score equal to the ten-episode mean at the final one-million-update checkpoint.
4. The local `O(beta^3)` expansion assumes a locally Lipschitz Hessian; the weaker `C^2` remainder is stated separately.
5. `E` consistently denotes the `expert-v2` datasets; `ME` and `medium-expert-v2` are not used.
6. The ideal proximal operator is separated from the independently initialized persistent actors trained by one Adam step. The analysis motivates but does not certify the live chain.
7. The main empirical result includes task-cluster uncertainty and distinguishes broad `K=4` versus `K=1` rescue from the less certain collapse-risk difference versus `K=3`.
8. Target movement is called **sample-anchored next-state target-action displacement** and is not conflated with the current-state final-policy proxy.
9. The simulator statistic uses medians of checkpoint-level medians (`-41.4` versus `1.70e12`), and the route-shadow result reports both the primary four-checkpoint `8/8` direction and the final-checkpoint-only `5/8` sensitivity.
10. Wasserstein Policy Optimization and one-step offline RL are positioned explicitly in related work.

## Build

Requirements: Python 3 with NumPy and Matplotlib, `pdflatex`, Poppler (`pdfinfo`, `pdffonts`), and the repository's compact `sweep_results/` directory.

```bash
./build_pdf.sh
```

The canonical build:

1. verifies the source score matrices and compact audit claims;
2. regenerates all figures;
3. compiles the main paper and supplement twice;
4. checks page limits, forbidden packages, LaTeX warnings, and embedded fonts;
5. writes:
   - `gpt_aaai26_manuscript.pdf`
   - `gpt_aaai26_supplement.pdf`

`verify_bundle.py` may also be run directly:

```bash
python3 verify_bundle.py --results-dir ../sweep_results --prebuild
python3 verify_bundle.py --results-dir ../sweep_results --postbuild
```

## Evidence boundary

The complete score grid and compact audit claims are reproducible from the released CSV/JSON material. The audit pack re-aggregates preserved local NPZ/per-state inputs, but those multi-gigabyte raw inputs are not duplicated here. Some historical targeted-control launch wrappers and exact environment lockfiles are also unavailable. The checklist therefore uses `Partial` rather than overstating end-to-end reproducibility.
