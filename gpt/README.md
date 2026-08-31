# GPT AAAI-26 Manuscript Bundle

This directory is the revised GPT manuscript bundle for the offline TD3+BC actor-refinement study. It replaces the earlier provisional AISTATS-2027 bundle with an AAAI-26 submission-format version and incorporates:

- the prior `aistats26_manuscript/` scientific narrative and implementation audit;
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
- The built main PDF has 9 pages: 6 technical-content pages, 1 reference page, and 2 pages for the complete official checklist. The checklist is outside the technical-content limit.
- `ReproducibilityChecklist.tex` is the unshortened 31-question template from the official AAAI-26 author kit with response slots filled.
- `supplement.tex` is a separate anonymous technical appendix in the same AAAI-26 style.
- `aaai2026.sty` has Git-blob SHA-1 `989b761198b6d142da3372a06e032fd8979972e3`; see `AUTHOR_KIT_SOURCE.md`.

## Main scientific revisions

1. `T` is tied to the trainer exactly: `T = tau = alpha/2`, `K=1` recovers the TD3+BC coefficient `alpha=2T`, and each depth-`K` hop uses `h=T/K`. It is not compute or realized policy distance.
2. The normalization factors `C_k` are defined separately for the first proximal hop, later proximal hops, the first linearized hop, and later linearized hops.
3. The linearized realization is described as **action-metric-matched**, not fully scale-matched, because its first-hop normalization point differs.
4. The headline score is the ten-episode mean at the final one-million-update checkpoint. Intermediate evaluation cadence is reported by run family rather than incorrectly forced to one universal frequency.
5. The local `O(beta^3)` expansion assumes a locally Lipschitz Hessian; the weaker `C^2` remainder is stated separately.
6. `E` consistently denotes the `expert-v2` datasets; `ME` and `medium-expert-v2` are not used.
7. The ideal proximal operator is separated from the independently initialized persistent actors trained by one Adam step. The analysis motivates but does not certify the live chain.
8. The four-seed proximal grid gives a strict, equally task-and-budget-weighted `T>=4` depth ordering (`25.38 -> 41.20 -> 51.92 -> 63.97`) reproduced separately by both disjoint host-separated seed pairs. Both score and collapse-risk intervals favor `K=4` over `K=3`.
9. The 270-checkpoint movement audit is explicitly scoped to TD3+BC/P2/P3; it supports the P3 exposure signature and is not presented as a direct P4 audit.
10. Target movement is called **sample-anchored next-state target-action displacement** and is not conflated with the current-state final-policy proxy.
11. The simulator statistic uses medians of checkpoint-level medians (`-41.4` versus `1.70e12`), and the route-shadow result reports both the primary four-checkpoint `8/8` direction and the final-checkpoint-only `5/8` sensitivity.
12. Wasserstein Policy Optimization and one-step offline RL are positioned explicitly in related work.
13. The abbreviated checklist was replaced by all 31 official AAAI-26 questions, with `Partial`/`No` answers tied to explicit artifact limits in the supplement.

## Build

Requirements: Python 3 with NumPy and Matplotlib, `pdflatex`, Poppler (`pdfinfo`, `pdffonts`), and the repository's compact `sweep_results/` directory. ZIP creation uses only Python's standard library.

```bash
./build_pdf.sh
```

The canonical build:

1. verifies the source score matrices and compact audit claims;
2. regenerates all figures;
3. compiles the main paper and supplement twice;
4. checks page limits, forbidden packages, LaTeX warnings, and embedded fonts;
5. creates and validates an anonymous deterministic code/data ZIP with logical checkpoint identifiers;
6. writes:
   - `gpt_aaai26_manuscript.pdf`
   - `gpt_aaai26_supplement.pdf`
   - `gpt_aaai26_bundle.zip`

`verify_bundle.py` may also be run directly:

```bash
python3 verify_bundle.py --results-dir ../sweep_results --prebuild
python3 verify_bundle.py --results-dir ../sweep_results --postbuild
```

## Evidence boundary

The complete score grid and compact audit claims are reproducible from the released CSV/JSON material. The generated ZIP includes the trainer, required score matrices, diagnostic scripts, and the four compact audit tables consumed by the verifier; checkpoint paths are rewritten as logical artifact identifiers. Proximal scores use four seeds in two disjoint host-separated execution pairs; linearized scores and mechanism audits use two. The exact per-run manifests for the newer proximal pair, the multi-gigabyte diagnostic NPZ/per-state inputs, some historical targeted-control launch wrappers, and exact environment lockfiles are not duplicated here. The official checklist therefore uses `Partial` or `No` for incomplete configuration-level and end-to-end historical reproduction while retaining the full archived score evidence.
