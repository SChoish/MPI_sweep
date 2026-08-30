# Revision log

## Interpretation correction

The prior GPT review misread the user's statement that “K is almost meaningless.” The statement concerned the alleged need to share critic initialization across K, not the scientific role of refinement depth. This revision therefore:

- keeps K=2/3/4 as the tested refinement-depth variable;
- keeps the complete K=4 result central;
- removes the claim that K-dependent initialization is a confound or that a shared-initialization rerun is required;
- states the correct common-random-number interpretation in the supplement.

## P0 fixes from the external audit

1. **Q-scale `C_k`:** split into path- and hop-specific definitions. First proximal hop uses the current pre-update actor output; later proximal hops use the predecessor/reference; linearized hops use the reference, including dataset action at hop 1. Blanket first-hop scale matching is removed.
2. **Evaluation frequency:** corrected from 5,000 to 50,000 updates; final score is the 1M checkpoint's 10-episode mean.
3. **Taylor remainder:** cubic remainder now assumes a locally Lipschitz Hessian (C3 suffices); the C2-only remainder is stated as weaker.
4. **Dataset names:** standardized to exact code-grounded `*-expert-v2` identifiers. The report's `medium-expert` suggestion is not adopted because it conflicts with the implementation.

## Theory/implementation gap

- Proximal results are framed as an ideal/local interpretation.
- The live method is identified as one Adam step per independently initialized persistent actor.
- The conditional margin premise is not claimed to have been verified in training.
- The fixed-K composition theorem explicitly assumes a near-identity map with `Psi_0 = id`.
- The frozen-critic audit is described as an ideal action-space check, not a live optimizer-chain validation.

## Empirical/statistical changes

- Added dataset-cluster bootstrap intervals for score and raw collapse risk.
- Distinguished broad K4-vs-K1 evidence from weaker incremental K4-vs-K3 collapse-risk evidence.
- Defined the stability envelope operationally and called the threshold descriptive.
- Clarified that T is a nominal actor-loss coefficient budget, not compute or realized displacement.
- Described the four-seed Hopper K4 result as a within-K4 check, not cross-K confirmation.
- Replaced “directly measured target-policy displacement” with the exact sample-anchored next-state target-action estimand.
- Distinguished it from the current-state online final-policy proxy.
- Changed simulator reporting to median checkpoint medians.
- Added route post-hoc selection, mean/median imbalance, and final-checkpoint-only 5/8 sensitivity.

## Reproducibility and presentation

- Added pseudocode and a complete implementation/configuration table.
- Added exact metric definitions, common critic provenance, bootstrap units, and iteration count.
- Updated checklist answers where full evidence is unavailable.
- Added WPO and corrected publication venues for adjacent OT-policy work.
- Rebuilt the main figure with distinguishable grayscale line/marker styles and uncertainty.
- Added a technical-end page label rather than checking only the start of the conclusion.
- Added complete K4 headline checks to the verifier.
- Removed local absolute paths from the submission-facing audit pack.
