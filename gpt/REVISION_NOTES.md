# Revision Map — AISTATS Rewrite

The GPT bundle was rewritten after the completed P0/P1/P2/P3 follow-up evidence changed the safest mechanism claim.

| Evidence | Manuscript consequence |
|---|---|
| Four-seed BAR depth sweep: high-budget mean `25.38 -> 41.20 -> 51.92 -> 63.97`, collapse cells `34 -> 22 -> 16 -> 8` | Retain BAR depth as a strong end-to-end stability intervention. |
| P3 direct movement: target displacement lower in 88/90 vs TD3+BC while typical final displacement does not contract | Retain target/deployment movement asymmetry as an empirical BAR signature. |
| P3 direct two-actor control: 59.22 vs 57.67 contemporaneous BAR-P3; 9/45 seed-mean collapses for both | Remove any claim that sequential re-centering is necessary for the P3 aggregate. |
| Locked P4 control: BAR minus two-actor mean `-2.31`, interval `[-7.25, 2.41]`, paired median `+0.68`, BAR wins 55/90 | Treat the chain's incremental return value as unresolved. Do not claim equivalence, BAR superiority, or two-actor superiority. |
| P0 heavy tails and opposite mean/median signs | Describe the procedures as having heterogeneous seed/task failures rather than a uniform ordering. |
| Verified P1 270-checkpoint target-value audit | Distinguish target-action displacement from actual target-value perturbation; both remain diagnostics rather than return certificates. |
| Verified P2 ReLU residence rates `.954/.911/.842/.748` at local horizons `.025/.05/.1/.2` | Narrow the Wasserstein/explicit--implicit discussion to a local organizing interpretation. |
| H200 actor-cost audit | Report measured cost in addition to the analytical `O(K)` actor-work statement. |

## New central claim

The current manuscript's strongest mechanistic statement is **target--deployment separation**, not sequential-chain superiority. A conservative branch supplies Bellman targets while a separately stronger branch may be deployed. BAR remains the depth-indexed procedure that exposed and stress-tested this design axis.

This framing explicitly acknowledges MCEP as prior work on conservative target / less constrained evaluation policies. The paper does not claim the broad two-policy principle as novel.

## P0 provenance wording

All 180 P0 final scores are present in a frozen cross-host score-level archive, but the archive does not satisfy the originally specified single-artifact-tree checkpoint-bound analyze/verify receipt. The manuscript therefore reports the locked numerical result and `unresolved` label while retaining this provenance limitation.
