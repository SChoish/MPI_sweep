# P0 BAR-P4 vs two-actor-P4 — merged 180-cell score archive

Cross-host merge of the locked P0 grid at source revision
`29fea94bee9d9276ac463caa2b897e3474fe1b2b`.

## Shards

| Half | Host | Artifact |
| --- | --- | --- |
| keys 1--90 | `ext_csv` | `first90_final_scores.csv` + this tree's `FROZEN_MANIFEST.json` |
| keys 91--180 | `ext_csh` | `hosts/ext_csh/p0_manifest_tail90/` |

Scientific configs, dataset normalizations, requirements hash, and source-file
hashes match across the two frozen manifests. Host paths, Python binaries, and
GPU UUIDs differ (expected).

## Primary outcome (score-level)

- task-equal mean (BAR − control): **-2.307552**
- 95% task-bootstrap interval: **[-7.247601909110738, 2.4117874955787904]**
- wins/ties: {'bar_wins': 55, 'two_actor_wins': 34, 'ties': 1, 'tie_tolerance': 1e-12}
- **outcome_label: `unresolved`**

Computed with `analyze_p0_bar_two_actor_p4.summarize` on the merged 90 pairs.
This is **not** an official create-only analyze/verify receipt over local
`params_1000000.pkl` for all 180 runs (tail checkpoints stay on `ext_csh`).

ext_csh retained 5 Walker2d-expert
cells whose critic optimizer diverged after finite weights.

## Files

| File | Role |
| --- | --- |
| `merged_final_scores.csv` | All 180 deployment scores + host provenance |
| `paired_scores.csv` | Locked 90-pair contrast table |
| `SUMMARY.json` | Locked primary/collapse summary + merge caveats |
| `MERGE_MANIFEST.json` | Head/tail hashes and scientific-identity checks |
| `first90_*` | Original ext_csv head-only compact shard |
| `STATUS.json` | Completeness + decision label snapshot |

Built: `2026-09-02T22:17:47+09:00`
