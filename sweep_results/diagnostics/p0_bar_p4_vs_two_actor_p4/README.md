# P0 BAR-P4 vs two-actor-P4 — merged 180-cell score archive

Cross-host merge of the locked P0 grid at source revision
`29fea94bee9d9276ac463caa2b897e3474fe1b2b`.

## Shards

| Half | Host | Artifact |
| --- | --- | --- |
| keys 1--90 | `ext_csv` | `first90_final_scores.csv` + this tree's `FROZEN_MANIFEST.json` |
| keys 91--180 | `ext_csh` | `hosts/ext_csh/p0_manifest_tail90/` |

Scientific configs, dataset normalizations, the requirements-file hash, and
source-file hashes match across the two frozen manifests. The resolved runtime
does not: the shards use Python 3.12/JAX 0.10/Flax 0.12 and Python
3.10/JAX 0.4/Flax 0.10 stacks, respectively (with corresponding Optax, NumPy,
Gym, driver, and kernel differences). Therefore this merge fails the frozen
P0 same-environment contract even though its score-table integrity passes.

## Primary outcome (score-level)

- task-equal mean (BAR − control): **-2.307552**
- 95% task-bootstrap interval: **[-7.247601909110738, 2.4117874955787904]**
- wins/ties: {'bar_wins': 55, 'two_actor_wins': 34, 'ties': 1, 'tie_tolerance': 1e-12}
- **outcome_label: `unresolved`**
- **compact_integrity_pass: `true`**
- **scientific_admissible_under_frozen_p0_contract: `false`**

Computed with `analyze_p0_bar_two_actor_p4.summarize` on the merged 90 pairs.
From the repository root, independently reconstruct the compact grid, hashes,
pair arithmetic, bootstrap, collapse tables, and admissibility flags with:

```bash
python scripts/diagnostics/verify_p0_score_merge.py
```

An exit status of zero certifies compact score integrity only; the verifier
still reports `scientific_admissibility: NOT_ADMISSIBLE`.
This is a descriptive sensitivity only, not a primary P0 result. It is **not**
an official create-only analyze/verify receipt over local
`params_1000000.pkl` for all 180 runs (tail checkpoints stay on `ext_csh`), and
the cross-stack merge cannot isolate the full-procedure contrast under the
locked protocol.

ext_csh retained 5 Walker2d-expert runs whose critic optimizer diverged after
finite weights. Keeping their finite 1M scores avoids selective outcome
filtering, but their nonfinite optimizer states would fail the official
checkpoint verifier.

## Files

| File | Role |
| --- | --- |
| `merged_final_scores.csv` | All 180 deployment scores + host provenance |
| `paired_scores.csv` | Locked 90-pair contrast table |
| `SUMMARY.json` | Locked primary/collapse summary + merge caveats |
| `MERGE_MANIFEST.json` | Head/tail hashes and scientific-identity checks |
| `first90_*` | Original ext_csv head-only compact shard |
| `tail90_final_scores.csv` | ext_csh first-actor + endpoint recovery from local `eval.csv` |
| `all180_first_endpoint_scores.csv` | first90 + tail90, same first90 schema |
| `FIRST_ENDPOINT_RECOVERY.json` | CPU recovery receipt; does not repair the primary gate |
| `STATUS.json` | Completeness + decision label snapshot |

Built: `2026-09-02T22:17:47+09:00`

First-actor recovery (`2026-09-05`, CPU, no re-evaluation): `d4rl_score` was already
in every tail90 `eval.csv`. The published compact shard had stored endpoint only.
Regenerate with `python scripts/diagnostics/recover_p0_tail90_first_actor.py`.
