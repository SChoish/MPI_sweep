# P0 BAR-P4 vs two-actor-P4 — first-90 host archive

Compact progress archive for the locked P0 experiment frozen at revision
`29fea94` (`FROZEN_MANIFEST.json`).

## Scope

- **Included:** first 90 of the 180 manifest keys (45 BAR-P4 + 45 two-actor-P4)
  with `params_1000000.pkl` and a `1000000` eval row on `ext_csv`.
- **Excluded here:** the second 90 keys (assigned to `ext_csh` as `hosts/ext_csh/p0_manifest_tail90/`; not merged into this archive).
- **Not included:** raw checkpoints (only SHA-256 + byte sizes in
  `SOURCE_MANIFEST.csv`), logs, or a full-grid decision label.

## Files

| File | Role |
| --- | --- |
| `FROZEN_MANIFEST.json` | Exact frozen 180-key contract (+ `.sha256`) |
| `first90_final_scores.csv` | Per-run deployment/target D4RL scores |
| `first90_paired_scores.csv` | Env×T×seed pairs present in the first half |
| `SOURCE_MANIFEST.csv` | Host paths + SHA-256 for config/eval/final ckpt |
| `STATUS.json` | Completeness snapshot; `decision_label` is null |

## Score columns

| Procedure | Deployment score |
| --- | --- |
| BAR-P4 | `d4rl_pi4` |
| two-actor-P4 | `d4rl_eval` |

## Interpretation gate

The author-defined P0 decision labels require the full 180-cell analyze/verify
pipeline. This archive records first-half finals only and **does not** release
`four_actor_procedure_supporting`, `two_actor_procedure_supporting`,
`author_band_comparable`, or `unresolved`.

Built: `2026-09-02T22:13:32+09:00`
