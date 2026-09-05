# I2 / I3 / I4 first-actor versus endpoint (available subset)

CPU recovery from existing `eval.csv`. No re-evaluation and no training.

Inclusion is frozen in `INCLUSION_GRID.json` before the score tables.

| Table | What it is | What it is not |
| --- | --- | --- |
| `i2_s23_first_endpoint.csv` | implicit I2, seeds 2/3, 14-tau × 9 task | seeds 0/1 |
| `i3_s23_first_endpoint.csv` | implicit I3, same grid | seeds 0/1 |
| `explicit_i3_s23_first_endpoint.csv` | explicit I3 s23, stored separately | not mixed into implicit I3 |
| `i4_p0_first_endpoint.csv` | P0 BAR-P4 contemporaneous I4 | historical 504-cell I4 |
| `K1_INVENTORY.json` | local K1 s23 inventory for P0.C | not the missing `h=T/4` grid |

`d4rl_score` is the online first actor. `d4rl_piK` is the endpoint.
Within-run endpoint minus first is extraction gain, not superiority over a
separately trained deployment actor.

Regenerate:

```bash
python scripts/diagnostics/recover_i234_first_endpoint.py
```
