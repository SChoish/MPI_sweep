# I2 / I3 / I4 first-actor versus endpoint (available subset)

CPU recovery from existing `eval.csv`. No re-evaluation and no training.

Inclusion is frozen before the score tables: `INCLUSION_GRID.json` for
ext_csh s23/P0, `EXT_CSV_INCLUSION_GRID.json` for ext_csv s01.

| Table | What it is | What it is not |
| --- | --- | --- |
| `i2_s23_first_endpoint.csv` | implicit I2, seeds 2/3, 14-tau × 9 task | seeds 0/1 |
| `i3_s23_first_endpoint.csv` | implicit I3, same grid | seeds 0/1 |
| `explicit_i3_s23_first_endpoint.csv` | explicit I3 s23, stored separately | not mixed into implicit I3 |
| `i4_p0_first_endpoint.csv` | P0 BAR-P4 contemporaneous I4 | historical 504-cell I4 |
| `i2_s01_first_endpoint.csv` | implicit I2, seeds 0/1, 14-tau × 9 task | seeds 2/3; not a four-seed headline |
| `i3_s01_first_endpoint.csv` | implicit I3, seeds 0/1, same grid | seeds 2/3; not a four-seed headline |
| `I4_HISTORICAL_504_COVERAGE.json` | historical I4 endpoint 504/504; first-actor 4/504 | not a first/endpoint score table |
| `K1_INVENTORY.json` | local K1 s23 inventory for P0.C | not the missing `h=T/4` grid |

`d4rl_score` is the online first actor. `d4rl_piK` is the endpoint.
Within-run endpoint minus first is extraction gain, not superiority over a
separately trained deployment actor.

Historical `mpi4_norm` leaves `d4rl_score` empty on 500/504 cells, so no
eval.csv first/endpoint table is written from those blanks. CPU re-eval of
both online policies from `params_1000000.pkl`:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
LAB_ROOT=/home/ext_csv/mpi_sweep_lab \
python scripts/diagnostics/eval_i4_historical_first_endpoint.py
```

That writes `i4_historical_504_first_endpoint.csv` without overwriting original
`eval.csv`. Do not fill those blanks from P0 or s23 tables.

Regenerate eval.csv recoveries:

```bash
# ext_csh s23 / P0 tables (do not run this on ext_csv; it would rewrite s23 artifacts)
python scripts/diagnostics/recover_i234_first_endpoint.py --host s23

# ext_csv s01 tables and I4-504 coverage only; does not touch s23 files
python scripts/diagnostics/recover_i234_first_endpoint.py --host ext_csv
```
