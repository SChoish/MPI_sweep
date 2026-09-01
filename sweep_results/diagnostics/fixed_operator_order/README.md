# Fixed-operator order audit

Replacement for the archived actor-path semigroup defect tables as manuscript
evidence for discretization / semigroup-error claims. Spec: repo `TODO.md` P0-A.

## Artifacts

| File | Role |
| --- | --- |
| `HARNESS.json` | Analytic d=2 validation |
| `FROZEN_PROTOCOL.json` | Locked grid, formulas, gates |
| `CHECKPOINTS.json` / `CHECKPOINTS_UNRESOLVED.json` | 18 T=1 critic fingerprints |
| `state_indices/*.npy` | Frozen 512-row index sets |
| `endpoint_errors.csv` | Compact 720-cell endpoint table (after audit) |
| `SUMMARY.json` | Run slopes and scientific gates |
| `STATUS.json` | Host readiness / blockers |

## Run

```bash
cd /path/to/MPI_sweep
python scripts/diagnostics/run_fixed_operator_order.py --phase all \
  --checkpoint-roots /path/to/results_qnorm
```

On ext_csh the 18 critics are currently unresolved (owned by ext_csv
`results_qnorm`). Harness + protocol + state indices can still be produced.
