# Fixed-operator order audit

Replacement for the archived actor-path semigroup defect tables as manuscript
evidence for discretization / semigroup-error claims. Spec: repo `TODO.md` P0-A.

## Artifacts

| File | Role |
| --- | --- |
| `HARNESS.json` | Analytic d=2 validation |
| `PROTOCOL_DRAFT.json` | Pre-checkpoint draft formulas, schemas, gates |
| `FROZEN_PROTOCOL.json` | Written only after all 18 fingerprints resolve |
| `CHECKPOINTS.json` / `CHECKPOINTS_UNRESOLVED.json` | 18 T=1 critic fingerprints |
| `state_indices/*.npy` | Frozen 512-row index sets |
| `endpoint_errors.csv` | Compact 720-cell endpoint table (after audit) |
| `raw_state_diagnostics.npz` | 368,640 pre-mask state records |
| `implicit_substeps.npz` | 1,142,784 implicit state/substep diagnostics |
| `euler_paths.npz` | Unprojected explicit/implicit paths |
| `SUMMARY.json` | Run slopes and scientific gates |
| `RESULT_MANIFEST.json` | Realized mask and artifact hashes |
| `STATUS.json` | Host readiness / blockers |

## Run

```bash
cd /path/to/MPI_sweep
python scripts/diagnostics/run_fixed_operator_order.py --phase all \
  --checkpoint-roots /path/to/results_qnorm
```

On ext_csh the 18 critics are currently unresolved (owned by ext_csv
`results_qnorm`). Harness + protocol draft + state indices can still be
produced. The script refuses to freeze or run the learned audit until every
checkpoint/config/weight fingerprint resolves.
