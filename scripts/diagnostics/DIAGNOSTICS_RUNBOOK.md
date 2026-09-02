# Diagnostics runbook (index)

This file is kept for the old path `scripts/DIAGNOSTICS_RUNBOOK.md`.

**Canonical package:** [`scripts/diagnostics/`](./)

| Doc | Purpose |
| --- | --- |
| [README.md](README.md) | Package index + quick start |
| [HOST_RUNBOOK.md](HOST_RUNBOOK.md) | Per-host how-to |
| [ARTIFACT_MAP.md](ARTIFACT_MAP.md) | What host dumps regenerate vs archive/retrain |
| [P1_TARGET_VALUE_RUNBOOK.md](P1_TARGET_VALUE_RUNBOOK.md) | Locked exact-grid target-value audit |
| [ACTOR_COST_RUNBOOK.md](ACTOR_COST_RUNBOOK.md) | Locked actor timing/memory measurement |

```bash
bash scripts/diagnostics/run_host.sh --host "$(hostname -s)" --seeds 0 1 --job ckpt
```

Compatibility shim (same flags): `bash scripts/run_host_diagnostics.sh ...`

P1 and actor-cost are standalone protocols. Neither is invoked by
`run_host.sh --job all`; use its dedicated runbook and independent verifier.
