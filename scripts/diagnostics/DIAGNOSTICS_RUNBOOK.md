# Diagnostics runbook (index)

This file is kept for the old path `scripts/DIAGNOSTICS_RUNBOOK.md`.

**Canonical package:** [`scripts/diagnostics/`](./)

| Doc | Purpose |
| --- | --- |
| [README.md](README.md) | Package index + quick start |
| [HOST_RUNBOOK.md](HOST_RUNBOOK.md) | Per-host how-to |
| [ARTIFACT_MAP.md](ARTIFACT_MAP.md) | What host dumps regenerate vs archive/retrain |

```bash
bash scripts/diagnostics/run_host.sh --host "$(hostname -s)" --seeds 0 1 --job ckpt
```

Compatibility shim (same flags): `bash scripts/run_host_diagnostics.sh ...`
