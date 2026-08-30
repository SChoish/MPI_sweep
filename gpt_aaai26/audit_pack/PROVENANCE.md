# Provenance and sanitization

The compact files in this directory were transcribed from the repository audit at `sweep_results/diagnostics/audit_pack/` and from the complete sweep matrices. The source audit recomputed statistics from preserved NPZ/per-state data and recorded SHA-256 checksums of the local inputs.

This submission-facing copy deliberately removes:

- local absolute filesystem paths;
- host names and user names;
- local output directories;
- non-portable checkpoint paths.

Logical identities are expressed by method, dataset ID, nominal budget, seed, checkpoint step, and aggregation rule. The original raw inputs remain too large for the compact manuscript bundle and are not duplicated here.
