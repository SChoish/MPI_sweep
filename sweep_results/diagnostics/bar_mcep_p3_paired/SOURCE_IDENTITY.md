# Source identity conclusion (historical P3 control)

Scope: the exact code revision that produced the historical BAR-P3 and
MCEP-inspired two-actor P3 control runs archived in this folder.

## Conclusion

The exact historical Git revision is **verified unrecoverable**. The identity
search is closed with a negative result. Only file digests identify the local
training sources; they do not pin a repository commit or an environment lock.

Do not attribute these runs to any specific commit hash. None was recorded at
training time, and none can be reconstructed after the fact.

## Evidence

- `SUMMARY.json` → `source_identity.qualification`: "The run configs did not
  record a Git commit. File digests identify the local training sources but do
  not establish a clean repository snapshot or environment lock."
- `SUMMARY.json` → `source_identity.training_file_sha256`: SHA-256 of the
  recovered training sources at recovery time:
  - `train_td3bc.py`: `aea510db3fafb4829f92902a58c638d9d2f29939cc069315204c9655b942bc9c`
  - `launch_mpi_sweep.py`: `0874903d8f4fe3608b8a8e7d07c64fecc7bae60768ec6c926b64f3ebaa8da5f2`
- `SOURCE_MANIFEST.csv`: per-artifact relative path, byte size, and SHA-256 for
  all 180 raw config/eval/final-checkpoint rows. The per-run `config.json`
  payloads contain no Git revision field.

## Why it is unrecoverable

The trainer that produced these runs wrote `config.json` from parsed CLI
arguments only (`vars(args)`); it never captured a Git revision, dirty-worktree
flag, or dependency lock. Because provenance was never persisted at run time,
the commit cannot be derived from the surviving configs, evaluation tables, or
checkpoints. A file digest confirms *which bytes* of source were present locally
at recovery, but cannot establish a clean-tree commit or the full environment.

## What identifies these sources instead

The digests above are the strongest available identity: they fix the exact
bytes of the recovered `train_td3bc.py` and `launch_mpi_sweep.py`. Treat this
control as a **compatible recovered implementation**, not a proof of the exact
historical source revision.

## Forward fix (new runs only)

Historical runs cannot be backfilled. New training runs now retain provenance
automatically (Git revision + dirty flag, source digests, environment/package
versions, host/accelerator inventory, dataset and normalization hashes, and a
final-checkpoint digest). See `scripts/PROVENANCE.md` for the contract.
