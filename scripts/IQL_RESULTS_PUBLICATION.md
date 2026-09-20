# IQL score publication

`STATUS.json` reports progress; it cannot supply missing numerical scores.
Run the publisher on the machine containing the original `config.json`,
`eval.csv`, `COMPLETE.json`, and final checkpoint. It needs Python 3.10+ and
the existing repository's Git authentication, but no JAX/GPU dependencies.

## Start once on ext_csv

```bash
cd /home/ext_csv/MPI_sweep
git fetch origin main
git show origin/main:scripts/start_iql_results_publisher.sh | bash -s -- ext_csv "$PWD"
```

This publishes current completed scores and starts a five-minute background
loop. It does not modify training scripts, restart jobs, or replace the
existing progress uploader. The loop lasts until stopped or the machine
reboots; rerun the same command after a reboot. Its PID and log are under
the Git directory's `score-publisher-ext_csv/` folder. Repeated starts detect
the existing process. Stop it using the recorded PID after checking the
process command. No system-wide scheduler is changed.

The source is `/raid/ext_csv/MPI_store/iql_qbc_gaussian_w2_hc_walker_s0123`.
Outputs are `sweep_results/iql_qbc_gaussian_w2_hc_walker_s0123/scores_verified.csv`
and `EXPORT.json`. Only those files are staged and pushed, using a disposable
checkout of latest main. The active checkout and its uncommitted changes are
left alone. Remote push races retry at most five times; no force push is used.

## Optional verified export on ext_csh

Use the same commands in `/home/ext_csh/MPI_sweep`, replacing `ext_csv` with
`ext_csh`. The source is `/home/ext_csh/MPI_sweep/results/iql_awr_fr`.
The existing `scores.csv` continues to work without installing this publisher.

## Evidence and selection

- `COMPLETE.json` must match config signature/schema and the 1M-step evaluation
  file hash. The final checkpoint must exist with the recorded nonzero size.
  Checkpoints are never deserialized, uploaded, or repeatedly rehashed; their
  training-time recorded checksum is retained with that limitation in `EXPORT.json`.
- All expected final evaluation rows must be present exactly once. Export only
  the final actor's mean-action score, preserving the original score precision.
  Intermediate actors, sample-action evaluation, and K>1 auxiliary baselines
  do not become standalone K=1 results.
- The two profiles follow the current queue's environment/T/K/seed grids. Older
  pilots outside those grids are recorded as excluded in the manifest.
- Any completed run with inconsistent evidence stops that tick without replacing
  published files. Incomplete runs remain pending; completion counts are never
  converted into scores. Distinct runs remain distinct and score conflicts stay
  visible in the table audit.
- The old ext_csh final-score CSV lacks step metadata. Its exact schema, run
  identities, final-actor fields, and reported completion count must agree before
  the table accepts it as `publisher_final_export`. No step value or checkpoint
  verification is invented. Unknown step-less sources remain excluded.
- A row with explicit final step wins over an old rounded export at the same
  actor priority, independent of which score is higher.

For a local export without Git writes:

```bash
python3 scripts/publish_iql_results.py --profile ext_csv --output /tmp/iql-score-export
```
