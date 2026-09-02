# Provenance contract (new runs only)

Forward-only provenance retention for MPI_sweep. This applies to **new**
training runs and sweeps; historical runs are not backfilled (see
`sweep_results/diagnostics/bar_mcep_p3_paired/SOURCE_IDENTITY.md` for why the
historical P3 control revision is unrecoverable).

All capture is **best-effort**: every collector degrades to a recorded error
string and never crashes or blocks training. Implementation lives in
`provenance.py` at the repo root.

## Per-run: `PROVENANCE.json`

Written by `train_td3bc.py` into each run directory (next to `config.json`),
then enriched in place at two later points:

1. **At config write** (`write_initial_provenance`):
   - `git`: revision, dirty flag, branch (best-effort; failure recorded)
   - `source_files`: path + size + SHA-256 of `train_td3bc.py` and
     `launch_mpi_sweep.py`
   - `packages`: python + `jax`/`jaxlib`/`flax`/`optax`/`numpy` versions (or
     `null` if not importable)
   - `host`: hostname, platform
   - `accelerator`: `CUDA_VISIBLE_DEVICES` + `nvidia-smi` GPU index/UUID/name
   - `config`: resolved CLI args
   - `dataset`: path + size + mtime. **Full HDF5 SHA-256 is intentionally
     skipped** (`hash_skipped` documents this) — rehashing multi-GB datasets
     every run is not worth the cost and no cached hash helper exists here.
   - `normalization`: placeholder (`deferred until after data load`)
   - `final_checkpoint`: `null` placeholder
2. **After data load** (`update_provenance_normalization`): fills
   `normalization` with the SHA-256 of the loaded `mean` and `std` arrays.
3. **At the final checkpoint** (`update_provenance_final_checkpoint`): fills
   `final_checkpoint` with the `params_1000000.pkl` path + SHA-256 when the run
   reaches `max_timesteps`.

## Per-sweep: `SWEEP_PROVENANCE.json`

Written once by `launch_mpi_sweep.py` into `--log-dir` (skipped on `--dry-run`).
Same base block as above plus a `sweep` section: method, integrator, hops, tau
grid, seeds, envs, GPUs, `max_timesteps`, and the pending job tags/count.

## Notes

- These files are small JSON and safe to commit alongside a run's `config.json`
  / `eval.csv`. Raw checkpoints and datasets remain uncommitted as before.
- A missing `git`/`nvidia-smi` binary or an unimportable package yields a
  recorded `null`/error, not a failed run.
