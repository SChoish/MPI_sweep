# P2 ReLU activation-region residence pilot

This development-only P2 v4 pilot replaces an invalid learned-ReLU convergence-
order estimate with an exact activation-region scope audit.  It is an
engineering validation artifact, not a confirmatory result or a learned-network
order claim.

## What this pilot establishes

For a fixed state, the frozen Q1 ReLU network is affine inside one activation
region.  Along

```text
v = action_dim * grad_a Q1(s, a0) / C_ref
```

the runner reconstructs that region and computes the first time the straight
path `a0 + t v` reaches either a ReLU boundary or the action box.  Before that
time, the explicit candidate and the same-region local backward-Euler candidate
coincide exactly.  A boundary-crossing row is a scope diagnostic, not a
numerical error.  This does not identify a global proximal maximizer and does
not estimate learned-critic convergence order.

The estimand is the unique-horizon empirical survival curve
`P(t_exit > t)`.  Full residence depends only on `T`, and first-step residence
depends only on `h = T/K`; repeated `(T,K)` cells with the same horizon are
deterministic views, not independent evidence.

The nontrivial first-order check remains the smooth quadratic harness with an
exact matrix-exponential solution.  Synthetic ReLU cases cover a layer-1 exit,
a layer-2 exit, an action-box exit, simultaneous exits, an anchor on a boundary,
and a resident horizon.

## Pilot/final separation

`DESIGN_LOCK.json` is written before pilot input selection.  `INPUT_LOCK.json`
then matches predeclared SHA-256 values and semantic filenames before any
JSON, pickle, HDF5, or NumPy input is deserialized.  Hashing necessarily reads
raw bytes; the temporal guarantee is specifically before semantic use.  The
pilot is restricted to one already exposed HalfCheetah checkpoint and 64 new
indices that exclude both canonical seed-0 and seed-1 archived fixed-operator
index files (512 rows each, 1,022-row union).  Missing, extra, duplicate, or
hash-mismatched exclusions are rejected.  Residence rates have no pass/support
threshold and are never merged into a scientific result.

The locked final contract excludes the entire HalfCheetah family.  A later
final run must use the six Hopper/Walker tasks, seeds 0--1, and 512 newly sampled
indices per run after protocol freeze.  Its exclusion inventory must contain
every development-attempt index set.  This includes failed and superseded bundles
`VnOjA4`, `InTWgf`, `OLDBdx`, `xlMNfc`, and `sqfIVj`.  The last is
the final import-safe development bundle.  This pilot runner intentionally does
not implement or execute the confirmatory run.

## Run

Use a new, external output directory.  The commands are CPU-only and scientific
artifacts are create-only.

Only run this against the pinned trusted local checkpoint.  The runner and
verifier deserialize that pickle; an untrusted replacement is unsafe and will
also fail `INPUT_LOCK` before deserialization.

```bash
export PY=/home/ext_csv/miniconda3/envs/offrl/bin/python
export OUT=/home/ext_csv/mpi_sweep_lab/p2-relu-residence-pilot-YYYYMMDD

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
"$PY" scripts/diagnostics/run_p2_relu_residence.py \
  --phase all \
  --out-dir "$OUT" \
  --checkpoint /home/ext_csv/mpi_sweep_lab/results_qnorm/halfcheetah-medium-v2_tau1_seed0/params_1000000.pkl \
  --config /home/ext_csv/mpi_sweep_lab/results_qnorm/halfcheetah-medium-v2_tau1_seed0/config.json \
  --dataset /raid/ext_csv/datasets/d4rl/halfcheetah_medium-v2.hdf5

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
"$PY" "$OUT/SOURCE_SNAPSHOT/verify_p2_relu_residence.py" --out-dir "$OUT"
```

The runner copies its exact runner/verifier sources into the create-only bundle
and records the dirty Git revision; invoke that snapshot verifier as shown.
The verifier does not import the runner.  It recomputes the smooth harness with
high-precision `Decimal`, reconstructs Q1 with independent NumPy code,
regenerates selected rows, checks coordinate finite differences, directly
brackets reported ReLU exits within the tested horizon and every finite action-
box hit, checks combined first-event/tie ordering, and recomputes every CSV and
summary cell including scale-aware same-region affine residuals.

If a run stops after creating a partial directory, preserve it as a failed
attempt and restart from a different empty output directory.  Never resume by
overwriting create-only artifacts.

## Inclusion boundary

The pilot is healthy only when `MANIFEST.json.status == "pilot_complete"` and
`VERIFY.json.pass == true`.  Even then, both files must state
`scientific_admissible == false`.  Residence coverage is development output;
there is no learned slope, learned support gate, return claim, or live-chain
claim in this bundle.
