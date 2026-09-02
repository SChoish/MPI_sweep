# Actor-update cost protocol

This is the execution and reverification protocol for the released
measured-complexity item in [`TODO.md`](../../TODO.md). The compact snapshot is
under `sweep_results/diagnostics/actor_cost/`; the commands below reproduce the
measurement on a fresh stack. A plan-only invocation is the safe default and
does not import JAX, open a dataset, create an output directory, or start a
worker.

## Released measurement

The archived run fixed an NVIDIA H200, JAX/JAXlib 0.10.2,
`hopper-medium-v2`, batch size 256, and `T=12`. The independent verifier
reported `evidence_status: measured`.

| K | Compiled actor-phase dispatch (ms) | K=1 ratio | Backend peak (bytes) | K=1 ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.434567 | 1.000 | 9,684,736 | 1.000 |
| 2 | 1.542470 | 1.075 | 14,680,064 | 1.516 |
| 3 | 1.685672 | 1.175 | 15,627,520 | 1.614 |
| 4 | 1.887347 | 1.316 | 18,045,440 | 1.863 |

The timed dispatch contains the full K-actor production phase and excludes
lowering, compilation, warmup, and critic optimization. The memory column is
the fresh worker's process-lifetime backend high-water mark through data
transfer, initialization, compilation, warmup, and trials, not an isolated
steady-state actor-call peak. These are one-stack measurements, not a
hardware-independent scaling law.

## What is measured

The sweep holds the dataset batch, initialization seed, total budget `T`,
optimizer settings, software environment, and one visible accelerator fixed.
Only `K` varies over `{1,2,3,4}`. Each `K` runs in a fresh sequential process
and calls the production BAR actor functions. Critic optimization is excluded;
the target-network update that accompanies the first actor update remains part
of the production actor path.

The release run must select one local dataset before execution and retain its
path, SHA-256, filtered row count, observation/action shapes, and normalization
hash. The harness freezes the raw file path and hash before the first K worker;
each fresh worker independently derives the filtered dimensions,
normalization, and fixed batch, and the verifier requires those records to
match. The protocol does not prescribe a benchmark dataset; do not choose
among datasets after inspecting candidate timings. Unless a new protocol is
reviewed first, retain the printed defaults: batch size 256, `T=12`,
initialization seed 0, sample seed 20260902, learning rate 3e-4, Polyak rate
0.005, three warmups, ten trials, and ten synchronized calls per trial.

Timing uses synchronized calls to an already compiled executable. Lowering,
compilation, and warmup are recorded separately and excluded from the reported
median actor-update time. Every timed call starts from the same state and batch,
so the benchmark measures the cost of one actor-update dispatch rather than a
training trajectory.

The primary memory field is JAX backend `peak_bytes_in_use`. It is the fresh
worker's process-lifetime backend high-water mark through data transfer, state
initialization, compilation, warmup, and trials. It is **not** an isolated
steady-state actor-call peak. The manifest records both the absolute high-water
mark and a baseline-current-adjusted value computed as
`max(0, absolute_peak_bytes - runtime_baseline_current_bytes)`. If the backend
does not expose the required counters, the default is to stop. The explicit
process-RSS fallback is a smoke diagnostic and is never accelerator-memory
evidence.

JAX execution is asynchronous, so the worker blocks every returned JAX leaf
before stopping a timer, following the official
[JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
Backend memory-counter support is device dependent; the profiler therefore
checks
[`Device.memory_stats()`](https://docs.jax.dev/en/latest/_autosummary/jax.Device.html)
and fails closed instead of inventing missing accelerator measurements. JAX GPU
preallocation is disabled for this protocol
and recorded in the environment; see the
[JAX GPU memory notes](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

## 1. Preview without side effects

Run from the repository root with the exact interpreter that will execute all
four workers:

```bash
export PY=/absolute/path/to/the/python-environment/bin/python
"$PY" scripts/diagnostics/profile_actor_cost.py
```

Review the printed K grid and requested configuration. Supplying dataset or
output paths without `--execute` still performs only this preview.

## 2. Execute on one locked GPU

Use a local HDF5 dataset and an absent or empty output directory outside this
repository. Select one physical GPU by index or UUID and ensure no unrelated
workload uses it.

```bash
"$PY" scripts/diagnostics/profile_actor_cost.py \
  --execute \
  --dataset-path /absolute/path/to/offline_dataset.hdf5 \
  --output-dir /absolute/external/path/bar_actor_cost \
  --accelerator 0 \
  --confirm-exclusive-accelerator
```

A release-eligible GPU run additionally requires a clean Git worktree, a
resolved accelerator UUID and driver record, exactly one JAX-visible device,
and working backend memory counters. The tool never downloads a dataset. Do not
reuse a completed output directory.

The orchestrator resolves `--accelerator` through `nvidia-smi`, then exposes
the resolved physical UUID, rather than a potentially remapped ordinal, through
the child CUDA visibility variable. Each worker must resolve exactly one JAX
GPU. A UUID reported by JAX is recorded with its JAX attribute as the
observation source; when JAX exposes no UUID, the field remains null and the
separate resolved-UUID visibility binding is retained. The verifier checks the
driver inventory, UUID binding, worker environment, unique worker identities,
and harness lock. It cannot independently prove the absence of an unrelated
process; `--confirm-exclusive-accelerator` remains an operator attestation.

For a non-evidentiary CPU wiring check only:

```bash
"$PY" scripts/diagnostics/profile_actor_cost.py \
  --execute \
  --platform cpu \
  --allow-cpu-smoke \
  --memory-fallback process-rss \
  --dataset-path /absolute/path/to/offline_dataset.hdf5 \
  --output-dir /absolute/external/path/bar_actor_cost_cpu_smoke \
  --confirm-exclusive-accelerator
```

The resulting status is `cpu_smoke`, not a measured accelerator result.
For this smoke path, the current CLI reuses the confirmation flag as an
isolation acknowledgement; it does not claim that a CPU run used an accelerator.

A GPU run with `--memory-fallback process-rss` is similarly labeled
`fallback_smoke`.

## 3. Verify independently

```bash
"$PY" scripts/diagnostics/verify_actor_cost.py \
  /absolute/external/path/bar_actor_cost
```

The verifier does not import the profiler. For a measured run, the current
repository must contain the recorded Git object; source bytes are read from
`git cat-file blob <revision>:<path>`, so the checkout itself need not be moved
back to that revision. The recorded dataset must still exist. If an identical copy
or another clone is used, make both overrides explicit:

```bash
"$PY" scripts/diagnostics/verify_actor_cost.py \
  /absolute/external/path/bar_actor_cost \
  --source-repo /absolute/path/to/repository-with-recorded-commit \
  --dataset-path /absolute/path/to/identical-offline_dataset.hdf5
```

The verifier hashes the actual Git blobs, dataset bytes, `RUN_CONTEXT.json`,
and result files. It recomputes trial means and medians, K=1 ratios, memory
derivations, and cross-process invariants; requires unique fresh-worker
identities; and checks that K is the only experimental field that changed.
Treat a directory as evidence only when this command succeeds and reports
`evidence_status: measured`. CPU and process-RSS fallback bundles remain
`cpu_smoke` and `fallback_smoke`, respectively, even when their integrity
checks pass.

The directory contains `RUN_CONTEXT.json`, one `K=<K>.json` per depth, and
`MANIFEST.json`. Keep the raw JSON with any compact table derived from it; do
not commit local datasets or unreviewed benchmark work files.

## Interpretation boundary

The benchmark supports a measured cost statement for one fixed
hardware/software/data shape. It does not establish a hardware-independent
constant, end-to-end training time, or memory complexity theorem. Report raw
milliseconds and bytes alongside K=1 ratios, and retain the analytical `O(K)`
actor-work statement separately.
