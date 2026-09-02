# P1 target-value audit runbook

## Status and scope

This repository contains the P1 postprocessing code and tests. It does not
contain a completed 270-checkpoint inventory or P1 scientific results. Do not
describe the audit as admissible until a new external output bundle has
`MANIFEST.json.status == "analysis_complete"`,
`MANIFEST.json.finite_value_gate == true`, `VERIFY.json.pass == true`,
`VERIFY.json.artifact_integrity_pass == true`, and
`VERIFY.json.scientific_inclusion_gate_pass == true`.

The audit is CPU-only postprocessing. It does not launch or resume training and
is intentionally not wired into `run_host.sh`. It accepts only the fixed grid of
9 tasks, 5 budgets, 2 seeds, and 3 methods at step 1,000,000. Missing, duplicate,
or incompatible checkpoints block the whole run; no nearby checkpoint or method
is substituted.

The locked tasks are HalfCheetah, Hopper, and Walker2d with `medium-v2`,
`medium-replay-v2`, and `expert-v2` data for each. The methods are the
single-actor TD3+BC baseline (`td3`), BAR-P3 (`p3`), and BAR-P4 (`p4`). Budgets
are `T = {4,7,10,14,20}` and seeds are exactly `0` and `1`.

## Required paths

Use an output directory outside the repository. Scientific artifacts are
create-only, so start with a fresh `OUT`. If preflight or verification blocks,
preserve that bundle as evidence and choose another fresh directory after fixing
the input problem.

```bash
export LAB_ROOT=/absolute/path/to/this/repository
export RESULTS_ROOT=/absolute/path/to/the/results/root
export DATA_DIR=/absolute/path/to/the/d4rl-hdf5-directory
export OUT=/absolute/path/outside/the/repository/p1-audit-YYYYMMDD
export PY=/absolute/path/to/the/python-environment/bin/python
```

`PY` must provide the repository dependencies, including JAX, Flax, Optax,
NumPy, and HDF5 support. `RESULTS_ROOT` must contain the exact explicit paths
listed by dry-run, together with each run's `config.json` and `eval.csv`.

## 1. Freeze the expected grid without reading data

```bash
cd "$LAB_ROOT"
"$PY" scripts/diagnostics/run_p1_target_value_audit.py \
  --mode dry-run \
  --root "$RESULTS_ROOT" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT"
```

This creates `EXPECTED_GRID.json` and `DRY_RUN_STATUS.json`. It needs no
checkpoints or datasets and performs no analysis.

## 2. Run the atomic preflight

```bash
"$PY" scripts/diagnostics/run_p1_target_value_audit.py \
  --mode preflight \
  --root "$RESULTS_ROOT" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT"
```

Success requires exactly 270 compatible final checkpoints. Preflight also binds
the configs, external score rows, weights, dataset normalization, common batches
and noise, optimizer states, source files, package versions, and current Git
revision/dirty state. The fixed config contract includes normalization, implicit
integration, the 1,000,000-step horizon, 10 evaluation episodes, batch size 256,
discount 0.99, Polyak factor 0.005, policy/noise settings, actor-update frequency
2, learning rate 3e-4, and 8 jitted updates. Historical configs may omit
`method`; when present it must be `bar`. Evaluation cadence is not locked because
it differs across the eligible checkpoint families. Every stored actor step must
equal both its Adam count and the schedule-derived value 500,000. If exact
optimizer reconstruction is unsupported, P1-C is unsupported and the entire
preflight exits 2; it is never approximated with a fresh optimizer.

Exit 2 creates `CHECKPOINTS_UNRESOLVED.json` and `PREFLIGHT_BLOCKED.json`, but no
`CHECKPOINTS.json`, frozen protocol, or scientific CSV. A successful preflight
creates `CHECKPOINTS.json`, `FROZEN_PROTOCOL.json`, `PREFLIGHT_READY.json`, and 9
frozen common-batch NPZ files. A directory containing either blocked marker is
permanently ineligible for preflight or analysis; preserve it and use a fresh
`OUT`.

The source hashes show the code currently available to the audit. Historical
training identity remains unproved because the checkpoints do not embed a Git
revision.

## 3. Run CPU postprocessing

Use the same unchanged ready bundle and inputs:

```bash
"$PY" scripts/diagnostics/run_p1_target_value_audit.py \
  --mode run \
  --root "$RESULTS_ROOT" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT"
```

`run` repeats preflight and requires the canonical serialized content of the
frozen inventory and protocol to match before analysis. Before creating any
scientific output, it resolves the active JAX backend and aborts unless it is
actually `cpu`. It then creates these scientific outputs:

| Artifact | Expected rows/files |
| --- | ---: |
| `td_target_value.csv` | 540 rows: 270 cells × 2 critic scopes |
| `same_next_state_geometry.csv` | 270 rows |
| `comparator_residuals.csv` | 450 rows, only k ≥ 2 |
| `raw/**/*.npz` | 270 files |
| `method_contrasts.csv` | 580 rows, including 9 task means per contrast and a task-equal aggregate |
| `geometry_correlations.csv` | 9 rows |
| `residual_aggregates.csv` | data-dependent stability groups |
| `MANIFEST.json` and `RUN_STATUS.json` | one each |

All bundle references to common batches, raw arrays, and CSVs are relative to
`OUT`, so the finished bundle can be relocated intact. Do not commit the raw NPZ
files or generated CSV/JSON bundle to Git.

## 4. Verify independently

Run the artifact-only verifier once after `run`:

```bash
"$PY" scripts/diagnostics/verify_p1_target_value_audit.py \
  --out-dir "$OUT"
```

The verifier does not import the analysis runner. It independently rechecks the
fixed keys and counts, complete per-record metadata/config/optimizer semantics,
the CPU-complete manifest, hashes, frozen formulas, clipping, smoothed and
no-noise TD-target arrays, geometry, P1-C residual fields, and the derived
`method_contrasts.csv`, `residual_aggregates.csv`, and
`geometry_correlations.csv`. The executing verifier must hash to the verifier
source fingerprint frozen in provenance. `VERIFY.json` binds the manifest,
inventory, protocol, and verifier hashes and is created without overwriting an
existing verification report. Inclusion requires all five manifest/verification
fields listed in **Status and scope** to have the stated values.

This is artifact-only independence: the verifier recomputes serialized
arithmetic and integrity relationships but does not reexecute checkpoint
networks or independently prove that the runner produced the raw arrays from
the declared checkpoints. The manifest fingerprints those inputs and code; it
does not turn this verifier into a second network evaluator.

## Interpretation boundaries

The runner requests the JAX CPU platform before importing JAX, asserts the
resolved backend on the scientific path, and records it in the manifest. The
value metric is an operational readout of the stored learned target critic, not
a critic-accuracy estimate. Report within-run target-critic results separately
from the matched common TD3 target-critic sensitivity. The same
frozen audit batch and common noise are used within each task; it is not a
holdout and supports no broader task-generalization claim. The no-noise columns
measure smoothing sensitivity on the same critic, actions, and rows.

P1-C is a post-hoc final-checkpoint intervention: one Adam step reconstructed
from the stored final parameters, optimizer state, and step count on the fixed
audit batch with the online critic frozen. It is not the historical live
training update, minibatch, or pre-update state. Its residual and slack are
sampled feasible-comparator diagnostics, not a global optimality gap or an
`epsilon_k` bound. Seeds are marginal run replicates inside this fixed
nine-task, five-budget grid.
