# K=4 recenter versus fixed_ref at high T

Status: prepared, no training launched. Protocol: [RECENTER_FIXED_REF_K4_PROTOCOL.json](RECENTER_FIXED_REF_K4_PROTOCOL.json).

Use the same shared-driver code for both branches. Each matched environment/T/seed
pair stays on one host: svcho gets seeds 0,2 on even-index environments and 1,3
on odd-index environments; choi gets the complementary seeds. This alternates
host ownership across nine environments so each host sees every environment,
every T, and all seed labels. The nine supported locomotion environments are
listed in the protocol; this is not a 12-task grid.

| Stage | T | Each host | Both hosts |
| --- | --- | ---: | ---: |
| primary | 4, 7 | 36 pairs / 72 full runs | 72 pairs / 144 runs |
| boundary | 10 | 18 pairs / 36 full runs | 36 pairs / 72 runs |
| all | 4, 7, 10 | 54 pairs / 108 full runs | 108 pairs / 216 runs |

Each run trains 1M critic updates and 500k updates of each of four actors.
At a given T, h=T/4 in both branches. The primary question is the signed
paired difference d4rl_pi4(recenter) - d4rl_pi4(fixed_ref) at T=4,7;
T=10 tests the high-T boundary. Earlier standalone K=4 results are not
members of these matched pairs. The K=3 scripts and P0 data_anchor protocol
are separate and unchanged.

## Setup on svcho and choi separately

Execute the following on each host. Set HOST to the matching literal host
label, choose an interpreter with JAX/Flax/Optax/Gymnasium, a directory
containing the nine D4RL HDF5 files, and an output filesystem with sufficient
space. Replace the example GPU ID with the GPU assigned on that host. The
worktree avoids changing the active checkout.

```bash
HOST=svcho  # on choi use HOST=choi
SRC="$HOME/MPI_sweep"
WORK="$HOME/MPI_sweep_recenter_k4"
PY="$(command -v python)"
DATA="$SRC/data"
SAVE="/raid/$USER/MPI_store/recenter_fixed_ref_k4"
GPU=0

git -C "$SRC" fetch origin experiment/recenter-fixed-ref-k4-high-t-20260925
git -C "$SRC" worktree add --detach "$WORK" FETCH_HEAD
cd "$WORK"
git rev-parse HEAD
test -d "$DATA"
mkdir -p "$SAVE"
"$PY" -m pytest -q tests/test_shared_driver.py
"$PY" scripts/experiments/run_recenter_fixed_ref_k4.py \
  --host "$HOST" --pilot --python "$PY" --data-dir "$DATA" --save-dir "$SAVE"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u scripts/experiments/run_recenter_fixed_ref_k4.py \
  --host "$HOST" --pilot --execute --python "$PY" \
  --data-dir "$DATA" --save-dir "$SAVE"
```

The pilot executes two 1024-step runs and validates checkpoint identity.
It writes `$SAVE/PILOT_VERIFIED.json`, tied to the code, host, Python path
and data directory. The pilot is only a wiring check. Inspect its device
assignment and its `pair_verified` output. If the worktree already exists,
inspect it before reusing; do not force-reset an active checkout.

## Stage 1 (both machines)

The preview prints 36 pairs / 72 runs on each host. Start a worker only on
an allocated idle GPU. Keep one worker per assigned GPU.

```bash
"$PY" scripts/experiments/run_recenter_fixed_ref_k4.py \
  --host "$HOST" --stage primary --python "$PY" \
  --data-dir "$DATA" --save-dir "$SAVE"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u scripts/experiments/run_recenter_fixed_ref_k4.py \
  --host "$HOST" --stage primary --execute --python "$PY" \
  --data-dir "$DATA" --save-dir "$SAVE" > "$SAVE/primary.log" 2>&1 &
echo "primary PID=$!"
```

Wait for both machines to finish stage 1. Inspect `primary.log` for 36
`pair_verified` lines on each host. If a worker stops, inspect the log and
resume only on the same host, same code/interpreter/data/output directory.
The training script resumes from its own latest checkpoint. Never mix one
arm of a pair with a run from another host.

## Stage 2 (both machines after primary)

The preview prints 18 pairs / 36 runs per host. The output is the same
`$SAVE/full`; the T=10 run names remain disjoint.

```bash
"$PY" scripts/experiments/run_recenter_fixed_ref_k4.py \
  --host "$HOST" --stage boundary --python "$PY" \
  --data-dir "$DATA" --save-dir "$SAVE"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -u scripts/experiments/run_recenter_fixed_ref_k4.py \
  --host "$HOST" --stage boundary --execute --python "$PY" \
  --data-dir "$DATA" --save-dir "$SAVE" > "$SAVE/boundary.log" 2>&1 &
echo "boundary PID=$!"
```

The pair verifier checks bitwise first-actor, second-actor, critic, target,
optimizer, RNG and normalization identity, four matched actor-update
counts, and the two final scores. It aborts on a failed/missing pair.
Save both hosts' logs and full run directories for analysis; report missing
pairs and full signed task/T effects, including failures.
