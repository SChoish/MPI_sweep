# TD3+BC K=3: recenter versus fixed reference

Status: **prepared; no training launched or watcher queue registered**.
Protocol: [`RECENTER_FIXED_REF_K3_PROTOCOL.json`](RECENTER_FIXED_REF_K3_PROTOCOL.json).
This is a separate question from the existing K=4 `P0_SHARED_DRIVER_PROTOCOL.json`:
that file compares `recenter` to `data_anchor` and remains unchanged.

## Comparison

`train_td3bc.py --method shared --integrator implicit --mpi-steps 3` trains
three actors in either branch. The online critic and its target use the first
actor only. At each actor-update step both runs use the same dataset minibatch,
RNG seed, critic update, first-actor update, and two downstream actor updates.
With total horizon `T`, all three actor objectives use `h=T/3`.

| Branch | hop 2 reference | hop 3 reference | final score column |
| --- | --- | --- | --- |
| `recenter` | freshly updated `pi_1` | freshly updated `pi_2` | `d4rl_pi3` |
| `fixed_ref` | freshly updated `pi_1` | freshly updated `pi_1` | `d4rl_pi3` |

`fixed_ref` still updates `pi_2` once per scheduled actor step, even though
hop 3 does not use it as reference. All three actors therefore receive the
same **number** of optimizer updates in both arms; fixed_ref's pi_2 update
does not influence its final actor. Its additional computation is nevertheless
real and should be reported in runtime measurements rather than discarded.

The fixed grid is nine D4RL locomotion tasks × `T={0.4,2.5,10}` × seeds
`0,1,2,3` = 108 paired cells, 216 one-million-step training runs. The
T grid and all cells were specified before these new outcomes exist; earlier
K=4 observations and other actor experiments are exploratory motivation,
not independent confirmation. Use final checkpoint, mean-action evaluation,
ten episodes per run. Compare pairwise `d4rl_pi3` and check first-actor
`d4rl_score` equality. Present every T rather than selecting a winner.

## Host preparation and pilot

Example paths for **ext_csv**, to be used only after its existing queue drains
and its assigned GPU is available. Run from its own host, in the interpreter
that can import JAX, Flax, Optax and Gymnasium. The pilot is 1024 training
updates, then 10 episodes for each actor. This verifies wiring, not return.

```bash
cd /home/ext_csv/MPI_sweep
git status --short --branch
git rev-parse HEAD
PY=/home/ext_csv/miniconda3/envs/offrl/bin/python
SAVE=/raid/ext_csv/MPI_store/recenter_fixed_ref_k3
"$PY" -m pytest -q tests/test_shared_driver.py
"$PY" scripts/experiments/run_recenter_fixed_ref_k3.py \
  --pilot --execute --python "$PY" --save-dir "$SAVE" \
  --data-dir /home/ext_csv/MPI_sweep/data
```

The pilot runs `recenter` then `fixed_ref` in isolated `pilot/` output under
`$SAVE` and automatically verifies the checkpoint pair. Inspect the printed
`pair_verified` JSON, pilot `eval.csv` files and device assignment. A passing
pilot writes `PILOT_VERIFIED.json`; full execution rejects a missing gate or
changed source/interpreter/data paths. The runner
uses the caller's `CUDA_VISIBLE_DEVICES` and does not acquire a GPU or register
a queue. If the installation has only a subset of the allotted devices,
restrict the invocation to the allocated GPU via that variable.

## Full queue, after pilot passes

The default invocation prints the 216-run count and first pair of commands;
it does not start training. Use `--execute` on
the host with available resources after the pilot gate. To run a
bounded subset, use any combination of `--env`, `--tau`, and `--seed`; these
options reject values outside the protocol grid. Adjacent branches are run
sequentially for each matched cell. Existing interrupted runs resume from
their own checkpoint. The runner verifies identity after each completed pair.

```bash
"$PY" scripts/experiments/run_recenter_fixed_ref_k3.py \
  --python "$PY" --save-dir "$SAVE" \
  --data-dir /home/ext_csv/MPI_sweep/data

# Use this as the queue worker command only after the active queue has drained,
# the pilot has passed, and the same command is registered in the host watcher:
"$PY" scripts/experiments/run_recenter_fixed_ref_k3.py \
  --execute --python "$PY" --save-dir "$SAVE" \
  --data-dir /home/ext_csv/MPI_sweep/data
```

The host watcher registration must point to the **actual** host configuration,
contain both `train_td3bc.py` and the new queue command in its process/queue
patterns, use a CPU and a GPU wrapper for the same saved results, and preserve
the existing active queue until it drains. `server_management` owns the
registration; this runbook does not change its configuration or start
training. The runner's full mode saves every 100,000 updates for soft-stop
resume; the pilot writes at 1024. `--execute` is explicit. Record host,
git revision, dataset fingerprint, GPU ID, start/end time and any failures
with each final score. The training script already writes `PROVENANCE.json`.

The full comparison keeps the normal 5,000-update evaluation schedule and
the existing critic/actor update order. Passing the pilot only validates a
short trajectory; it does not prove 1M-step runtime correctness or an MPI
effect. As a primary analysis, use paired final differences per seed,
four-seed summaries per environment × T, and an equally weighted average of
the 27 cell means. Report missing pairs and non-finite runs explicitly.
