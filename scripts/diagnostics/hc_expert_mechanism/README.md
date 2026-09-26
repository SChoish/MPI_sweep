# HalfCheetah-expert: why does re-centering fail after actor 2?

## Current evidence

In the six matched cells (T=4,7,10; training seeds 0,2), the critic and
actors 1 and 2 are identical across branches. Re-centered and fixed-reference
return first diverge at actor 3. The final re-centered policy scores about
35--43 D4RL points lower than fixed reference. A previous MC test used initial
states and followed actor 1 after the first action; it does not check the
states actually visited by the failing actors. Dataset-state critic Q rose by
about 1 while endpoint return collapsed. Dataset-state action saturation rose.

## Smallest informative diagnostic

1. Re-evaluate actors 1, 3, and 4 of both matched branches with 10 identical
   episode seeds. Compare velocity, reward, and near-boundary action frequency
   by 100-step blocks. Verify actor-1 and actor-4 scores against `eval.csv`.
2. At times 0, 50, 150, 300 of the first episode of each branch and actor
   3/4, clone the *actual visited simulator state*. Check that cloning
   reproduces the next observed transition. At each state, evaluate both
   actor actions, their common critic Q1/Q2 values, and the discounted MC
   return of **one chosen action followed by common actor 1**. Compare the
   difference between actions within the same state, not the absolute value
   of Q against a whole-actor deployment return.
3. The independent experimental unit is a training seed for each T. Episode
   seeds and states are repeated measurements. Keep the six cells distinct;
   this is an exploratory diagnosis of HalfCheetah-expert, not a new benchmark.

| Prediction | Interpretation if observed | Observation that weakens it |
| --- | --- | --- |
| On visited states the critic favors recentered actions but MC favors fixed-reference actions | Local Q action ranking may favor harmful moves in the failing chain | Critic and MC consistently rank actions the same on those states |
| MC favors re-centered first actions but their full rollouts lose reward | Loss arises from repeated closed-loop decisions and changed visited states; one-step MC does not capture their long-term effect | MC already disfavors re-centered actions early |
| The falling branch shifts to low-velocity states or early failure together with rising saturated actions | Diagnoses where in the rollout failure develops; saturation is a candidate mediator | Stable velocity and action pattern before the loss |

Even perfect agreement between Q and one-action MC does not certify Q at
states that the selected snapshots miss; the two branches visit different
state distributions. Saturation and return can covary without proving that
saturation caused the failure. If these tests remain ambiguous, the next
intervention is a **paired action-cap pilot** on frozen actor-4 checkpoints:
evaluate `clip(pi4(s), -0.95, 0.95)` for both branches at the same episode
seeds, comparing its within-branch change in return against the unmodified
policy. Recovery only in the re-centered branch supports a role for actions
near the boundary, while an unchanged gap weakens that explanation. A cap also
changes action magnitudes, so recovery alone does not isolate clipping as a
unique mechanism.

## Run on `choi` (no training, no checkpoint changes)

Copy `diagnose_hc_expert.py` and `summarize.py` to `choi` and run with the
same Python interpreter used for the matched K=4 training runs. Point
`--repo` to its source checkout; `--checkpoints` is the directory containing
all six `halfcheetah-expert-v2_tau...` run directories. For example:

```bash
python diagnose_hc_expert.py \
  --repo /home/choi/MPI_sweep_recenter_k4 \
  --checkpoints /home/choi/MPI_store/recenter_fixed_ref_k4/full \
  --output /home/choi/MPI_store/hc_expert_mechanism_pilot
python summarize.py /home/choi/MPI_store/hc_expert_mechanism_pilot
```

The script validates source checkpoint equality, the Gymnasium clone's next
transition, and observed scores versus the original `eval.csv`. It aborts on
a mismatch and writes partial CSVs after a fully evaluated T/seed pair.
`counterfactual_mc.csv` contains state-level evidence. The four MC timepoints
and first evaluation episode are a bounded exploratory pilot. If a decision
depends on a noisy or mixed pattern, repeat with more evaluation episodes and
times **without selecting only favorable cells**.
