# Frozen-critic extra hop-actor optimization

Built: 2026-09-06T01:20:53+09:00

Understood as: freeze the 1M critic and μ_{k-1} on Hopper-medium/expert
T=10 MART snapshots, extra-optimize μ_k with the JKO loss, and test
whether snapshot ΔL>0 shrinks. Not a return experiment.

## Protocol

- Extra Adam steps: 10000 at lr=3e-4, batch 256, fresh optimizer.
- Critic and previous actor are not updated.
- ΔL is measured on the same 4096 diagnostic dataset states as hop_objective.csv.
- Optimization samples the full D4RL observations, checkpoint-normalized.
- One JAX process per GPU via `launch_p0_frozen_hop_extra_opt.sh` (`nohup`, AMO left running).
- This dump is Hopper T=10 seed 0 only. Seed 1 was not launched.

## Snapshot vs extra-opt

### hopper-medium

- hops with ΔL≥0 at step 0: 3/3.
- hops with ΔL<0 after 10000 steps: 3/3.
- mean ΔL +0.001439 → -0.0006378.
- mean Q-gain/move-cost after extra steps: 1.216.

- seed 0 mu1→mu2: ΔL +0.0008746 → -0.0009771, ratio 0.790 → 1.321, grad 0.0113, move from snapshot RMS 0.0477.
- seed 0 mu2→mu3: ΔL +0.001204 → -0.0006372, ratio 0.738 → 1.221, grad 0.00898, move from snapshot RMS 0.0462.
- seed 0 mu3→mu4: ΔL +0.002238 → -0.0002991, ratio 0.532 → 1.107, grad 0.00952, move from snapshot RMS 0.0522.

### hopper-expert

- hops with ΔL≥0 at step 0: 3/3.
- hops with ΔL<0 after 10000 steps: 0/3.
- mean ΔL +0.003756 → +0.001054.
- mean Q-gain/move-cost after extra steps: 0.288.

- seed 0 mu1→mu2: ΔL +0.004563 → +0.001268, ratio 0.077 → 0.267, grad 0.0139, move from snapshot RMS 0.0614.
- seed 0 mu2→mu3: ΔL +0.003169 → +0.0009598, ratio 0.125 → 0.301, grad 0.00917, move from snapshot RMS 0.0496.
- seed 0 mu3→mu4: ΔL +0.003536 → +0.0009337, ratio 0.099 → 0.294, grad 0.00591, move from snapshot RMS 0.0527.

## Read

If extra steps drive ΔL below 0, snapshot non-improvement was optimizer
slack on this frozen objective. Medium seed 0 is that case for hops 2–4.
Expert ΔL shrinks but stays positive after 10k steps, so slack does not
fully explain that snapshot. Extra-opt does not by itself prove return
recovery. This does not reconstruct per-minibatch decreases during joint
training.

## Figures

- `fig_deltaL_vs_extra_steps.png`

