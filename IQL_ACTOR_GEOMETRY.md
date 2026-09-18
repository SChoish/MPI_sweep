# IQL critic with three actor / MPI geometry choices

This comparison runs all three paths in **one training job with one Q/V pair**.
The original TD3+BC trainer remains available through `--algorithm td3bc`.

| CLI variant | Base actor | Refinement |
| --- | --- | --- |
| `awr_gaussian_fr` | Diagonal Gaussian, advantage-weighted negative log likelihood | Expected Q + closed-form Gaussian FR |
| `qbc_gaussian_w2` | Park et al. DDPG+BC: raw Gaussian with fixed std 1, negative Q at clipped mean + dataset NLL | Negative expected Q + Gaussian W2, updating mean **and** standard deviation |
| `qbc_deterministic_w2` | Deterministic, TD3+BC actor loss with detached Q normalization + mean-squared BC | Negative Q + Dirac W2 |

## Objective and update contract

Write `Q = min(Q1_target, Q2_target)`. Each iteration updates V by expectile
regression on **dataset actions**, extracts/updates all actors using this same
target Q and updated V, then fits both online Q networks to
`r + gamma * not_done * V(next_state)` and updates target Q by EMA. This is
the update order in the [original IQL learner](https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py).
Actors never enter Q/V targets. Fixed initialization keys and a separate
per-step actor RNG ensure changing actor variants, K, or evaluation frequency
does not change the critic/minibatch stream on the same numerical stack.

The default base losses are:

```text
AWR Gaussian: E_D[min(exp(beta * (Q(s,a_D)-V(s))), 100) * -log N(a_D; m, sigma^2)]
DDPG+BC Gaussian: -E_D[Q(s,clip(m,-1,1))]
                  + bc_coef * E_D[-log N(a_D; m, I)]
TD3+BC Dirac:    -lambda * E_D[Q(s,m)] + mean_{batch,coordinate}((m-a_D)^2)
                lambda = stop_gradient(td3bc_alpha / (mean_batch(abs(Q(s,m))) + 1e-6))
```

The DDPG+BC base follows Equation (6) and Appendix C.6.1 of
[Park et al., *Is Value Learning Really the Main Bottleneck in Offline RL?*](https://arxiv.org/html/2406.09329v2#S5.SS1.SSS2):
Q is evaluated at the clipped **mean**, not a sampled action. The BC term is
the negative log density of **dataset** actions under the raw Gaussian, summed
over action dimensions. The mean has no tanh, and std is fixed at 1 with no
trainable std parameters in the base. Clipping affects only the Q input, not
the mean in the BC density. Thus NLL = 0.5 * sum_j((m_j-a_D,j)^2) + constant;
the factor 0.5 and dimension sum are part of the BC coefficient convention.
`--bc-coef` is the paper's alpha (default 1), independent of refinement tau.
This replaces the previous expected-squared-error, sampled-Q base.

The deterministic path uses the TD3+BC **actor-loss form**, with
`--td3bc-alpha 2.5` by default. Its Q comes from the common IQL min-target-Q,
not the online Q1 used by the original full TD3+BC algorithm. It uses no
TD3 critic targets, target-policy smoothing, or delayed actor schedule.
Its BC coefficient is 1; `--bc-coef` controls only Gaussian DDPG+BC.

The existing AWR base retains its bounded tanh mean and learned
state-dependent diagonal std, bounded in log space by `[-5, 2]` and initialized
at log-std `-1`. This is the original shared-critic comparison's AWR policy
family, not the fixed-std AWR used in the bottleneck paper's data-scaling study.
The AWR loss is a KL-derived extraction/projection objective; it is not labeled
an exact Fisher-Rao proximal step.

**MPI is an extension, not a reproduction of the bottleneck paper.** For the
DDPG+BC path, all refinement actors start with the same raw mean as the base
and std 1, but have learnable state-dependent std heads. Their Q term is
expected Q under reparameterized samples, so variance can improve as originally
requested. `--log-std-init` applies to the AWR bank; Gaussian DDPG+BC refinement
always starts at log-std 0 to match its base. The std bounds apply to both
refinement families and must contain 0 for the Gaussian DDPG+BC path.

Each bank contains a base actor `pi_0` and **K additional** refinement actors.
`--tau T` specifies only the total refinement horizon; `h = T/K`. In contrast,
the historical TD3+BC launcher counts its dataset-anchored actor inside K.
`--hops 0` / `--mpi-steps 0` train just the three base actors; tau has no effect
in this case. Each training iteration updates the base actor once and then
the K persistent refinement actors sequentially:

```text
reference_k = stop_gradient(distribution of updated pi_(k-1)(s))

W2: loss_k = -E[Q(s,a)] + E[sum_j((m-m_ref)^2 + (sigma-sigma_ref)^2)] / (2h)
FR: loss_k = -E[Q(s,a)] + E[4 * acos(BC(pi, pi_ref))^2] / (2h)
```

The deterministic path uses `sigma = sigma_ref = 0` and evaluates Q at m.
The two Gaussian paths estimate expected Q using reparameterized antithetic
normal samples (`--mc-samples 8`), preserving the gradient through sigma.
The FR penalty uses the closed form from MPI Appendix A.3:

```text
BC = product_j sqrt(2*sigma_j*sigma_ref_j / (sigma_j^2+sigma_ref_j^2))
               * exp(-(m_j-m_ref_j)^2 / (4*(sigma_j^2+sigma_ref_j^2)))
d_FR^2 = 4 * acos(BC)^2
```

This is the exact FR distance of the full density space evaluated between
Gaussian endpoints. It is symmetric and bounded by pi squared after squaring.
It is not the intrinsic geodesic distance constrained to remain in the Gaussian
submanifold. The Bhattacharyya coefficient is combined across all dimensions
**before** applying acos; squared marginal FR distances are not summed.

The implementation computes `D_B = -log(BC)` using nonnegative `log1p` terms
and evaluates `acos(exp(-D_B))` through an equivalent atan2 expression. Its
custom derivative uses the removable limit at `D_B=0` (derivative of squared
FR is 8) and a near-zero series for numerical stability. The actual distance
value is the closed form everywhere; no KL approximation is used in refinement.

All refinement actors start from the same initial base distribution with
separate optimizer states. They remain persistent during training. Each hop
performs `--inner-updates` Adam updates (default 1) on its proximal loss,
holding Q, the predecessor reference, and MC noise fixed throughout these
inner updates. The reference is recentered at the next hop, and gradients
never propagate into the predecessor or critic. This is amortized neural
proximal refinement; it is not an exact JKO solver, and it does not guarantee
objective descent or return improvement. Increasing `inner-updates` changes
the compute budget and is recorded in the run identity.

## Distance, action bounds, and evaluation

The default `--metric-reduction sum` matches the unnormalized Euclidean W2
formula above. `--metric-reduction mean` divides W2 and the final squared FR
distance by the action dimension. It does not change base losses: Gaussian
NLL always sums dimensions, and TD3+BC always uses coordinate-mean MSE. The legacy TD3+BC
trainer uses mean-squared transport, so nominal tau values are not directly
comparable across these conventions.

In MPI, `--q-action-transform identity` evaluates Q at raw Gaussian
samples, exactly matching the displayed expected-Q objective. These samples
can leave `[-1,1]`; per-hop `out_of_bounds_fraction` is logged. Selecting
`--q-action-transform clip` instead optimizes `E[Q(s,clip(m+sigma*eps))]`.
The Gaussian W2 / FR penalty still applies **before** clipping. It is not
claimed to be the exact W2 / FR between clipped action distributions.
The Gaussian DDPG+BC base always uses clipped-mean Q independently of this flag.

Simulator actions are always clipped to `[-1,1]`. `--eval-mode both` (default)
reports separate mean-action and sampled-policy returns for Gaussian actors;
deterministic actors have one `mean` row. Evaluation RNG is independent from
training RNG. Base and final actors are evaluated by default; `--eval-hops all`
also evaluates intermediate actors. Mean-only evaluation can hide the effect
of changing variance.

Gaussian DDPG+BC and MPI Q are unnormalized by default, as in the displayed
objectives. `--iql-q-scale-norm` is an optional departure from the paper's base
objective: Gaussian DDPG+BC uses dataset-action `mean(abs(Q))`; refinement uses the frozen
predecessor's sampled `mean(abs(Q))`. This changes the effective energy/time
scale and is recorded. The historical `--q-scale-norm` launcher flag only
controls TD3+BC; use `--iql-q-scale-norm` for this comparison.
The deterministic base always uses its detached TD3+BC lambda, independently
of this flag and `--metric-reduction`.

Other defaults: expectile 0.7, AWR inverse temperature 3, BC coefficient 1,
constant actor/Q/V learning rates 3e-4, two 256-unit ReLU layers, gamma .99,
Q-target EMA .005, and batch size 256. Rewards are raw unless
`--reward-scale` is specified; state normalization follows the repository
loader (`std + 1e-3`) unless `--no-iql-normalize-state` is selected. These
are a controlled common-critic protocol, not a reproduction of the bottleneck
paper's full data-scaling experiment, architecture (including LayerNorm),
preprocessing, or hyperparameter selection. The paper's project page does not
link a dedicated implementation; the port follows its equation and appendix.
Supported datasets/evaluation environments remain those in `d4rl_data.py`
and the existing Gymnasium MuJoCo evaluator.

## Commands

Install with `pip install -e ".[test]"` (install the appropriate GPU JAX wheel
first on a GPU host). Preview one shared-critic job:

```bash
python launch_mpi_sweep.py --algorithm iql --hops 4 \
  --domains hopper --datasets medium-replay --taus 1 --seeds 0 --dry-run
```

Run it on GPU 0:

```bash
python launch_mpi_sweep.py --algorithm iql --hops 4 \
  --domains hopper --datasets medium-replay --taus 1 --seeds 0 --gpus 0 \
  --save-dir results/iql_geometry --log-dir logs/iql_geometry
```

Sweep refinement horizons with all three variants, four seeds, and two GPUs:

```bash
python launch_mpi_sweep.py --algorithm iql --hops 4 \
  --taus "0.1 0.5 1 2 5" --seeds "0 1 2 3" --gpus "0 1" \
  --save-dir results/iql_geometry --log-dir logs/iql_geometry
```

The launcher retains the existing nine-task `medium/medium-replay/expert`
grid. Change `--domains` / `--datasets` to select a supported subset. To run
one actor family alone, pass e.g. `--variants qbc_gaussian_w2`; separate runs
still use the same Q/V initialization and data RNG when other settings match.

Direct entrypoint (K can also be 0 for baselines):

```bash
mpi-iql-train --env hopper-medium-replay-v2 --mpi-steps 4 --tau 1 --seed 0
```

## Outputs and restart

Run directories include environment, T, K, seed, and a hash of the actor,
geometry, training, and evaluation settings. They cannot collide with the
historical TD3+BC results.

This protocol uses schema `iql_actor_geometry_v3_bottleneck_ddpgbc`;
previous local-KL and expected-MSE Gaussian BC checkpoints/completion markers
are not reused. The deterministic base normalization also changed in v3.

- `config.json`, `PROVENANCE.json`: precise configuration, source hashes,
  package versions, dataset digest, state-normalization statistics and action
  transform. New runs use the current installed source.
- `eval.csv`: long-form `step,variant,hop,eval_mode,return,d4rl_score`.
  `hop=0` is the base; `hop=K` is the refined policy. Do not pass this table
  to the legacy TD3+BC-only result parser.
- `metrics.jsonl`: last-minibatch losses at each dispatch; per-hop expected Q,
  sampled Q gain, mean/std displacement, std magnitude, Q scale, and the
  fraction of raw sample coordinates outside action bounds. Q gains are
  estimates from the frozen critic, not measured return improvements.
- `params_<step>.pkl`: Q/V, target Q, all base/refinement actors, every
  optimizer state, training RNG, normalization and configuration. Automatic
  resume rejects changed source, dataset, or mathematical settings. These
  are trusted local pickle checkpoints.
- `COMPLETE.json`: written only after the final checkpoint and all requested
  final actor/evaluation rows exist. A final checkpoint without completed
  evaluation is resumed through evaluation. SIGINT/SIGTERM saves after the
  current dispatch and leaves the run incomplete.

Use a new `--save-dir` after changing source. Increase `--max-timesteps` to
extend an existing constant-LR run without changing its identity. Checkpoint
and evaluation times may change without altering the training RNG stream.

Run targeted validation:

```bash
pytest tests/test_iql_mpi.py tests/test_iql_launcher.py tests/test_core.py tests/test_launcher.py
```

The tests check the paper's DDPG+BC loss and an analytic SGD step (including
Q-only clipping, NLL dimension sum, fixed std, and independence from MC noise),
the deterministic TD3+BC detached scale and MSE convention,
W2 and FR against independent Gaussian overlap integration,
finite gradients/Hessians at identical policies, the local Fisher metric,
nonzero variance gradients,
variance updates on a concave quadratic critic, stopped/recentered references,
IQL target semantics, critic independence from actor choices, JIT dispatch
invariance, checkpoint continuation, and launcher/evaluation recovery.

V3 validation: 48 tests passed on CPU (Python 3.12, JAX 0.11.2, Flax 0.12.9,
Optax 0.2.8), including two actual training updates, all 10 base/final evaluation
paths with a stubbed simulator, checkpoint continuation, and final-eval recovery.
The earlier v2 simulator smoke run does not validate this changed base loss.
No D4RL training-performance claim follows from the unit/integration tests.
