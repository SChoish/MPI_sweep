# IQL actor geometry: fixed total time, K total steps

All selected paths share one actor-independent IQL Q/V pair. `--tau T` is
the **total** horizon, `--hops K` / `--mpi-steps K` counts **all** steps,
and `h = T/K`. K=1 is the base actor at T. For K>1, the chain contains one
dataset-anchored actor at h followed by K-1 proximal refinement actors.
K=0 is invalid. Independent `--awr-beta`, `--bc-coef`, and `--td3bc-alpha`
flags are removed: their coefficients are derived from T, K, and geometry.

## Base losses and their clocks

Write `Q=min(Q1_target,Q2_target)`, `A=Q-V`, `Qbar=Q/C`, and let `d` be
action dimension. The default `--metric-reduction native` resolves as follows.
Expectations include dataset states; NLL is evaluated at dataset actions.

| Variant | Dataset-anchored loss at h=T/K | Default C | Distance convention |
| --- | --- | --- | --- |
| `awr_gaussian_fr` | `E[min(exp(h*A/C),100) * NLL(a_D;m,sigma)]` | 1 | full-density FR squared |
| `qbc_gaussian_w2` | `-E[Qbar(s,clip(m))] + (1/h)*E[NLL(a_D;m,I)]` | 1 | coordinate-sum W2 squared |
| `qbc_deterministic_w2` | `-2h*E[Qbar(s,m)] + E[sum_j((m_j-a_D,j)^2)/d]` | stopped mean absolute Q | coordinate-mean W2 squared |

Thus native coefficients are `beta=T/K`, `alpha_G=K/T`, and
`alpha_D=2T/K`. At K=1 they are T, 1/T, and 2T, respectively.
The Gaussian NLL uses a dimension **sum** and, at std 1, equals
`sum_j((m_j-a_D,j)^2)/2 + constant`; this accounts for the factor of two
relative to TD3+BC's dimension-mean MSE.

For an explicit sum/mean override, set `M=1` for sum and `M=d` for mean.
The base coefficients become `beta=h*M`, `alpha_G=1/(h*M)`, and
`alpha_D=2*h*M/d`. Refinement uses the same metric convention. For example,
canonical TD3+BC alpha 2.5 corresponds to **T=1.25 with mean W2**, or
**T=1.25*d with sum W2**. Changing a reduction without this conversion
changes the experiment.

AWR's `beta` multiplies advantage; if temperature is instead written in a
**denominator**, that temperature is `1/beta`. The beta/time identification
comes from Fisher-Rao gradient flow (exponential tilting before projection).
Clipped AWR weights and Gaussian projection are not an exact finite FR
proximal solve. The requested AWR base is retained; only its time coefficient
and the subsequent exact closed-form FR penalty are matched.

The Gaussian DDPG+BC base follows Park et al. Eq. 6 / Appendix C.6.1:
unsquashed mean, fixed std 1, clipped-mean Q, raw-Gaussian dataset NLL.
Subsequent MPI actors learn both mean and std through expected Q. This change
of policy family/energy is an explicit MPI extension, not a claim that the
paper's fixed-std mean-Q algorithm is an exact stochastic W2 semigroup.
The AWR family retains its tanh mean and learned state-dependent std;
it is not the fixed-std AWR family in that paper's data-scaling experiment.

## Refinement and Q normalization

For k=2,...,K, with the latest updated predecessor stopped:

```text
L_k = -E[Q(s, m_k + sigma_k*epsilon)]/C
      + E[D_squared(pi_k, stop(pi_(k-1)))]/(2h)
```

For deterministic policies sigma=0. Gaussian expected Q uses reparameterized
antithetic normal samples (default 8), so variance receives a Q gradient.
W2 squared is `sum_j((m-m_ref)^2 + (sigma-sigma_ref)^2)`; the mean convention
divides this by d. FR uses the MPI Appendix A.3 Gaussian overlap closed form:

```text
B = product_j sqrt(2*sigma_j*r_j/(sigma_j^2+r_j^2))
              * exp(-(m_j-m_ref_j)^2/(4*(sigma_j^2+r_j^2)))
D_FR_squared = 4*acos(B)^2
```

This is ambient density-space FR between Gaussian endpoints, not intrinsic
Gaussian-submanifold FR. No KL approximation is used. Evaluation uses a stable
atan2 expression and an analytic derivative at identical endpoints.

Each variant also has an independent full-T base control. Before updating it,
compute `C=stop(mean_batch(abs(Q(s,control_mean))) + 1e-6)` when normalization
is enabled, otherwise C=1. This **one constant is shared by the control,
first chain step, and every refinement**, including all inner updates. The
control's update/initialization does not depend on K, making C identical
across K for identical critic, batch, and seed. K=1 reuses the control as its
only chain actor, without an extra update. K>1 pays for one extra control
update per variant; the control is not a chain predecessor.

At K=1 the deterministic loss is the canonical TD3+BC actor-loss form with
alpha=2T (apart from the numerical epsilon). Using a full-T control scale at
K>1 is our comparison protocol, not a normalization prescribed by the paper.
Optional `--iql-q-scale-norm` / `--no-iql-q-scale-norm` overrides apply to both
base and refinement, including AWR's advantage. The legacy `--q-scale-norm`
flag affects only the separate TD3+BC trainer.

Normalized time is time for Q/C: relative to raw Q, each hop has effective
time h/C. Holding C common prevents hidden changes of this clock with K.
If Q and V are multiplied by c>0 with no Q normalization, preserve the
objective by `T_new=T_old/c`, `beta_new=beta_old/c`, and
`alpha_G,new=c*alpha_G,old`. With a fixed C, normalized and raw conventions
match at `T_normalized=C*T_raw`. Native TD3+BC's Q/C approximately cancels
positive Q rescaling (up to epsilon); this does not imply invariance to
additive Q shifts or finite neural training dynamics. Equal numerical T
across FR/W2 families is not an assertion of equal geometric displacement.

## Source audit and reward scale

Sources checked:

- [Park et al., paper, Eq. 6, Appendix C.6.1, Table 3](https://arxiv.org/html/2406.09329v2).
- [Official NeurIPS supplemental code ZIP](https://proceedings.neurips.cc/paper_files/paper/2024/file/8ffb4e3118280a66b192b6f06e0e2596-Supplemental-Conference.zip).
  `neurips/main.py` defaults `dual_type='none'`; `src/agents/trl.py` uses
  raw `-mean(Q)` plus `alpha*NLL` in that mode. Its optional `dual_type='avg'`
  scales BC by mean absolute Q, but is not the default. The prior claim that
  no dedicated official implementation was available was incomplete.
- [Original IQL actor](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py)
  uses unnormalized advantage with an exponential-weight cap of 100.
- [Official TD3+BC actor](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py)
  uses detached `alpha/mean(abs(Q))` and coordinate-mean MSE;
  [its default alpha is 2.5](https://github.com/sfujim/TD3_BC/blob/main/main.py).

Official-code audit identifiers (SHA256): ZIP
`43de7cc53a67f5ba858572efc68a7c56b4fc127c239da0d0be85f590a19d6ecd`;
`src/agents/trl.py`
`d96699e1518d80a87ca0992cca166f647e05b5c87e6b61942d311907fe3f7f09`;
`src/d4rl_utils.py`
`aadd0b766ea3adea3376fe8524463d14d19683e117a69f4b9a36916eaa284a71`.

**No Q normalization does not mean raw rewards.** The supplement's
`src/d4rl_utils.py:normalize_dataset` and the
[original IQL preprocessing](https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py)
scale locomotion rewards by `1000/(max trajectory return - min trajectory return)`.
This is now the default `--iql-reward-normalization iql`. Returns use
post-qlearning-dataset trajectories, including terminal boundaries, state
discontinuities from skipped timeouts, and the final tail. Bellman masks
still distinguish terminal transitions from timeout boundaries. Normalization
is calculated before state normalization. Dataset actions are clipped at
`1-1e-5`, also following these loaders.

`--iql-reward-normalization none` is an explicit raw-reward ablation;
`--reward-scale` multiplies rewards after this normalization. Both settings
and the actual normalization factor are recorded. State normalization
(`std+1e-3`, on by default) is separate from reward and Q normalization.

The actor loss/time conversions reproduce coefficient conventions, not the
full original agents: Q is the common IQL min-target-Q, not TD3's online Q1
or the supplement's online critic for DDPG. LayerNorm, evaluation protocol,
and the AWR policy family also differ from the bottleneck experiments.
The critic always uses dataset-action expectile V, then `r+gamma*V(s')`
Q targets; actor actions never enter its targets.

## T grids and commands

The default IQL grid is the union of the selected variants' native grids:

| Variant | Grid basis | Total T candidates |
| --- | --- | --- |
| AWR | Existing positive-beta defaults | 1, 3, 10 |
| Gaussian DDPG+BC | MART small-T spacing, range 1/50 to 5 | 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4, 5 |
| Deterministic TD3+BC | Existing canonical alpha=2.5 control | 1.25 |

The DDPG+BC grid is a range sweep, not the inverse of the bottleneck paper's
four alpha candidates. It retains the existing TD3+BC/MART points up to 4
and adds the requested endpoints 0.02 and 5. The
[MART supplement](https://github.com/SChoish/PART/blob/main/supplement.tex)
uses 0.05 through 40; its small-T spacing is the reference here, without
claiming a calibrated Q-scale conversion between the two algorithms.
At K=1 the new range corresponds to alpha_G from 50 down to 0.2.
Each K uses the same total-T grid and still sets alpha_G=K/T in its first step.
AWR beta=0 is a BC-only control outside this positive-time sweep.
With multiple variants, all selected actors run at every T in the union;
select only `qbc_gaussian_w2` for the ten-point 0.02--5 sweep. Selecting all
three also includes their existing T=1, 1.25, 3, and 10 points.
`--n-tau` takes the first N sorted values. Explicit `--taus` overrides it.
Overriding native geometry or Q/reward scale requires explicit T candidates.

For a matched K comparison, change only K, retaining T candidates and seeds:

```bash
for K in 1 2 4; do
  python launch_mpi_sweep.py --algorithm iql --hops "$K" \
    --domains hopper --datasets medium-replay --seeds "0 1" --gpus 0 \
    --variants qbc_gaussian_w2 \
    --save-dir results/iql_time --log-dir logs/iql_time
done
```

Add `--dry-run` to inspect commands without training. To use a family-specific
default grid, pass e.g. `--variants qbc_gaussian_w2` and omit `--taus`.
Direct training: `mpi-iql-train --mpi-steps 1 --tau 1.25`.

## Evaluation, compute, and restart

`eval.csv` columns are `step,variant,policy,hop,K,T,h,time,eval_mode,return,d4rl_score`.
`policy=baseline` has K=1, h=T, time=T. `policy=mpi` has h=T/K and time=hop*h.
Defaults evaluate the full-T baseline and final MPI policy; at K=1 they are
one policy and are evaluated once. `--eval-hops all` adds intermediate MPI
policies. Gaussian mean-action and sampled evaluation are separate; simulator
actions are clipped. `--q-action-transform clip` optionally clips MPI Q inputs,
but geometry is always measured on the pre-clipping distributions.

Actors and optimizer states persist across training iterations. Each chain
uses `1+(K-1)*inner_updates` actor optimizer updates, plus one full-T control
update when K>1. These are amortized neural updates, not exact proximal
minimizers or equal-compute runs. Holding T fixed aligns the declared
objective clock; it does not make the finite-update optimization errors equal.
The learning rates and `inner_updates` are separate from h and are recorded.

Schema `iql_actor_geometry_v4_total_horizon` rejects previous K+1/raw-reward
checkpoints. Run identities include T, K, scale choices and all training/eval
settings. `config.json` / `PROVENANCE.json` record resolved coefficients,
per-hop times, metric conventions, reward factors, compute counts, source
hashes and dataset identity. Metrics include the common Q scale and first-step
coefficients. Checkpoints include Q/V, all controls/chains, optimizer states,
and RNG. Resume verifies the source/dataset/configuration and recovers missing
final evaluations before writing `COMPLETE.json`. Use a new save directory
after source changes. Legacy TD3+BC result parsers do not consume this CSV.

Validation command:

```bash
pytest tests/test_iql_mpi.py tests/test_iql_launcher.py tests/test_core.py tests/test_launcher.py
```

Tests cover analytic losses/gradients, K=1 base identity, shared Q scales and
control/critic independence across K, closed-form FR, variance gradients,
reward trajectory boundaries, coefficient-grid conversion, evaluation clocks,
and checkpoint/resume. Unit tests and simulator-stub integration are not
D4RL training-performance evidence.

V4 validation: all 66 targeted tests passed on CPU with Python 3.12,
JAX 0.11.2, Flax 0.12.9, and Optax 0.2.8. The integration tests execute K=1
and K=3 training, full-time evaluation metadata, and resume/final-evaluation
recovery using synthetic data and a stubbed simulator. The subsequent grid
revision changes only DDPG+BC's default T candidates, not these actor updates.
