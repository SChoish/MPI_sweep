# Budgeted Actor Refinement Sweep

Reproducible Budgeted Actor Refinement (BAR) sweeps for offline
TD3+BC, with proximal-loss and action-metric-matched projected-linearized realizations
implemented in JAX/Flax. The release supports the
nine D4RL MuJoCo locomotion datasets built from Hopper, HalfCheetah, and
Walker2d with the `medium`, `medium-replay`, and `expert` splits.

## Method

For a total step size `tau` and `K` policy-improvement hops, each hop uses
`h = tau / K`. Let `d` be the action dimension. The implementation averages
the squared displacement over action coordinates, so its ground cost is

```text
c_d(a, b) = ||a - b||_2^2 / d.
```

The default implicit integrator regularizes the first actor toward the dataset
action and later actors toward the previous actor with a stopped gradient:

```text
reference_1(s) = dataset action
reference_k(s) = stop_gradient(actor_(k-1)(s))

q_scale_1 = mean(abs(Q(s, actor_1_pre(s))))
q_scale_k = mean(abs(Q(s, reference_k(s))))  for k > 1
lambda_k  = 2 (tau / K) / (q_scale_k + epsilon)
loss_k   = -lambda_k mean(Q(s, actor_k(s)))
           + mean(square(actor_k(s) - reference_k(s)))
```

Here `actor_1_pre` is the first actor before its current optimizer update.
`--no-q-scale-norm` replaces `lambda_k` with `2 * tau / K`.

The matched projected explicit integrator uses the same metric convention:

```text
q_scale_exp,k   = mean(abs(Q(s, reference_k(s))))
g_k             = grad_a Q(s, reference_k(s)) / q_scale_exp,k
target_k        = clip(reference_k(s) + d h g_k, -max_action, max_action)
loss_explicit_k = mean(square(actor_k(s) - stop_gradient(target_k)))
```

Select it with `--integrator explicit`. The factor `d` is required because the
implicit transport cost is a mean over action coordinates. Omitting `d` gives
an explicit target that is `d` times smaller and does not represent the same
nominal flow time. Because the target is clipped and fitted by a neural actor,
this BAR-Lin variant is a projected-and-regressed Euler approximation rather
than an exact unconstrained Euler trajectory.

### Wasserstein-2 gradient-flow interpretation

The conceptual starting point is the Jordan--Kinderlehrer--Otto (JKO)
*minimizing-movement* discretization of a gradient flow in the space
`P_2(A)` of action distributions with finite second moment. Hold the offline
state marginal `d_D(s)` fixed and represent a stochastic policy by the kernel
`pi(da | s)`. A convenient state-conditioned transport metric uses `W_{2,d}`,
whose ground cost is `c_d(a,b) = ||a-b||_2^2 / d`:

```text
W_D^2(pi, nu) = E_{s ~ d_D}[W_{2,d}^2(pi(. | s), nu(. | s))].
```

For a frozen critic, define the energy

```text
E_Q(pi) = -E_{s ~ d_D, a ~ pi(. | s)}[Q(s, a)].
```

One implicit time step of length `h` is then formally

```text
pi_next in argmin_pi E_Q(pi) + W_D^2(pi, pi_ref) / (2 h).
```

This is an implicit-Euler step in `W_2` space: the energy term moves policy
mass toward actions with larger `Q`, while the transport term penalizes moving
too far in one step. In the ideal frozen-critic construction, `K` accurately
solved steps with `h = tau / K` approximate total flow time `tau` in the
small-step limit. The live implementation instead keeps independently
initialized persistent actors and applies one Adam update to each; the ideal
flow motivates its local geometry but does not certify the learned finite-step
chain.

The deterministic-policy limit makes the transport term especially simple.
If the conditional policies collapse to Dirac measures,

```text
pi(. | s)     = delta_{u(s)},
pi_ref(. | s) = delta_{u_ref(s)},
```

then

```text
W_{2,d}^2(delta_{u(s)}, delta_{u_ref(s)})
    = ||u(s) - u_ref(s)||_2^2 / d.
```

Consequently, after multiplying the JKO objective by `2 h`, its deterministic
form is

```text
u_next in argmin_u
    -2 h E_s[Q(s, u(s))]
    + E_s[||u(s) - u_ref(s)||_2^2 / d].
```

This is the `-lambda * Q + MSE` actor loss used here. Q-scale normalization
replaces `h` by the locally rescaled time step `h / (mean(abs(Q)) + epsilon)`,
making the nominal flow time less sensitive to the critic's numerical scale.

Formally, as `h -> 0`, a policy density `rho_t(a | s)` follows the continuity
equation

```text
partial_t rho_t + div_a(rho_t v_t) = 0,
v_t(a, s) = d grad_a Q(s, a) / q_scale.
```

In the deterministic limit, the corresponding characteristics satisfy

```text
d u_t(s) / dt = d grad_a Q(s, u_t(s)) / q_scale,
```

so deterministic actions behave like particles transported along the critic's
action gradient in the normalized action metric. The factor `d` disappears if
the unnormalized Euclidean ground cost is used together with an action-summed,
rather than action-averaged, proximal loss. There is no diffusion term because
this implementation has no policy entropy or stochastic temperature;
“deterministic limit” describes the Dirac/particle interpretation of the
Wasserstein flow.

This correspondence should be read as a modeling interpretation rather than
an exact JKO solver:

- The state marginal is fixed, and transport occurs only within each action
  fiber. The MSE is exactly the conditional `W_{2,d}^2` cost for Dirac
  policies, not unrestricted optimal transport over joint state-action
  measures.
- Neural actors restrict the admissible maps `u(s)`, and minibatch Adam updates
  only approximate each proximal minimization.
- The critic is learned, non-convex, and changes during training. Classical JKO
  convergence or uniqueness results for a fixed geodesically convex energy do
  not automatically apply.
- Action clipping and finite step sizes further modify the formal continuous
  gradient flow.

The variational construction originates with
[Jordan, Kinderlehrer, and Otto (1998)](https://doi.org/10.1137/S0036141096303359).
For the general metric-space theory, see
[Ambrosio, Gigli, and Savare (2005)](https://doi.org/10.1007/b137080), and for
an accessible overview see
[Santambrogio (2017)](https://arxiv.org/abs/1609.03890).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/SChoish/MPI_sweep.git
cd MPI_sweep
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The command above installs CPU JAX. For an NVIDIA GPU, install the JAX wheel
that matches the driver first (CUDA 13 is the current default; CUDA 12 wheels
are also available), then run `pip install -e .`:

```bash
pip install --upgrade "jax[cuda13]"
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for CUDA and platform compatibility. Gymnasium's MuJoCo dependency is included
by this package.

## Quick start

The installed command and flag names retain their original identifiers for
backward compatibility; they are interfaces, not the paper's algorithm name.

Inspect a small job matrix without downloading data or starting workers:

```bash
mpi-sweep --hops 4 --n-tau 2 --seeds 0 --gpus 0 --dry-run
```

Run the complete default grid with `K=4` on two GPUs:

```bash
mpi-sweep \
  --hops 4 \
  --seeds "0 1" \
  --gpus "0 1" \
  --slots-per-gpu 1 \
  --data-dir ./data \
  --save-dir ./results/mpi4 \
  --log-dir ./logs/mpi4
```

Run the matched projected explicit variant on the same grid by changing only
the integrator and output directories:

```bash
mpi-sweep \
  --integrator explicit \
  --hops 4 \
  --seeds "0 1" \
  --gpus "0 1" \
  --slots-per-gpu 1 \
  --data-dir ./data \
  --save-dir ./results/exp4 \
  --log-dir ./logs/exp4
```

`--hops K` accepts any positive integer; BAR creates exactly `K` actors and
uses `tau / K` at each hop. A custom grid can be passed with
`--taus "0.1 0.4 1.5"`. Domains, dataset splits, and seeds accept either spaces
or commas.

To run one configuration without the sweep scheduler:

```bash
mpi-train \
  --env hopper-medium-v2 \
  --tau 1.5 \
  --mpi-steps 4 \
  --integrator implicit \
  --seed 0 \
  --data-dir ./data \
  --save-dir ./results
```

## Resume and outputs

Each run writes to:

```text
<save-dir>/<env>_tau<tau>_<mpi|exp><hops>_seed<seed>/
├── config.json
├── eval.csv
└── params_<step>.pkl
```

- Completed cells are skipped when `eval.csv` contains `max_timesteps`.
- Incomplete cells automatically resume from the newest `params_<step>.pkl`.
- `SIGINT`, `SIGTERM`, and `SIGHUP` request a checkpoint after the current
  jitted update block. Checkpoints are Python pickle files; load only trusted
  files.
- Dataset downloads use a temporary file and an atomic rename.
- Add `--cpu-affinity --cpus-per-job N` to pin workers with Linux `taskset`.

Use `mpi-sweep --help` and `mpi-train --help` for all options.

## Released results

The compact final-score tables are under [`sweep_results/`](sweep_results/).
Directories `K=1` through `K=4` are organized by hop count and integrator.
The proximal `K={1,2,3,4}` and projected-linearized `K={2,3}` CSVs for
seeds 0--3 each complete the same nine-task, fourteen-budget grid. In the
high-budget region, the four-seed projected-linearized mean rises from 38.43
at `K=2` to 46.44 at `K=3`, with score-below-20 cells falling from 25/63
to 21/63. Large training checkpoints are not committed. The host-specific
diagnostic bundle includes compact outputs and post-hoc manifests derived from
all 504 local
K4 checkpoints, including a 180-cell four-seed target-exposure audit. Those
audit manifests do not reconstruct the missing historical training
configurations for proximal seeds 2--3.

Audited movement, critic failure, target exposure, simulator calibration, and
targeted controls are documented in
[`sweep_results/diagnostics/README.md`](sweep_results/diagnostics/README.md).
Per-state simulator rollouts and other large intermediates are excluded.

### MCEP-inspired policy-separation control

A completed two-seed control tests whether separate conservative target and
full-budget deployment actors reproduce key BAR-P3 aggregate outcomes without
sequential re-centering. It is inspired by
[MCEP](https://openreview.net/forum?id=imAROs79Pb), but uses BAR's normalization
and fixed tau/3 versus tau mapping; it is neither a reproduction of the
published method nor a compute-matched re-centering ablation.

Across 90 matched task--budget--seed cells, the control scores 59.22 versus
57.67 for a contemporaneous BAR-P3 rerun. Control minus BAR is +1.56 with a
95% task-resampling interval of [-1.52, 5.44], while the paired median is
-0.42 and BAR wins 52/90 cells; after averaging the two seeds, both procedures
collapse in 9/45 task--budget cells. Sequential re-centering is therefore not
a prerequisite for the observed mean and seed-mean collapse outcomes on this
grid. Because the control also changes actor work, anchoring, and the realized
critic trajectory, it establishes neither superiority nor equivalence and does
not isolate which element of the simpler procedure is responsible.

The compact [paired archive](sweep_results/diagnostics/bar_mcep_p3_paired/)
contains all final scores, independently checked aggregates, recovery
classification, and a 180-row source manifest. Every row hashes its config,
evaluation, and final checkpoint, plus available logs. Large raw artifacts are
excluded. Exact local training-file digests are recorded because the run
configs did not store a Git revision; releasing a clean runnable snapshot remains tracked in
[TODO.md](TODO.md).

## Paper

The anonymous AISTATS 2026 sources, official style files, build instructions,
and generated PDFs are in [aistats26_manuscript/](aistats26_manuscript/).
Submission-facing experiment decisions, including the redesigned fixed-operator
error audit, are tracked in [TODO.md](TODO.md).

## Development

```bash
pip install -e ".[test]"
pytest
```

## Acknowledgements

The trainer is based on TD3+BC by Fujimoto and Gu, *A Minimalist Approach to
Offline Reinforcement Learning* (NeurIPS 2021), and was developed from the JAX
port at [ethanluoyc/td3_bc_jax](https://github.com/ethanluoyc/td3_bc_jax).

## License

MIT. See [LICENSE](LICENSE).
