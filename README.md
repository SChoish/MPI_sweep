# MPI Sweep

Reproducible multi-step policy improvement (MPI) sweeps for offline TD3+BC,
implemented in JAX/Flax. The release supports the nine D4RL MuJoCo locomotion
datasets built from Hopper, HalfCheetah, and Walker2d with the `medium`,
`medium-replay`, and `expert` splits.

## Method

For a total step size `tau` and `K` policy-improvement hops, each hop uses
`tau / K`. The first actor is regularized toward the dataset action. Later
actors are regularized toward the previous actor with a stopped gradient:

```text
reference_1(s) = dataset action
reference_k(s) = stop_gradient(actor_(k-1)(s))

lambda_k = 2 (tau / K) / (mean(abs(Q(s, reference_k(s)))) + epsilon)
loss_k   = -lambda_k mean(Q(s, actor_k(s)))
           + mean(square(actor_k(s) - reference_k(s)))
```

`--no-q-scale-norm` replaces `lambda_k` with `2 * tau / K`.

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

`--hops K` accepts any positive integer; the implementation creates exactly
`K` actors and uses `tau / K` at each hop. A custom grid can be passed with
`--taus "0.1 0.4 1.5"`. Domains, dataset splits, and seeds accept either spaces
or commas.

To run one configuration without the sweep scheduler:

```bash
mpi-train \
  --env hopper-medium-v2 \
  --tau 1.5 \
  --mpi-steps 4 \
  --seed 0 \
  --data-dir ./data \
  --save-dir ./results
```

## Resume and outputs

Each run writes to:

```text
<save-dir>/<env>_tau<tau>_mpi<hops>_seed<seed>/
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
