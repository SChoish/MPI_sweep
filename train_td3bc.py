"""Multi-step policy improvement (MPI) on TD3+BC in Flax/JAX.

The actor objective is canonical scale-normalized TD3+BC, parameterized by
tau = alpha / 2:

  lambda = 2 * tau / mean(abs(Q1))
  L_pi = -lambda * mean(Q1) + MSE(π, a)

``--tau`` is alpha / 2, not the Polyak coefficient. With ``--mpi-steps K``,
the actor performs K implicit JKO hops of size tau/K. The final actor is
evaluated at total time tau.

Other defaults: lr=3e-4, policy_noise=0.2*max_a, noise_clip=0.5*max_a,
policy_freq=2, batch=256, 1M steps, state norm eps=1e-3, dataset terminals,
no critic LayerNorm.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import signal
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from d4rl_data import DATASET_FILES, download_dataset

EVAL_ENV = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}
# Official D4RL locomotion reference returns (same for v0/v2).
REF_MIN = {
    "halfcheetah": -280.178953,
    "hopper": -20.272305,
    "walker2d": 1.629008,
}
REF_MAX = {
    "halfcheetah": 12135.0,
    "hopper": 3234.3,
    "walker2d": 4592.3,
}


class Transition(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    not_dones: jax.Array


class TD3BCTrainState(NamedTuple):
    actors: tuple[TrainState, ...]
    critic: TrainState
    target_actor: TrainState
    target_critic: TrainState
    max_action: float
    policy_noise: float
    noise_clip: float


class Actor(nn.Module):
    action_dim: int
    max_action: float
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, state):
        x = state
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        x = nn.Dense(self.action_dim)(x)
        return self.max_action * jnp.tanh(x)


class TwinCritic(nn.Module):
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        def q_net(x, name: str):
            for i, width in enumerate(self.hidden_dims):
                x = nn.relu(nn.Dense(width, name=f"{name}_l{i}")(x))
            return nn.Dense(1, name=f"{name}_out")(x)

        return q_net(sa, "q1"), q_net(sa, "q2")


def _domain(env_name: str) -> str:
    return env_name.split("-", 1)[0]


def normalized_score(env_name: str, return_: float) -> float:
    domain = _domain(env_name)
    return (return_ - REF_MIN[domain]) / (REF_MAX[domain] - REF_MIN[domain]) * 100.0


def qlearning_from_hdf5(path: Path, max_episode_steps: int = 1000) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        observations = np.array(handle["observations"], dtype=np.float32)
        actions = np.array(handle["actions"], dtype=np.float32)
        rewards = np.array(handle["rewards"], dtype=np.float32)
        terminals = np.array(handle["terminals"], dtype=np.float32)
        timeouts = (
            np.array(handle["timeouts"], dtype=np.float32)
            if "timeouts" in handle
            else None
        )

    n = rewards.shape[0]
    obs_list, next_list, act_list, rew_list, done_list = [], [], [], [], []
    episode_step = 0
    for i in range(n - 1):
        done_bool = bool(terminals[i])
        final_timestep = bool(timeouts[i]) if timeouts is not None else episode_step == (
            max_episode_steps - 1
        )
        if final_timestep:
            episode_step = 0
            continue
        if done_bool:
            episode_step = 0
        else:
            episode_step += 1
        obs_list.append(observations[i])
        next_list.append(observations[i + 1])
        act_list.append(actions[i])
        rew_list.append(rewards[i])
        done_list.append(float(done_bool))

    return {
        "observations": np.stack(obs_list),
        "actions": np.stack(act_list),
        "next_observations": np.stack(next_list),
        "rewards": np.asarray(rew_list, dtype=np.float32)[:, None],
        "not_dones": 1.0 - np.asarray(done_list, dtype=np.float32)[:, None],
    }


def load_transition(env_name: str, data_dir: Path, normalize: bool, eps: float = 1e-3):
    raw = qlearning_from_hdf5(download_dataset(env_name, data_dir))
    mean = np.zeros(raw["observations"].shape[-1], dtype=np.float32)
    std = np.ones_like(mean)
    if normalize:
        mean = raw["observations"].mean(axis=0)
        std = raw["observations"].std(axis=0) + eps
        raw["observations"] = (raw["observations"] - mean) / std
        raw["next_observations"] = (raw["next_observations"] - mean) / std
    data = Transition(
        observations=jnp.asarray(raw["observations"]),
        actions=jnp.asarray(raw["actions"]),
        rewards=jnp.asarray(raw["rewards"]),
        next_observations=jnp.asarray(raw["next_observations"]),
        not_dones=jnp.asarray(raw["not_dones"]),
    )
    return data, mean, std


def target_update(model: TrainState, target: TrainState, tau: float) -> TrainState:
    new_params = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1.0 - tau) * tp, model.params, target.params
    )
    return target.replace(params=new_params)


def sample_batch(data: Transition, rng: jax.Array, batch_size: int) -> Transition:
    idx = jax.random.randint(rng, (batch_size,), 0, data.observations.shape[0])
    return jax.tree_util.tree_map(lambda x: x[idx], data)


def update_critic(
    ts: TD3BCTrainState, batch: Transition, rng: jax.Array, discount: float
) -> tuple[TD3BCTrainState, jax.Array]:
    noise = jax.random.normal(rng, batch.actions.shape) * ts.policy_noise
    noise = jnp.clip(noise, -ts.noise_clip, ts.noise_clip)
    next_action = jnp.clip(
        ts.target_actor.apply_fn(ts.target_actor.params, batch.next_observations) + noise,
        -ts.max_action,
        ts.max_action,
    )
    target_q1, target_q2 = ts.target_critic.apply_fn(
        ts.target_critic.params, batch.next_observations, next_action
    )
    target_q = batch.rewards + batch.not_dones * discount * jnp.minimum(target_q1, target_q2)
    target_q = jax.lax.stop_gradient(target_q)

    def loss_fn(params):
        q1, q2 = ts.critic.apply_fn(params, batch.observations, batch.actions)
        return jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))

    loss, grads = jax.value_and_grad(loss_fn)(ts.critic.params)
    critic = ts.critic.apply_gradients(grads=grads)
    return ts._replace(critic=critic), loss


def normalized_q_weight(
    q: jax.Array, tau: float, eps: float = 1e-6, scale_norm: bool = True
) -> jax.Array:
    """Canonical TD3+BC lambda. scale_norm=True → α / mean(|Q|); False → α=2τ."""
    alpha = 2.0 * tau
    if not scale_norm:
        return jax.lax.stop_gradient(jnp.asarray(alpha, dtype=q.dtype))
    q_scale = jnp.mean(jnp.abs(q))
    return jax.lax.stop_gradient(alpha / (q_scale + eps))


def update_first_actor(
    ts: TD3BCTrainState, batch: Transition, tau: float, polyak: float, scale_norm: bool
) -> tuple[TD3BCTrainState, jax.Array]:
    actor = ts.actors[0]

    def loss_fn(params):
        pi = actor.apply_fn(params, batch.observations)
        q1, _ = ts.critic.apply_fn(ts.critic.params, batch.observations, pi)
        bc = jnp.mean(jnp.square(pi - batch.actions))
        q_weight = normalized_q_weight(q1, tau, scale_norm=scale_norm)
        return -q_weight * jnp.mean(q1) + bc

    loss, grads = jax.value_and_grad(loss_fn)(actor.params)
    actor = actor.apply_gradients(grads=grads)
    ts = ts._replace(actors=(actor, *ts.actors[1:]))
    ts = ts._replace(
        target_actor=target_update(actor, ts.target_actor, polyak),
        target_critic=target_update(ts.critic, ts.target_critic, polyak),
    )
    return ts, loss


def update_jko(
    actor: TrainState,
    apply_fn,
    critic: TrainState,
    batch: Transition,
    ref_actions: jax.Array,
    tau_step: float,
    scale_norm: bool,
) -> tuple[TrainState, jax.Array]:
    """One W2 JKO hop: prox to stop-grad ref, λ from |Q(s, ref)| unless unnorm."""
    ref = jax.lax.stop_gradient(ref_actions)
    q_ref, _ = critic.apply_fn(critic.params, batch.observations, ref)
    q_weight = normalized_q_weight(q_ref, tau_step, scale_norm=scale_norm)

    def loss_fn(params):
        pi = apply_fn(params, batch.observations)
        q1, _ = critic.apply_fn(critic.params, batch.observations, pi)
        w2 = jnp.mean(jnp.square(pi - ref))
        return -q_weight * jnp.mean(q1) + w2

    loss, grads = jax.value_and_grad(loss_fn)(actor.params)
    return actor.apply_gradients(grads=grads), loss


def update_actor_hop(
    ts: TD3BCTrainState,
    batch: Transition,
    actor_index: int,
    tau_step: float,
    scale_norm: bool,
) -> tuple[TD3BCTrainState, jax.Array]:
    """Update one JKO hop toward the preceding actor."""
    previous_actor = ts.actors[actor_index - 1]
    actor = ts.actors[actor_index]
    ref = previous_actor.apply_fn(previous_actor.params, batch.observations)
    actor, loss = update_jko(
        actor, actor.apply_fn, ts.critic, batch, ref, tau_step, scale_norm
    )
    actors = (*ts.actors[:actor_index], actor, *ts.actors[actor_index + 1 :])
    return ts._replace(actors=actors), loss


def update_n_times(
    ts: TD3BCTrainState,
    data: Transition,
    rng: jax.Array,
    start_it: jax.Array,
    n_updates: int,
    batch_size: int,
    discount: float,
    tau: float,
    polyak: float,
    policy_freq: int,
    scale_norm: bool,
) -> tuple[TD3BCTrainState, dict]:
    tau_step = tau / float(len(ts.actors))
    initial_actor_losses = tuple(jnp.array(0.0) for _ in ts.actors)

    def body(i, carry):
        ts, rng, critic_loss, actor_losses = carry
        rng, b_rng, c_rng = jax.random.split(rng, 3)
        batch = sample_batch(data, b_rng, batch_size)
        ts, critic_loss = update_critic(ts, batch, c_rng, discount)
        total_it = start_it + i + 1

        def do_actor(operands):
            ts, batch, _actor_losses = operands
            ts, first_loss = update_first_actor(
                ts, batch, tau_step, polyak, scale_norm
            )
            losses = [first_loss]
            for actor_index in range(1, len(ts.actors)):
                ts, loss = update_actor_hop(
                    ts, batch, actor_index, tau_step, scale_norm
                )
                losses.append(loss)
            return ts, tuple(losses)

        def skip_actor(operands):
            ts, _batch, actor_losses = operands
            return ts, actor_losses

        ts, actor_losses = jax.lax.cond(
            (total_it % policy_freq) == 0,
            do_actor,
            skip_actor,
            (ts, batch, actor_losses),
        )
        return ts, rng, critic_loss, actor_losses

    ts, rng, critic_loss, actor_losses = jax.lax.fori_loop(
        0,
        n_updates,
        body,
        (ts, rng, jnp.array(0.0), initial_actor_losses),
    )
    return ts, {
        "critic_loss": critic_loss,
        "actor_loss": actor_losses[0],
        "final_actor_loss": actor_losses[-1],
    }


def create_train_state(
    rng: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    max_action: float,
    lr: float,
    policy_noise: float,
    noise_clip: float,
    mpi_steps: int,
) -> TD3BCTrainState:
    actor_model = Actor(action_dim=actions.shape[-1], max_action=max_action)
    critic_model = TwinCritic()
    keys = jax.random.split(rng, mpi_steps + 1)
    actors = tuple(
        TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_rng, observations),
            tx=optax.adam(lr),
        )
        for actor_rng in keys[:-1]
    )
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(keys[-1], observations, actions),
        tx=optax.adam(lr),
    )
    target_actor = actors[0].replace(
        params=jax.tree_util.tree_map(jnp.copy, actors[0].params)
    )
    target_critic = critic.replace(params=jax.tree_util.tree_map(jnp.copy, critic.params))
    return TD3BCTrainState(
        actors=actors,
        critic=critic,
        target_actor=target_actor,
        target_critic=target_critic,
        max_action=max_action,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
    )


def d4rl_normalized_score(env_name: str, return_: float) -> float:
    try:
        from d4rl.infos import REF_MAX_SCORE, REF_MIN_SCORE

        lo = float(REF_MIN_SCORE[env_name])
        hi = float(REF_MAX_SCORE[env_name])
        return (return_ - lo) / (hi - lo) * 100.0
    except Exception:
        return normalized_score(env_name, return_)


def evaluate(policy_fn, env_name: str, seed: int, mean, std, episodes: int) -> tuple[float, float]:
    env = gym.make(EVAL_ENV[_domain(env_name)])
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 100 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            state = (np.asarray(obs, dtype=np.float32) - mean) / std
            action = np.asarray(policy_fn(state), dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
        returns.append(ep_ret)
    env.close()
    avg = float(np.mean(returns))
    return avg, float(d4rl_normalized_score(env_name, avg))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-timesteps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Half of canonical TD3+BC alpha. With scale normalization, λ=2τ/mean(|Q|).",
    )
    parser.add_argument(
        "--polyak",
        type=float,
        default=0.005,
        help="Target-network EMA / Polyak mix rate",
    )
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--q-scale-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide λ by mean(|Q|)",
    )
    parser.add_argument("--n-jitted-updates", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--save-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Periodic params_{step}.pkl interval; 0 = only 1M + emergency save",
    )
    parser.add_argument(
        "--restore-path",
        type=Path,
        help="Optional trusted checkpoint pickle to resume",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not auto-load the latest params_*.pkl in the run directory",
    )
    parser.add_argument(
        "--mpi-steps",
        type=int,
        default=2,
        help="Positive number K of implicit JKO hops; each uses tau / K",
    )
    return parser.parse_args(argv)


def _ckpt_step(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def latest_checkpoint(out_dir: Path) -> Path | None:
    files = [path for path in out_dir.glob("params_*.pkl") if path.stem[7:].isdigit()]
    if not files:
        return None
    return max(files, key=_ckpt_step)


def save_checkpoint(path: Path, ts: TD3BCTrainState, step: int, rng, mean, std, args) -> None:
    payload = {
        "step": int(step),
        "rng": jax.device_get(rng),
        "mean": np.asarray(mean),
        "std": np.asarray(std),
        "config": vars(args),
        "max_action": float(ts.max_action),
        "policy_noise": float(ts.policy_noise),
        "noise_clip": float(ts.noise_clip),
        "actors_params": jax.device_get(tuple(actor.params for actor in ts.actors)),
        "actors_steps": jax.device_get(tuple(actor.step for actor in ts.actors)),
        "critic_params": jax.device_get(ts.critic.params),
        "critic_step": jax.device_get(ts.critic.step),
        "target_actor_params": jax.device_get(ts.target_actor.params),
        "target_critic_params": jax.device_get(ts.target_critic.params),
        "actors_opt_states": jax.device_get(
            tuple(actor.opt_state for actor in ts.actors)
        ),
        "critic_opt_state": jax.device_get(ts.critic.opt_state),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_checkpoint(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def restore_train_state(ts: TD3BCTrainState, payload: dict) -> TD3BCTrainState:
    actor_params = payload["actors_params"]
    actor_opt_states = payload["actors_opt_states"]
    actor_steps = payload["actors_steps"]
    if len(actor_params) != len(ts.actors):
        raise ValueError(
            "checkpoint hop count does not match --mpi-steps: "
            f"{len(actor_params)} != {len(ts.actors)}"
        )
    actors = tuple(
        actor.replace(params=params, opt_state=opt_state, step=step)
        for actor, params, opt_state, step in zip(
            ts.actors, actor_params, actor_opt_states, actor_steps, strict=True
        )
    )
    critic = ts.critic.replace(
        params=payload["critic_params"],
        opt_state=payload["critic_opt_state"],
        step=payload["critic_step"],
    )
    target_actor = ts.target_actor.replace(params=payload["target_actor_params"])
    target_critic = ts.target_critic.replace(params=payload["target_critic_params"])
    return ts._replace(
        actors=actors,
        critic=critic,
        target_actor=target_actor,
        target_critic=target_critic,
        max_action=float(payload.get("max_action", ts.max_action)),
        policy_noise=float(payload.get("policy_noise", ts.policy_noise)),
        noise_clip=float(payload.get("noise_clip", ts.noise_clip)),
    )


def install_stop_handler(save_dir: Path) -> dict:
    stop_requested = {"flag": False, "signum": None}

    def _request_stop(signum, _frame):
        if stop_requested["flag"]:
            return
        stop_requested["flag"] = True
        stop_requested["signum"] = int(signum)
        print(
            f"[signal] signum={signum} — will emergency-save after current step "
            f"(save_dir={save_dir})",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        signal.signal(signal.SIGHUP, _request_stop)
    except (ValueError, OSError):
        pass
    return stop_requested


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env not in DATASET_FILES:
        raise ValueError(f"Unsupported env {args.env}. Expected one of {sorted(DATASET_FILES)}")
    if args.tau <= 0.0:
        raise ValueError("tau must be positive (tau = alpha / 2)")
    if args.mpi_steps < 1:
        raise ValueError("mpi_steps must be positive")
    if not 0.0 < args.polyak <= 1.0:
        raise ValueError("polyak must lie in (0, 1]")
    if args.eval_freq < 1 or args.n_jitted_updates < 1:
        raise ValueError("eval_freq and n_jitted_updates must be positive")
    if args.eval_freq % args.n_jitted_updates != 0:
        raise ValueError("eval_freq must be divisible by n_jitted_updates")
    if args.max_timesteps % args.n_jitted_updates != 0:
        raise ValueError("max_timesteps must be divisible by n_jitted_updates")

    if args.save_interval and args.save_interval % args.n_jitted_updates != 0:
        raise ValueError("save_interval must be divisible by n_jitted_updates")

    mpi_steps = args.mpi_steps
    run_name = f"{args.env}_tau{args.tau:g}_mpi{mpi_steps}_seed{args.seed}"
    out_dir = args.save_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    stop_requested = install_stop_handler(out_dir)

    data, mean, std = load_transition(args.env, args.data_dir, args.normalize)
    example = jax.tree_util.tree_map(lambda x: x[:1], data)
    max_action = float(np.max(np.abs(np.asarray(data.actions))))
    max_action = 1.0 if max_action <= 1.0 + 1e-5 else max_action

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    ts = create_train_state(
        init_rng,
        example.observations,
        example.actions,
        max_action=max_action,
        lr=args.lr,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        mpi_steps=mpi_steps,
    )

    start_step = 0
    restore_path = args.restore_path
    if restore_path is None and not args.no_resume:
        restore_path = latest_checkpoint(out_dir)
    if restore_path is not None:
        payload = load_checkpoint(restore_path)
        ts = restore_train_state(ts, payload)
        rng = payload["rng"]
        mean = payload["mean"]
        std = payload["std"]
        start_step = int(payload["step"])
        print(f"[resume] loaded {restore_path} step={start_step}", flush=True)

    update_fn = jax.jit(
        partial(
            update_n_times,
            n_updates=args.n_jitted_updates,
            batch_size=args.batch_size,
            discount=args.discount,
            tau=args.tau,
            polyak=args.polyak,
            policy_freq=args.policy_freq,
            scale_norm=bool(args.q_scale_norm),
        )
    )

    def _act(params, obs):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
            squeeze = True
        else:
            squeeze = False
        action = ts.actors[0].apply_fn(params, obs)
        return action[0] if squeeze else action

    act_fn = jax.jit(_act)

    eval_path = out_dir / "eval.csv"
    write_header = (
        start_step == 0
        or not eval_path.is_file()
        or eval_path.stat().st_size == 0
    )
    eval_mode = "w" if start_step == 0 else "a"
    with eval_path.open(eval_mode, newline="", encoding="utf-8") as file:
        fieldnames = ["step", "return", "d4rl_score", "critic_loss", "actor_loss"]
        if mpi_steps > 1:
            fieldnames.extend(
                [f"return_pi{mpi_steps}", f"d4rl_pi{mpi_steps}", "final_actor_loss"]
            )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        n_outer = args.max_timesteps // args.n_jitted_updates
        start_outer = start_step // args.n_jitted_updates
        eval_every = args.eval_freq // args.n_jitted_updates
        save_every = (
            args.save_interval // args.n_jitted_updates if args.save_interval else 0
        )
        start = time.time()
        metrics = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "final_actor_loss": 0.0,
        }
        for i in range(start_outer + 1, n_outer + 1):
            rng, update_rng = jax.random.split(rng)
            start_it = jnp.asarray((i - 1) * args.n_jitted_updates)
            ts, metrics = update_fn(ts, data, update_rng, start_it)
            step = i * args.n_jitted_updates
            emergency_stop = bool(stop_requested["flag"])
            if i % eval_every == 0 or i == n_outer:
                policy = partial(act_fn, ts.actors[0].params)
                avg_ret, score = evaluate(
                    policy, args.env, args.seed, mean, std, args.eval_episodes
                )
                row = {
                    "step": step,
                    "return": avg_ret,
                    "d4rl_score": score,
                    "critic_loss": float(metrics["critic_loss"]),
                    "actor_loss": float(metrics["actor_loss"]),
                }
                extra = ""
                if mpi_steps > 1:
                    final_policy = partial(act_fn, ts.actors[-1].params)
                    final_return, final_score = evaluate(
                        final_policy,
                        args.env,
                        args.seed,
                        mean,
                        std,
                        args.eval_episodes,
                    )
                    row[f"return_pi{mpi_steps}"] = final_return
                    row[f"d4rl_pi{mpi_steps}"] = final_score
                    row["final_actor_loss"] = float(metrics["final_actor_loss"])
                    extra = f" d4rl_pi{mpi_steps}={final_score:.1f}"
                writer.writerow(row)
                file.flush()
                elapsed = time.time() - start
                print(
                    f"[{run_name}] step={step} return={avg_ret:.1f} "
                    f"d4rl={score:.1f}{extra} elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
            if (save_every and i % save_every == 0) or i == n_outer or emergency_stop:
                ckpt_path = out_dir / f"params_{step}.pkl"
                save_checkpoint(ckpt_path, ts, step, rng, mean, std, args)
                print(f"[ckpt] Saved to {ckpt_path}", flush=True)
            if emergency_stop:
                marker = out_dir / f"EMERGENCY_SAVE_step{step}"
                marker.write_text(
                    f"signal={stop_requested['signum']} step={step}\n",
                    encoding="utf-8",
                )
                print(
                    f"[signal] emergency_save_done step={step} "
                    f"signum={stop_requested['signum']} save_dir={out_dir}",
                    flush=True,
                )
                break

    print(f"Wrote {eval_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
