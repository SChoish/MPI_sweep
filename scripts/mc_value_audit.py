#!/usr/bin/env python3
"""CPU-only simulator Monte-Carlo audit for MPI checkpoints.

This diagnostic compares a selected driver or route-shadow Q1 with simulator
returns under its route-native continuation and two online policies:

* rho_1 or rho_K: the selected Polyak target actor plus TD3 clipped noise;
* mu_1: the online first actor used by the standard driver route;
* mu_K: the deployed final refinement actor.

The output is a post-hoc simulator reference. It is not used for training,
checkpoint selection, or hyperparameter selection, and it does not by itself
identify bootstrap routing as the cause of critic error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4",
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.mujoco_state import restore_env, snapshot_env  # noqa: E402
from train_td3bc import (  # noqa: E402
    Actor,
    EVAL_ENV,
    TwinCritic,
    _domain,
    load_checkpoint,
)


Array = np.ndarray
PolicyFn = Callable[[Array, np.random.Generator | None], Array]


@dataclass(frozen=True)
class SampledState:
    snapshot: Any
    observation: Array
    episode: int
    time_index: int
    occupancy_weight: float
    rollout_seed: int


class CheckpointEvaluator:
    """Expose checkpoint policies and a selected driver/shadow critic on CPU."""

    CRITIC_SOURCES = ("driver", "shadow_mu1", "shadow_muk")

    def __init__(self, checkpoint: Path, critic_source: str = "driver"):
        self.checkpoint = checkpoint.resolve()
        self.payload: Mapping[str, Any] = load_checkpoint(self.checkpoint)
        self.config = dict(self.payload.get("config", {}))
        self.mean = np.asarray(self.payload["mean"], dtype=np.float32)
        self.std = np.asarray(self.payload["std"], dtype=np.float32)
        self.max_action = float(self.payload.get("max_action", 1.0))
        self.policy_noise = float(self.payload.get("policy_noise", 0.2))
        self.noise_clip = float(self.payload.get("noise_clip", 0.5))
        self.action_dim = self._infer_action_dim()
        self.actor = Actor(action_dim=self.action_dim, max_action=self.max_action)
        self.critic = TwinCritic()
        self.final_actor_key, self.hops = self._infer_final_actor()
        if critic_source not in self.CRITIC_SOURCES:
            raise ValueError(
                f"unknown critic source {critic_source!r}; "
                f"expected one of {self.CRITIC_SOURCES}"
            )
        source_keys = {
            "driver": ("critic_params", "target_critic_params", "rho_1"),
            "shadow_mu1": (
                "shadow_critic_mu1_params",
                "shadow_target_critic_mu1_params",
                "rho_1",
            ),
            "shadow_muk": (
                "shadow_critic_muk_params",
                "shadow_target_critic_muk_params",
                "rho_K",
            ),
        }
        self.critic_source = critic_source
        self.critic_key, self.target_critic_key, self.route_continuation = source_keys[
            critic_source
        ]
        required = (self.critic_key, self.target_critic_key)
        if self.route_continuation == "rho_K":
            required += ("target_actor_k_params",)
        missing = [key for key in required if key not in self.payload]
        if missing:
            raise ValueError(
                f"checkpoint does not contain {critic_source} state; missing {missing}"
            )
        self._actor_apply = jax.jit(self.actor.apply)
        self._critic_apply = jax.jit(self._critic_values)

    def _infer_action_dim(self) -> int:
        params = self.payload["actor_params"]["params"]
        kernel = np.asarray(params["Dense_0"]["kernel"])
        output_kernel = np.asarray(params["Dense_2"]["kernel"])
        if kernel.shape[0] != self.mean.shape[0]:
            raise ValueError(
                "checkpoint actor input dimension does not match normalization"
            )
        return int(output_kernel.shape[-1])

    def _infer_final_actor(self) -> tuple[str, int]:
        if bool(self.config.get("mpi_four_step")):
            return "actor4_params", 4
        if bool(self.config.get("mpi_three_step")) or bool(
            self.config.get("explicit_three_step")
        ):
            return "actor3_params", 3
        if bool(self.config.get("mpi_two_step")) or bool(
            self.config.get("explicit_two_step")
        ) or bool(self.config.get("fb")):
            return "actor2_params", 2
        name = self.checkpoint.parent.name
        if "_mpi4_" in name:
            return "actor4_params", 4
        if "_mpi3_" in name or "_expl3_" in name:
            return "actor3_params", 3
        if "_mpi2_" in name or "_expl2_" in name or "_fb_" in name:
            return "actor2_params", 2
        return "actor_params", 1

    def normalize(self, observation: Array) -> Array:
        observation = np.asarray(observation, dtype=np.float32)
        return (observation - self.mean) / self.std

    def _apply_actor(self, key: str, observation: Array) -> Array:
        state = self.normalize(observation)[None]
        action = self._actor_apply(self.payload[key], state)
        return np.asarray(action[0], dtype=np.float32)

    def mu1(self, observation: Array, _rng: np.random.Generator | None = None) -> Array:
        return self._apply_actor("actor_params", observation)

    def muk(self, observation: Array, _rng: np.random.Generator | None = None) -> Array:
        return self._apply_actor(self.final_actor_key, observation)

    def _rho(
        self,
        actor_key: str,
        observation: Array,
        rng: np.random.Generator | None,
    ) -> Array:
        if rng is None:
            raise ValueError("rho requires an explicit target-noise RNG")
        action = self._apply_actor(actor_key, observation)
        noise = rng.normal(0.0, self.policy_noise, size=action.shape)
        noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        return np.clip(
            action + noise, -self.max_action, self.max_action
        ).astype(np.float32)

    def rho1(
        self, observation: Array, rng: np.random.Generator | None = None
    ) -> Array:
        return self._rho("target_actor_params", observation, rng)

    def rhok(
        self, observation: Array, rng: np.random.Generator | None = None
    ) -> Array:
        return self._rho("target_actor_k_params", observation, rng)

    def route_policy(
        self, observation: Array, rng: np.random.Generator | None = None
    ) -> Array:
        if self.route_continuation == "rho_K":
            return self.rhok(observation, rng)
        return self.rho1(observation, rng)

    def _critic_values(
        self, params: Any, observations: jax.Array, actions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        q1, q2 = self.critic.apply(params, observations, actions)
        return jnp.squeeze(q1, -1), jnp.squeeze(q2, -1)

    def learned_values(
        self, observation: Array, action1: Array, actionk: Array
    ) -> dict[str, float]:
        states = np.repeat(self.normalize(observation)[None], 2, axis=0)
        actions = np.stack([action1, actionk]).astype(np.float32)
        q1, q2 = self._critic_apply(
            self.payload[self.critic_key], states, actions
        )
        target_q1, target_q2 = self._critic_apply(
            self.payload[self.target_critic_key], states, actions
        )
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        target_q1 = np.asarray(target_q1, dtype=np.float64)
        target_q2 = np.asarray(target_q2, dtype=np.float64)
        qmin = np.minimum(q1, q2)
        target_min = np.minimum(target_q1, target_q2)
        return {
            "qhat_q1_a1": float(q1[0]),
            "qhat_q1_ak": float(q1[1]),
            "delta_qhat_q1": float(q1[1] - q1[0]),
            "qhat_q2_a1": float(q2[0]),
            "qhat_q2_ak": float(q2[1]),
            "delta_qhat_q2": float(q2[1] - q2[0]),
            "qhat_min_a1": float(qmin[0]),
            "qhat_min_ak": float(qmin[1]),
            "delta_qhat_min": float(qmin[1] - qmin[0]),
            "target_q1_a1": float(target_q1[0]),
            "target_q1_ak": float(target_q1[1]),
            "delta_target_q1": float(target_q1[1] - target_q1[0]),
            "target_q2_a1": float(target_q2[0]),
            "target_q2_ak": float(target_q2[1]),
            "delta_target_q2": float(target_q2[1] - target_q2[0]),
            "target_min_a1": float(target_min[0]),
            "target_min_ak": float(target_min[1]),
            "delta_target_min": float(target_min[1] - target_min[0]),
        }


def _set_time_limit_elapsed(env: gym.Env, value: int) -> None:
    current: Any = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_elapsed_steps"):
            current._elapsed_steps = int(value)
            return
        child = getattr(current, "env", None)
        if child is None:
            break
        current = child
    raise RuntimeError("environment has no TimeLimit _elapsed_steps")


def collect_occupancy_states(
    env: gym.Env,
    evaluator: CheckpointEvaluator,
    *,
    episodes: int,
    n_states: int,
    gamma: float,
    base_seed: int,
    sample_seed: int,
) -> list[SampledState]:
    candidates: list[SampledState] = []
    for episode in range(episodes):
        rollout_seed = int(base_seed + episode)
        observation, _ = env.reset(seed=rollout_seed)
        time_index = 0
        done = False
        while not done:
            candidates.append(
                SampledState(
                    snapshot=snapshot_env(env),
                    observation=np.asarray(observation, dtype=np.float32).copy(),
                    episode=episode,
                    time_index=time_index,
                    occupancy_weight=float(gamma**time_index),
                    rollout_seed=rollout_seed,
                )
            )
            action = evaluator.muk(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            time_index += 1
    if not candidates:
        raise RuntimeError("no occupancy states were collected")
    if n_states >= len(candidates):
        return candidates
    weights = np.asarray(
        [state.occupancy_weight for state in candidates], dtype=np.float64
    )
    weights /= np.sum(weights)
    rng = np.random.default_rng(sample_seed)
    indices = rng.choice(len(candidates), size=n_states, replace=False, p=weights)
    return [candidates[int(index)] for index in indices]


def rollout_from_snapshot(
    env: gym.Env,
    snapshot: Any,
    first_action: Array,
    continuation: PolicyFn,
    *,
    gamma: float,
    horizon: int,
    noise_seed: int | None,
    reset_time_limit: bool,
) -> tuple[float, float, int, bool, bool]:
    observation = restore_env(env, snapshot)
    if reset_time_limit:
        _set_time_limit_elapsed(env, 0)
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    discounted = 0.0
    undiscounted = 0.0
    discount = 1.0
    terminated = False
    truncated = False
    action = np.asarray(first_action, dtype=np.float32)
    steps = 0
    for step in range(horizon):
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        discounted += discount * reward
        undiscounted += reward
        steps = step + 1
        if terminated or truncated:
            break
        discount *= gamma
        action = continuation(observation, rng)
    return discounted, undiscounted, steps, bool(terminated), bool(truncated)


def _paired_policy_values(
    env: gym.Env,
    state: SampledState,
    action1: Array,
    actionk: Array,
    continuation: PolicyFn,
    *,
    gamma: float,
    horizon: int,
    reset_time_limit: bool,
    noise_seeds: list[int | None],
) -> dict[str, float]:
    discounted_a1: list[float] = []
    discounted_ak: list[float] = []
    undiscounted_a1: list[float] = []
    undiscounted_ak: list[float] = []
    steps_a1: list[int] = []
    steps_ak: list[int] = []
    for noise_seed in noise_seeds:
        result_a1 = rollout_from_snapshot(
            env,
            state.snapshot,
            action1,
            continuation,
            gamma=gamma,
            horizon=horizon,
            noise_seed=noise_seed,
            reset_time_limit=reset_time_limit,
        )
        result_ak = rollout_from_snapshot(
            env,
            state.snapshot,
            actionk,
            continuation,
            gamma=gamma,
            horizon=horizon,
            noise_seed=noise_seed,
            reset_time_limit=reset_time_limit,
        )
        discounted_a1.append(result_a1[0])
        discounted_ak.append(result_ak[0])
        undiscounted_a1.append(result_a1[1])
        undiscounted_ak.append(result_ak[1])
        steps_a1.append(result_a1[2])
        steps_ak.append(result_ak[2])
    return {
        "discounted_a1": float(np.mean(discounted_a1)),
        "discounted_ak": float(np.mean(discounted_ak)),
        "undiscounted_a1": float(np.mean(undiscounted_a1)),
        "undiscounted_ak": float(np.mean(undiscounted_ak)),
        "steps_a1_mean": float(np.mean(steps_a1)),
        "steps_ak_mean": float(np.mean(steps_ak)),
    }


def audit_state(
    env: gym.Env,
    evaluator: CheckpointEvaluator,
    state: SampledState,
    *,
    state_index: int,
    gamma: float,
    horizon: int,
    smoothing_rollouts: int,
    audit_seed: int,
    reset_time_limit: bool,
    dead_zone: float,
) -> dict[str, Any]:
    action1 = evaluator.mu1(state.observation)
    actionk = evaluator.muk(state.observation)
    learned = evaluator.learned_values(state.observation, action1, actionk)
    base_noise_seed = int(audit_seed + 1_000_003 * state_index)
    rho_noise_seeds = [
        base_noise_seed + repeat for repeat in range(smoothing_rollouts)
    ]
    values = {
        "rho": _paired_policy_values(
            env,
            state,
            action1,
            actionk,
            evaluator.route_policy,
            gamma=gamma,
            horizon=horizon,
            reset_time_limit=reset_time_limit,
            noise_seeds=rho_noise_seeds,
        ),
        "mu1": _paired_policy_values(
            env,
            state,
            action1,
            actionk,
            evaluator.mu1,
            gamma=gamma,
            horizon=horizon,
            reset_time_limit=reset_time_limit,
            noise_seeds=[None],
        ),
        "muk": _paired_policy_values(
            env,
            state,
            action1,
            actionk,
            evaluator.muk,
            gamma=gamma,
            horizon=horizon,
            reset_time_limit=reset_time_limit,
            noise_seeds=[None],
        ),
    }
    row: dict[str, Any] = {
        "state_index": int(state_index),
        "episode": int(state.episode),
        "time_index": int(state.time_index),
        "rollout_seed": int(state.rollout_seed),
        "occupancy_weight": float(state.occupancy_weight),
        "action_l2_mu1_muk": float(np.linalg.norm(actionk - action1)),
        "action_mu1": json.dumps(action1.tolist()),
        "action_muk": json.dumps(actionk.tolist()),
    }
    row.update(learned)
    for policy_name, policy_values in values.items():
        for key, value in policy_values.items():
            row[f"qmc_{policy_name}_{key}"] = value

    row["delta_mc_rho"] = (
        row["qmc_rho_discounted_ak"] - row["qmc_rho_discounted_a1"]
    )
    row["advantage_mc_mu1"] = (
        row["qmc_mu1_discounted_ak"] - row["qmc_mu1_discounted_a1"]
    )
    route_minus_mu1 = (
        row["qmc_rho_discounted_ak"] - row["qmc_mu1_discounted_ak"]
    )
    row["route_continuation_minus_mu1_at_ak"] = route_minus_mu1
    row["target_lag_smoothing_at_ak"] = (
        route_minus_mu1 if evaluator.route_continuation == "rho_1" else float("nan")
    )
    row["deployment_continuation_at_ak"] = (
        row["qmc_muk_discounted_ak"] - row["qmc_mu1_discounted_ak"]
    )
    row["critic_error_rho_at_ak"] = (
        row["qhat_q1_ak"] - row["qmc_rho_discounted_ak"]
    )
    row["critic_abs_error_rho_at_ak"] = abs(row["critic_error_rho_at_ak"])
    row["critic_symmetric_relative_error_rho_at_ak"] = (
        row["critic_abs_error_rho_at_ak"]
        / (
            abs(row["qhat_q1_ak"])
            + abs(row["qmc_rho_discounted_ak"])
            + 1e-12
        )
    )
    row["actual_policy_value_difference"] = (
        row["qmc_muk_discounted_ak"] - row["qmc_mu1_discounted_a1"]
    )
    row["ranking_error_rho"] = row["delta_qhat_q1"] - row["delta_mc_rho"]
    row["wrong_ranking_outside_dead_zone"] = int(
        row["delta_qhat_q1"] > dead_zone
        and row["delta_mc_rho"] < -dead_zone
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty MC audit")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, float | int | None]:
    keys = (
        "delta_qhat_q1",
        "delta_mc_rho",
        "advantage_mc_mu1",
        "route_continuation_minus_mu1_at_ak",
        "target_lag_smoothing_at_ak",
        "deployment_continuation_at_ak",
        "critic_error_rho_at_ak",
        "critic_abs_error_rho_at_ak",
        "critic_symmetric_relative_error_rho_at_ak",
        "actual_policy_value_difference",
        "ranking_error_rho",
        "wrong_ranking_outside_dead_zone",
    )
    summary: dict[str, float | int | None] = {"n_states": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{key}_mean"] = float(np.mean(finite)) if finite.size else None
        summary[f"{key}_p50"] = (
            float(np.percentile(finite, 50)) if finite.size else None
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--critic-source",
        choices=CheckpointEvaluator.CRITIC_SOURCES,
        default="driver",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--n-states", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--smoothing-rollouts", type=int, default=8)
    parser.add_argument("--rollout-seed", type=int, default=20260829)
    parser.add_argument("--sample-seed", type=int, default=20260830)
    parser.add_argument("--audit-seed", type=int, default=20260831)
    parser.add_argument("--dead-zone", type=float, default=0.0)
    parser.add_argument(
        "--preserve-time-limit",
        action="store_true",
        help=(
            "Preserve the sampled state's remaining episode clock. The default "
            "resets TimeLimit for a fixed-horizon critic-calibration rollout."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.n_states < 1:
        raise ValueError("episodes and n-states must be positive")
    if args.horizon < 1 or args.smoothing_rollouts < 1:
        raise ValueError("horizon and smoothing-rollouts must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")

    evaluator = CheckpointEvaluator(args.checkpoint, critic_source=args.critic_source)
    config_env = evaluator.config.get("env")
    if config_env and config_env != args.env:
        raise ValueError(
            f"checkpoint env {config_env!r} does not match --env {args.env!r}"
        )
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} is not empty; pass --overwrite to replace audit files"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(EVAL_ENV[_domain(args.env)])
    try:
        states = collect_occupancy_states(
            env,
            evaluator,
            episodes=args.episodes,
            n_states=args.n_states,
            gamma=args.gamma,
            base_seed=args.rollout_seed,
            sample_seed=args.sample_seed,
        )
        rows = []
        for index, state in enumerate(states):
            print(
                f"[mc] state={index + 1}/{len(states)} "
                f"episode={state.episode} t={state.time_index}",
                flush=True,
            )
            row = {
                "env": args.env,
                "checkpoint": str(evaluator.checkpoint),
                "checkpoint_step": int(evaluator.payload.get("step", -1)),
                "training_seed": int(evaluator.config.get("seed", -1)),
                "hops": int(evaluator.hops),
                "critic_source": evaluator.critic_source,
                "route_continuation": evaluator.route_continuation,
                "gamma": float(args.gamma),
                "horizon": int(args.horizon),
                "preserve_time_limit": bool(args.preserve_time_limit),
                "smoothing_rollouts": int(args.smoothing_rollouts),
                "dead_zone": float(args.dead_zone),
            }
            row.update(
                audit_state(
                    env,
                    evaluator,
                    state,
                    state_index=index,
                    gamma=args.gamma,
                    horizon=args.horizon,
                    smoothing_rollouts=args.smoothing_rollouts,
                    audit_seed=args.audit_seed,
                    reset_time_limit=not args.preserve_time_limit,
                    dead_zone=args.dead_zone,
                )
            )
            rows.append(row)
    finally:
        env.close()

    _write_csv(out_dir / "per_state.csv", rows)
    with (out_dir / "sampled_states.pkl").open("wb") as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = summarize_rows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "env": args.env,
        "checkpoint": str(evaluator.checkpoint),
        "checkpoint_step": int(evaluator.payload.get("step", -1)),
        "training_seed": int(evaluator.config.get("seed", -1)),
        "hops": int(evaluator.hops),
        "critic_source": evaluator.critic_source,
        "route_continuation": evaluator.route_continuation,
        "continuation_policies": [evaluator.route_continuation, "mu_1", "mu_K"],
        "first_actions": ["a_1=mu_1(s)", "a_K=mu_K(s)"],
        "critic_readout": f"{evaluator.critic_source} online Q1",
        "feeds_actor_updates": evaluator.critic_source == "driver",
        "gamma": float(args.gamma),
        "horizon": int(args.horizon),
        "preserve_time_limit": bool(args.preserve_time_limit),
        "episodes": int(args.episodes),
        "n_states": len(states),
        "smoothing_rollouts": int(args.smoothing_rollouts),
        "rollout_seed": int(args.rollout_seed),
        "sample_seed": int(args.sample_seed),
        "audit_seed": int(args.audit_seed),
        "dead_zone": float(args.dead_zone),
        "simulator": EVAL_ENV[_domain(args.env)],
        "software_versions": {
            "gymnasium": gym.__version__,
            "jax": jax.__version__,
            "mujoco": version("mujoco"),
            "numpy": np.__version__,
        },
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cpu_environment": {
            key: os.environ.get(key, "")
            for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_PLATFORM_NAME")
        },
        "interpretation": (
            "post-hoc simulator reference, not used for training or model "
            "selection. A shadow source is a route-only critic readout and "
            "does not feed back into the actor trajectory"
        ),
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] states={len(rows)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()

