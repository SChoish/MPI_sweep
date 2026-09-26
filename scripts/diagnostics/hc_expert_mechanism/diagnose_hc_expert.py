#!/usr/bin/env python3
"""Read-only paired HalfCheetah-expert policy and critic diagnosis.

Requires the SAME Python environment and train_td3bc.py used to evaluate the
K=4 checkpoints. This script neither trains nor writes into run directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ENV = "halfcheetah-expert-v2"
BRANCHES = ("recenter", "fixed_ref")
ACTORS = (3, 4)


def parse_ints(value: str) -> list[int]:
    result = [int(x) for x in value.split(",")]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("provide unique comma-separated integers")
    return result


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows produced for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_pair(root: Path, total_horizon: int, seed: int) -> dict:
    pair = {}
    for branch in BRANCHES:
        run = root / f"{ENV}_tau{total_horizon:g}_shared_{branch}4_seed{seed}"
        path = run / "params_1000000.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # Only open your own trusted checkpoints.
        if int(payload["step"]) != 1_000_000 or len(payload["actors_params"]) != 4:
            raise ValueError(f"unexpected checkpoint step/actor count: {path}")
        config = payload["config"]
        if (config["env"] != ENV or float(config["tau"]) != total_horizon
                or int(config["seed"]) != seed
                or config["deployment_branch"] != branch
                or config["integrator"] != "implicit"):
            raise ValueError(f"checkpoint config mismatch: {path}")
        pair[branch] = (payload, run)
    left, right = (pair[b][0] for b in BRANCHES)
    if not np.array_equal(left["mean"], right["mean"]) or not np.array_equal(
        left["std"], right["std"]
    ):
        raise ValueError(f"normalization mismatch at T={total_horizon}, seed={seed}")
    import jax

    for key in ("critic_params", "target_actor_params", "target_critic_params"):
        lvals = jax.tree_util.tree_leaves(left[key])
        rvals = jax.tree_util.tree_leaves(right[key])
        if len(lvals) != len(rvals) or any(
            not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(lvals, rvals)
        ):
            raise ValueError(f"shared {key} differs at T={total_horizon}, seed={seed}")
    for index in (0, 1):
        lvals = jax.tree_util.tree_leaves(left["actors_params"][index])
        rvals = jax.tree_util.tree_leaves(right["actors_params"][index])
        if any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(lvals, rvals)):
            raise ValueError(f"actor {index + 1} differs at T={total_horizon}, seed={seed}")
    return pair


def make_models(td3, payload):
    import jax
    import jax.numpy as jnp

    # HalfCheetah-v4 has six actions. Check the live environment below.
    action_dim = 6
    actor = td3.Actor(action_dim=action_dim, max_action=float(payload["max_action"]))
    critic = td3.TwinCritic()
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)

    @jax.jit
    def actor_batch(params, state):
        return actor.apply(params, state)

    @jax.jit
    def q_batch(params, state, action):
        return critic.apply(params, state, action)

    def act(params, obs):
        state = ((np.asarray(obs, dtype=np.float32) - mean) / std)[None, :]
        return np.asarray(actor_batch(params, jnp.asarray(state)))[0]

    def q(obs, action):
        state = ((np.asarray(obs, dtype=np.float32) - mean) / std)[None, :]
        q1, q2 = q_batch(payload["critic_params"], jnp.asarray(state),
                         jnp.asarray(action[None, :], dtype=jnp.float32))
        return float(np.asarray(q1).squeeze()), float(np.asarray(q2).squeeze())

    return act, q


def replay_mc(td3, env, snapshot, first_action, pi1, gamma: float) -> float:
    # Reset the TimeLimit wrapper; restore a state sampled from a live rollout.
    td3._reset_obs(env, snapshot["episode_seed"])
    env.unwrapped.set_state(snapshot["qpos"], snapshot["qvel"])
    restored = np.asarray(env.unwrapped._get_obs(), dtype=np.float32)
    if not np.allclose(restored, snapshot["obs"], rtol=1e-5, atol=1e-5):
        raise RuntimeError("state clone does not reproduce observation")
    action = first_action
    total = 0.0
    factor = 1.0
    for _ in range(snapshot["t"], 1000):
        obs, reward, done = td3._step_env(env, action)
        total += factor * reward
        if done:
            break
        factor *= gamma
        action = pi1(obs)
    return total


def rollout(td3, env, check_env, payload, policy, ep_seed: int,
            snapshot_times: set[int]):
    obs = td3._reset_obs(env, ep_seed)
    observations = []
    snapshots = []
    ep_return = 0.0
    for t in range(1000):
        action = policy(obs)
        if t in snapshot_times:
            snapshot = {"t": t, "episode_seed": ep_seed,
                        "qpos": np.asarray(env.unwrapped.data.qpos).copy(),
                        "qvel": np.asarray(env.unwrapped.data.qvel).copy(),
                        "obs": np.asarray(obs, dtype=np.float32).copy()}
            # Verify full-state restoration by reproducing the NEXT step.
            td3._reset_obs(check_env, ep_seed)
            check_env.unwrapped.set_state(snapshot["qpos"], snapshot["qvel"])
            obs_check = np.asarray(check_env.unwrapped._get_obs(), dtype=np.float32)
            if not np.allclose(obs_check, snapshot["obs"], rtol=1e-5, atol=1e-5):
                raise RuntimeError("clone observation mismatch")
            next_check, reward_check, done_check = td3._step_env(check_env, action)
            snapshots.append((snapshot, next_check, reward_check, done_check))
        next_obs, reward, done = td3._step_env(env, action)
        if t in snapshot_times:
            _, obs_check_next, reward_check, done_check = snapshots[-1]
            if (not np.allclose(next_obs, obs_check_next, rtol=1e-5, atol=1e-5)
                    or not np.isclose(reward, reward_check, rtol=1e-5, atol=1e-5)
                    or done != done_check):
                raise RuntimeError("clone next transition mismatch; MC invalid")
        # Full-return failure can reflect many steps, not only one action.
        xvel = (float(env.unwrapped.data.qvel[0])
                if hasattr(env.unwrapped.data, "qvel") else float("nan"))
        observations.append((t, reward, xvel, float(np.mean(np.abs(action) > .99))))
        ep_return += reward
        if done:
            break
        obs = next_obs
    return ep_return, observations, [x[0] for x in snapshots]


def checkpoint_score(run: Path, column: str) -> float:
    with (run / "eval.csv").open(newline="", encoding="utf-8") as handle:
        candidates = [r for r in csv.DictReader(handle) if int(float(r["step"])) == 1_000_000]
    if not candidates or column not in candidates[-1]:
        raise ValueError(f"missing final {column} in {run / 'eval.csv'}")
    return float(candidates[-1][column])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="MPI_sweep_recenter_k4 source checkout")
    parser.add_argument("--checkpoints", type=Path, required=True, help=".../recenter_fixed_ref_k4/full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=parse_ints, default=[4, 7, 10])
    parser.add_argument("--seeds", type=parse_ints, default=[0, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--mc-episodes", type=parse_ints, default=[0])
    parser.add_argument("--mc-times", type=parse_ints, default=[0, 50, 150, 300])
    parser.add_argument("--discount", type=float, default=.99)
    args = parser.parse_args()
    if args.episodes != 10:
        raise ValueError("10 episodes required to verify existing eval.csv scores")
    if any(i < 0 or i >= args.episodes for i in args.mc_episodes):
        raise ValueError("MC episode indices must be within evaluated episodes")
    if any(t < 0 or t >= 1000 for t in args.mc_times):
        raise ValueError("MC times must be between 0 and 999")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import train_td3bc as td3

    episode_rows, block_rows, mc_rows = [], [], []
    for total_horizon in args.horizons:
        for seed in args.seeds:
            print(f"T={total_horizon} seed={seed}", flush=True)
            pair = read_pair(args.checkpoints, total_horizon, seed)
            left = pair["recenter"][0]
            right = pair["fixed_ref"][0]
            models = {b: make_models(td3, pair[b][0]) for b in BRANCHES}
            live, clone = td3._make_eval_env(ENV), td3._make_eval_env(ENV)
            try:
                if live.action_space.shape != (6,):
                    raise RuntimeError("unexpected action dimension")
                for depth in (1, *ACTORS):
                    for branch in (("recenter",) if depth == 1 else BRANCHES):
                        payload, run = pair[branch]
                        act, _ = models[branch]
                        policy = lambda obs, p=payload["actors_params"][depth-1], a=act: a(p, obs)
                        for ep in range(args.episodes):
                            ep_seed = seed + 100 + ep
                            times = set(args.mc_times) if ep in args.mc_episodes and depth in ACTORS else set()
                            ret, records, snapshots = rollout(
                                td3, live, clone, payload, policy, ep_seed, times)
                            episode_rows.append({"T": total_horizon, "seed": seed,
                                "actor": depth, "branch": branch, "episode": ep,
                                "return": ret, "score": td3.d4rl_normalized_score(ENV, ret),
                                "length": len(records)})
                            for block_start in range(0, 1000, 100):
                                block = [x for x in records if block_start <= x[0] < block_start + 100]
                                if block:
                                    block_rows.append({"T": total_horizon, "seed": seed,
                                        "actor": depth, "branch": branch, "episode": ep,
                                        "block_start": block_start, "count": len(block),
                                        "reward_mean": float(np.mean([x[1] for x in block])),
                                        "velocity_mean": float(np.mean([x[2] for x in block])),
                                        "saturation_mean": float(np.mean([x[3] for x in block]))})
                            # Evaluate both counterfactual actions at the SAME visited state.
                            for snapshot in snapshots:
                                obs = snapshot["obs"]
                                actions = {b: models[b][0](pair[b][0]["actors_params"][depth-1], obs)
                                           for b in BRANCHES}
                                q_values = {b: models[b][1](obs, actions[b]) for b in BRANCHES}
                                pi1 = lambda o: models["recenter"][0](left["actors_params"][0], o)
                                targets = {b: replay_mc(td3, clone, snapshot, actions[b], pi1,
                                                        args.discount) for b in BRANCHES}
                                mc_rows.append({"T": total_horizon, "seed": seed,
                                    "actor": depth, "source_branch": branch,
                                    "episode": ep, "t": snapshot["t"],
                                    "q1_recenter": q_values["recenter"][0],
                                    "q1_fixed": q_values["fixed_ref"][0],
                                    "q2_recenter": q_values["recenter"][1],
                                    "q2_fixed": q_values["fixed_ref"][1],
                                    "mc_recenter": targets["recenter"],
                                    "mc_fixed": targets["fixed_ref"],
                                    "q1_delta": q_values["recenter"][0] - q_values["fixed_ref"][0],
                                    "mc_delta": targets["recenter"] - targets["fixed_ref"]})
                        if depth in (1, 4):
                            rows = [r for r in episode_rows if r["T"] == total_horizon
                                    and r["seed"] == seed and r["actor"] == depth
                                    and r["branch"] == branch]
                            observed = td3.d4rl_normalized_score(ENV, np.mean([r["return"] for r in rows]))
                            expected = checkpoint_score(run, "d4rl_score" if depth == 1 else "d4rl_pi4")
                            if abs(observed - expected) > .05:
                                raise RuntimeError(f"evaluation mismatch: T={total_horizon} seed={seed} "
                                                   f"{branch} pi{depth}: {observed:.3f} vs {expected:.3f}")
                            print(f"  {branch} pi{depth}: {observed:.2f} (eval.csv verified)", flush=True)
            finally:
                live.close()
                clone.close()
            # Write partial output after each pair; completed pairs survive interruption.
            save_rows(args.output / "episodes.csv", episode_rows)
            save_rows(args.output / "time_blocks.csv", block_rows)
            if mc_rows:
                save_rows(args.output / "counterfactual_mc.csv", mc_rows)
    with (args.output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump({"env": ENV, "horizons": args.horizons, "seeds": args.seeds,
                   "episodes": args.episodes, "mc_episodes": args.mc_episodes,
                   "mc_times": args.mc_times, "discount": args.discount,
                   "checkpoint_step": 1_000_000,
                   "target": "one first action, then common pi1; discount measured from sampled state",
                   "independent_unit": "training seed within horizon; episodes and states repeated measures"},
                  handle, indent=2)
    print(f"done; CSV files: {args.output}", flush=True)


if __name__ == "__main__":
    main()
