#!/usr/bin/env python3
"""Read-only policy-switching pilot for the matched TD3+BC K=4 checkpoints.

At each episode, act with one pi4 branch for a fixed prefix, then switch
permanently to the other branch. Training, critics, and checkpoints are frozen.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from diagnose_hc_expert import ENV, BRANCHES, checkpoint_score, make_models, parse_ints, read_pair


def run_episode(td3, env, act_first, act_second, prefix: int, eval_seed: int):
    obs = td3._reset_obs(env, eval_seed)
    total, velocities, saturations = 0.0, [], []
    for t in range(1000):
        action = (act_first if t < prefix else act_second)(obs)
        obs, reward, done = td3._step_env(env, action)
        total += reward
        velocities.append(float(env.unwrapped.data.qvel[0]))
        saturations.append(float(np.mean(np.abs(action) > .99)))
        if done:
            break
    return total, len(velocities), float(np.mean(velocities)), float(np.mean(saturations))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizons", type=parse_ints, default=[4, 10])
    p.add_argument("--seeds", type=parse_ints, default=[0, 2])
    p.add_argument("--prefixes", type=parse_ints, default=[0, 25, 100, 300, 1000])
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    if args.episodes != 10 or 0 not in args.prefixes or 1000 not in args.prefixes:
        p.error("ten episodes and both 0/1000 controls are required")
    if any(x < 0 or x > 1000 for x in args.prefixes):
        p.error("prefix length must lie in [0,1000]")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import train_td3bc as td3

    rows = []
    for T in args.horizons:
        for seed in args.seeds:
            pair = read_pair(args.checkpoints, T, seed)
            policies = {}
            for branch in BRANCHES:
                payload = pair[branch][0]
                act, _ = make_models(td3, payload)
                policies[branch] = lambda obs, a=act, params=payload["actors_params"][3]: a(params, obs)
            env = td3._make_eval_env(ENV)
            try:
                for first, second in (("recenter", "fixed_ref"), ("fixed_ref", "recenter")):
                    for prefix in args.prefixes:
                        cell = []
                        for episode in range(args.episodes):
                            ep_seed = seed + 100 + episode
                            ret, length, velocity, saturation = run_episode(
                                td3, env, policies[first], policies[second], prefix, ep_seed)
                            score = td3.d4rl_normalized_score(ENV, ret)
                            rows.append({"T": T, "seed": seed, "episode": episode,
                                "first_branch": first, "second_branch": second,
                                "prefix": prefix, "return": ret, "d4rl_score": score,
                                "length": length, "mean_velocity": velocity,
                                "mean_saturation": saturation})
                            cell.append(ret)
                        # Endpoints reproduce the original full-actor evaluation.
                        if prefix in (0, 1000):
                            branch = second if prefix == 0 else first
                            observed = td3.d4rl_normalized_score(ENV, np.mean(cell))
                            expected = checkpoint_score(pair[branch][1], "d4rl_pi4")
                            if abs(observed - expected) > .05:
                                raise RuntimeError(f"pi4 checkpoint score mismatch: T={T}, seed={seed}, "
                                                   f"branch={branch}: {observed:.3f} vs {expected:.3f}")
                        print(f"T={T} seed={seed} {first}->{second} "
                              f"prefix={prefix} score={td3.d4rl_normalized_score(ENV, np.mean(cell)):.2f}",
                              flush=True)
            finally:
                env.close()
            # Keep fully verified training-seed pairs if later pairs are interrupted.
            with (args.output / "switch_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
                w = csv.DictWriter(handle, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("completed; group by T, seed, first_branch, second_branch, prefix", flush=True)


if __name__ == "__main__":
    main()
