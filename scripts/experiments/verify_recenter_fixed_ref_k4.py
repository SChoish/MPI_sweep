#!/usr/bin/env python3
"""Check a trusted local K=4 shared-driver checkpoint pair and final eval rows.

Only load pickle files produced by your own train_td3bc.py runs. This script
does not modify checkpoints or start training.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np

BRANCHES = ("recenter", "fixed_ref")


def run_dir(save_dir: Path, env: str, tau: float, seed: int, branch: str) -> Path:
    return save_dir / f"{env}_tau{tau:g}_shared_{branch}4_seed{seed}"


def same_tree(left: object, right: object) -> bool:
    """Exact, shape-aware equality for checkpoint arrays and nested containers."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            same_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    try:
        a, b = np.asarray(left), np.asarray(right)
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    except (TypeError, ValueError):
        return left == right


def load_trusted_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_final_score(path: Path, step: int) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["step"]) == step]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one final evaluation at step {step}: {path}")
    return float(rows[0]["d4rl_score"]), float(rows[0]["d4rl_pi4"])


def verify_pair(save_dir: Path, env: str, tau: float, seed: int, step: int) -> dict:
    paths = {branch: run_dir(save_dir, env, tau, seed, branch) for branch in BRANCHES}
    checkpoints = {
        branch: load_trusted_checkpoint(path / f"params_{step}.pkl")
        for branch, path in paths.items()
    }
    recenter, fixed = (checkpoints[branch] for branch in BRANCHES)
    if any(int(checkpoint["step"]) != step for checkpoint in checkpoints.values()):
        raise ValueError("checkpoint step differs from expected final step")

    for branch, checkpoint in checkpoints.items():
        config = checkpoint["config"]
        if (
            config["method"] != "shared"
            or config["deployment_branch"] != branch
            or config["integrator"] != "implicit"
            or int(config["mpi_steps"]) != 4
            or int(config["seed"]) != seed
            or float(config["tau"]) != tau
            or config["env"] != env
        ):
            raise ValueError(f"checkpoint metadata differs from protocol: {branch}")
        if int(config["policy_freq"]) <= 0 or step % int(config["policy_freq"]):
            raise ValueError(f"nonintegral scheduled actor-update count: {branch}")
        actor_updates = step // int(config["policy_freq"])
        if len(checkpoint["actors_steps"]) != 4 or any(
            int(s) != actor_updates for s in checkpoint["actors_steps"]
        ):
            raise ValueError(f"actor optimizer update counts are not matched: {branch}")
        if len(checkpoint["actors_params"]) != 4 or len(checkpoint["actors_opt_states"]) != 4:
            raise ValueError(f"actor parameter or optimizer count differs: {branch}")
        if int(checkpoint["critic_step"]) != step:
            raise ValueError(f"critic update count differs: {branch}")

    ignored_config = {"deployment_branch", "save_dir"}
    for key in set(recenter["config"]) | set(fixed["config"]):
        if key not in ignored_config and recenter["config"].get(key) != fixed["config"].get(key):
            raise ValueError(f"training configurations differ beyond branch: {key}")

    identical = (
        "rng", "mean", "std", "max_action", "policy_noise", "noise_clip",
        "critic_params", "critic_opt_state", "critic_step",
        "target_actor_params", "target_critic_params",
    )
    for key in identical:
        if not same_tree(recenter[key], fixed[key]):
            raise ValueError(f"shared checkpoint state differs: {key}")
    for key in ("actors_params", "actors_steps", "actors_opt_states"):
        for index in (0, 1):
            if not same_tree(recenter[key][index], fixed[key][index]):
                raise ValueError(f"shared pi_{index + 1} state differs: {key}")

    first_a, final_a = read_final_score(paths["recenter"] / "eval.csv", step)
    first_b, final_b = read_final_score(paths["fixed_ref"] / "eval.csv", step)
    if first_a != first_b:
        raise ValueError("first-actor scores differ despite identical parameters")
    if not np.isfinite([first_a, final_a, final_b]).all():
        raise ValueError("nonfinite final score; retain failed run, do not silently drop")
    return {
        "status": "pair_verified", "env": env, "T": tau, "K": 4,
        "seed": seed, "step": step, "critic_updates": step,
        "actor_updates_per_actor": step // int(recenter["config"]["policy_freq"]),
        "first_score": first_a, "recenter_score": final_a,
        "fixed_ref_score": final_b, "paired_delta": final_a - final_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_pair(args.save_dir, args.env, args.tau, args.seed, args.step)))


if __name__ == "__main__":
    main()
