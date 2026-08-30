#!/usr/bin/env python3
"""Extract finite-step and actor-critic diagnostics across saved checkpoints.

This complements ``dump_mpi_frontier_geometry.py``.  It uses a deterministic
validation subset and reconstructs quantities available from checkpoints:

* hop-level realized movement, Q scale, local linearization error, saturation,
  stationarity residual, and post-hoc validation proximal margin;
* critic-coupled versus final-policy displacement;
* deterministic TD-residual and Q-scale trajectories;
* return-state occupancy and hysteretic switch counts from ``eval.csv``.

The original stochastic actor minibatch is not stored in checkpoints, so this
script deliberately does not label any post-hoc quantity as a train margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import numpy as np  # noqa: E402

from scripts.dump_mpi_frontier_geometry import (  # noqa: E402
    EPS,
    METHODS,
    _actor_path,
    _critic_path,
    _hop_row,
    _quantiles,
    _run_tag,
    _stable_seed,
    _training_diagnostics,
    _tree_params,
    _write_csv,
)
from train_td3bc import Actor, TwinCritic, load_checkpoint, load_transition  # noqa: E402


DEFAULT_ENVS = ("hopper-medium-replay-v2",)
DEFAULT_TAUS = (20.0, 24.0, 28.0, 34.0, 40.0)
CHECKPOINT_RE = re.compile(r"params_(\d+)\.pkl$")


def _checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a checkpoint path: {path}")
    return int(match.group(1))


def _checkpoint_paths(run_dir: Path, steps: set[int] | None) -> list[Path]:
    paths = []
    for path in run_dir.glob("params_*.pkl"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if steps is None or step in steps:
            paths.append(path)
    return sorted(paths, key=_checkpoint_step)


def _read_evaluations(
    run_dir: Path, score_key: str
) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    path = run_dir / "eval.csv"
    if not path.is_file():
        return [], {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    usable = [row for row in rows if row.get(score_key)]
    return usable, {int(row["step"]): row for row in usable}


def _hysteretic_switches(
    scores: list[float], *, enter: float, leave: float, high: bool
) -> int:
    """Count state changes using separate enter/leave thresholds."""
    active = False
    switches = 0
    for score in scores:
        should_enter = score >= enter if high else score < enter
        should_leave = score < leave if high else score >= leave
        if not active and should_enter:
            active = True
            switches += 1
        elif active and should_leave:
            active = False
            switches += 1
    return switches


def _stability_row(
    *,
    env_name: str,
    tau: float,
    seed: int,
    method: str,
    rows: list[dict[str, str]],
    score_key: str,
) -> dict[str, Any]:
    scores = [float(row[score_key]) for row in rows]
    result: dict[str, Any] = {
        "env": env_name,
        "tau": float(tau),
        "seed": int(seed),
        "method": method,
        "eval_points": len(scores),
    }
    if not scores:
        return result
    result.update(
        {
            "final_score": scores[-1],
            "tail5_score_mean": float(np.mean(scores[-5:])),
            "score_time_mean": float(np.mean(scores)),
            "score_time_std": float(np.std(scores)),
            "high_occupancy_ge80": float(np.mean(np.asarray(scores) >= 80.0)),
            "collapse_occupancy_lt20": float(np.mean(np.asarray(scores) < 20.0)),
            "high_state_switches_h80_l60": _hysteretic_switches(
                scores, enter=80.0, leave=60.0, high=True
            ),
            "collapse_state_switches_e20_l40": _hysteretic_switches(
                scores, enter=20.0, leave=40.0, high=False
            ),
        }
    )
    return result


def _displacement_diagnostics(actions: list[np.ndarray]) -> dict[str, float]:
    action_dim = int(actions[0].shape[-1])
    critic = np.sum(np.square(actions[1] - actions[0]), axis=-1) / float(
        action_dim
    )
    final = np.sum(np.square(actions[-1] - actions[0]), axis=-1) / float(
        action_dim
    )
    result = {}
    result.update(_quantiles(critic, "critic_displacement_sq_mean_metric"))
    result.update(_quantiles(final, "final_displacement_sq_mean_metric"))
    result["decoupling_displacement_ratio"] = float(
        np.sqrt(np.mean(final) / (np.mean(critic) + EPS))
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=os.environ.get("RESULTS_ROOT", str(_ROOT))
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR", str(_ROOT / "data"))
    )
    parser.add_argument(
        "--out-dir", default=str(_ROOT / "checkpoint_dynamics")
    )
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=["td3"],
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="Saved checkpoint steps to use; default is every available step.",
    )
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260828)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    requested_steps = set(args.steps) if args.steps else None
    critic_model = TwinCritic()
    checkpoint_rows: list[dict[str, Any]] = []
    hop_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for env_name in args.envs:
        data, _, _ = load_transition(env_name, Path(args.data_dir), normalize=True)
        n_available = int(data.observations.shape[0])
        rng = np.random.default_rng(_stable_seed(env_name, args.sample_seed))
        indices = rng.choice(
            n_available,
            size=min(int(args.n_states), n_available),
            replace=False,
        )
        common_dir = out_root / "common_batches"
        common_dir.mkdir(parents=True, exist_ok=True)
        np.save(common_dir / f"{env_name}_indices.npy", indices.astype(np.int64))
        states = np.asarray(data.observations[indices], dtype=np.float32)
        dataset_actions = np.asarray(data.actions[indices], dtype=np.float32)
        next_states = np.asarray(data.next_observations[indices], dtype=np.float32)
        rewards = np.asarray(data.rewards[indices], dtype=np.float32)
        not_dones = np.asarray(data.not_dones[indices], dtype=np.float32)

        for tau in args.taus:
            for seed in args.seeds:
                for method in args.methods:
                    result_dir, _, total_hops, score_key = METHODS[method]
                    tag = _run_tag(method, env_name, tau, seed)
                    run_dir = root / result_dir / tag
                    eval_rows, eval_by_step = _read_evaluations(
                        run_dir, score_key
                    )
                    stability_rows.append(
                        _stability_row(
                            env_name=env_name,
                            tau=tau,
                            seed=seed,
                            method=method,
                            rows=eval_rows,
                            score_key=score_key,
                        )
                    )
                    checkpoints = _checkpoint_paths(run_dir, requested_steps)
                    if not checkpoints:
                        missing.append(str(run_dir))
                        print(f"[missing] {run_dir}", flush=True)
                        continue

                    previous_checkpoint_actions: list[np.ndarray] | None = None
                    for checkpoint in checkpoints:
                        checkpoint_step = _checkpoint_step(checkpoint)
                        print(
                            f"[checkpoint] {tag} step={checkpoint_step}",
                            flush=True,
                        )
                        payload = load_checkpoint(checkpoint)
                        max_action = float(payload.get("max_action", 1.0))
                        actor = Actor(
                            action_dim=int(dataset_actions.shape[-1]),
                            max_action=max_action,
                        )
                        actions = _actor_path(
                            payload,
                            method,
                            actor,
                            states,
                            dataset_actions,
                        )
                        critic_params = _tree_params(payload, "critic_params")
                        q1s, q2s, grads = _critic_path(
                            critic_model, critic_params, states, actions
                        )
                        eval_row = eval_by_step.get(checkpoint_step, {})
                        row: dict[str, Any] = {
                            "env": env_name,
                            "tau": float(tau),
                            "seed": int(seed),
                            "method": method,
                            "total_hops": int(total_hops),
                            "checkpoint_step": checkpoint_step,
                            "checkpoint": str(checkpoint),
                        }
                        if eval_row.get(score_key):
                            row["d4rl_score"] = float(eval_row[score_key])
                        if eval_row.get("critic_loss"):
                            row["critic_loss"] = float(eval_row["critic_loss"])
                        row.update(
                            _training_diagnostics(
                                payload,
                                critic_model,
                                states,
                                dataset_actions,
                                next_states,
                                rewards,
                                not_dones,
                            )
                        )
                        if previous_checkpoint_actions is not None:
                            action_dim = int(dataset_actions.shape[-1])
                            critic_drift = np.sum(
                                np.square(
                                    actions[1]
                                    - previous_checkpoint_actions[1]
                                ),
                                axis=-1,
                            ) / float(action_dim)
                            final_drift = np.sum(
                                np.square(
                                    actions[-1]
                                    - previous_checkpoint_actions[-1]
                                ),
                                axis=-1,
                            ) / float(action_dim)
                            row.update(
                                _quantiles(
                                    critic_drift,
                                    "critic_policy_checkpoint_drift",
                                )
                            )
                            row.update(
                                _quantiles(
                                    final_drift,
                                    "final_policy_checkpoint_drift",
                                )
                            )
                        row.update(_displacement_diagnostics(actions))
                        row["q_final_abs_mean"] = float(
                            np.mean(np.abs(q1s[-1]))
                        )
                        row["q_final_abs_p99"] = float(
                            np.percentile(np.abs(q1s[-1]), 99)
                        )
                        checkpoint_rows.append(row)
                        previous_checkpoint_actions = actions

                        for hop in range(1, len(actions)):
                            hop_row, _ = _hop_row(
                                env_name=env_name,
                                tau=tau,
                                seed=seed,
                                method=method,
                                critic_scope="own",
                                hop=hop,
                                total_hops=total_hops,
                                actions=actions,
                                q1s=q1s,
                                q2s=q2s,
                                grads=grads,
                                max_action=max_action,
                            )
                            hop_row["checkpoint_step"] = checkpoint_step
                            if eval_row.get(score_key):
                                hop_row["d4rl_score"] = float(
                                    eval_row[score_key]
                                )
                            hop_rows.append(hop_row)

    _write_csv(out_root / "checkpoint_dynamics.csv", checkpoint_rows)
    _write_csv(out_root / "hop_dynamics.csv", hop_rows)
    _write_csv(out_root / "run_stability.csv", stability_rows)
    manifest = {
        "methods": args.methods,
        "envs": args.envs,
        "taus": args.taus,
        "seeds": args.seeds,
        "requested_steps": args.steps or "all available",
        "n_states": int(args.n_states),
        "sample_seed": int(args.sample_seed),
        "device": "CPU",
        "td_residual": (
            "post-hoc deterministic target action without TD3 target noise"
        ),
        "prox_margin": (
            "post-hoc fixed validation subset; original train minibatch is "
            "not recoverable from saved checkpoints"
        ),
        "checkpoint_policy_drift": (
            "mean-per-action squared policy difference on the fixed validation "
            "states versus the preceding saved checkpoint"
        ),
        "missing_runs": missing,
        "checkpoint_rows": len(checkpoint_rows),
        "hop_rows": len(hop_rows),
        "stability_rows": len(stability_rows),
    }
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] checkpoints={len(checkpoint_rows)} "
        f"hops={len(hop_rows)} out={out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
