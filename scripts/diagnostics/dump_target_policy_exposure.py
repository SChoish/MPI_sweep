#!/usr/bin/env python3
"""CPU-only audit of the target policy's realized next-state displacement.

The training loader drops timeout transitions, retains true terminal
transitions with ``not_done=0``, and pairs retained observations with the next
row's observation.  This audit follows that convention and then restricts its
sample to the nonterminal transitions that actually bootstrap.  Keeping the
raw HDF5 row index makes the additional pairing with the next dataset action
``actions[i + 1]`` auditable.

For a common batch of transitions and common seeded target-smoothing noises,
each checkpoint records four mean-per-action squared distances:

* deterministic Polyak target actor at s' to the paired next dataset action;
* the noise expectation of the clipped TD3 target action's distance to that
  paired action;
* Polyak actor lag between target_mu_1(s') and online mu_1(s'); and
* the existing current-state online-mu_1-to-sample-action proxy.

This is a post-hoc diagnostic.  It neither changes checkpoints nor launches
training, and all JAX work is forced onto CPU before JAX is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# These assignments deliberately override inherited accelerator settings: a
# script advertised as CPU-only must remain CPU-only inside GPU job shells.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4",
)

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from train_td3bc import (  # noqa: E402
    Actor,
    DATASET_FILES,
    dataset_path,
    download_dataset,
    load_checkpoint,
)


METHODS = {
    "td3": ("results_qnorm", ""),
    "mpi2": ("results_mpi2", "mpi2"),
    "mpi3": ("results_mpi3", "mpi3"),
    "expl2": ("results_expl2_matched", "expl2"),
    "expl3": ("results_expl3_matched", "expl3"),
}
DEFAULT_ENVS = tuple(DATASET_FILES)
DEFAULT_TAUS = (4.0, 7.0, 10.0, 14.0, 20.0)
METRICS = (
    "target_deterministic_to_next_data_sq_mean_metric",
    "target_smoothed_to_next_data_sq_mean_metric",
    "polyak_lag_sq_mean_metric",
    "current_sample_anchor_proxy_sq_mean_metric",
)


@dataclass(frozen=True)
class BootstrapTransitions:
    """Transitions that survive loader filtering and have ``not_done=1``."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    next_actions: np.ndarray
    rewards: np.ndarray
    source_indices: np.ndarray
    stats: dict[str, int]

    def take(self, indices: np.ndarray) -> "BootstrapTransitions":
        indices = np.asarray(indices, dtype=np.int64)
        return BootstrapTransitions(
            observations=self.observations[indices],
            actions=self.actions[indices],
            next_observations=self.next_observations[indices],
            next_actions=self.next_actions[indices],
            rewards=self.rewards[indices],
            source_indices=self.source_indices[indices],
            stats=dict(self.stats),
        )


def _tau_token(value: float) -> str:
    return f"{float(value):g}"


def _run_tag(method: str, env_name: str, tau: float, seed: int) -> str:
    _, method_tag = METHODS[method]
    middle = f"_{method_tag}" if method_tag else ""
    return f"{env_name}_tau{_tau_token(tau)}{middle}_seed{seed}"


def _stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**63)


def _validate_dataset_arrays(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    terminals = np.asarray(terminals, dtype=np.float32).reshape(-1)
    timeouts_array = (
        None
        if timeouts is None
        else np.asarray(timeouts, dtype=np.float32).reshape(-1)
    )
    n = observations.shape[0]
    if n < 2:
        raise ValueError("dataset must contain at least two rows")
    if actions.shape[0] != n or rewards.shape[0] != n or terminals.shape[0] != n:
        raise ValueError("dataset arrays have inconsistent leading dimensions")
    if timeouts_array is not None and timeouts_array.shape[0] != n:
        raise ValueError("timeouts have an inconsistent leading dimension")
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be rank-two arrays")
    return observations, actions, rewards, terminals, timeouts_array


def align_bootstrap_transitions(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray | None,
    *,
    max_episode_steps: int = 1000,
) -> BootstrapTransitions:
    """Reproduce ``qlearning_from_hdf5`` and retain bootstrap-active rows.

    A timeout row is omitted exactly as in the training loader.  A true
    terminal row is part of that loader's transition table but is excluded
    here because its TD target has ``not_done=0``.  On every remaining row
    ``i``, ``observations[i + 1]`` and ``actions[i + 1]`` are the paired next
    observation and dataset action.
    """

    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    observations, actions, rewards, terminals, timeouts = _validate_dataset_arrays(
        observations, actions, rewards, terminals, timeouts
    )
    n = observations.shape[0]
    terminal_mask = terminals[: n - 1].astype(bool)
    if timeouts is not None:
        timeout_mask = timeouts[: n - 1].astype(bool)
    else:
        timeout_mask = np.zeros(n - 1, dtype=bool)
        episode_step = 0
        for i in range(n - 1):
            is_timeout = episode_step == max_episode_steps - 1
            timeout_mask[i] = is_timeout
            if is_timeout:
                episode_step = 0
            elif terminal_mask[i]:
                episode_step = 0
            else:
                episode_step += 1

    loader_mask = ~timeout_mask
    bootstrap_mask = loader_mask & ~terminal_mask
    source_indices = np.flatnonzero(bootstrap_mask).astype(np.int64)
    if source_indices.size == 0:
        raise ValueError("dataset has no bootstrap-relevant nonterminal transitions")
    stats = {
        "hdf5_rows": int(n),
        "candidate_row_pairs": int(n - 1),
        "loader_transition_count": int(np.sum(loader_mask)),
        "bootstrap_transition_count": int(source_indices.size),
        "timeout_rows_excluded_by_loader": int(np.sum(timeout_mask)),
        "terminal_rows_excluded_from_bootstrap": int(
            np.sum(loader_mask & terminal_mask)
        ),
    }
    return BootstrapTransitions(
        observations=np.asarray(observations[source_indices], dtype=np.float32),
        actions=np.asarray(actions[source_indices], dtype=np.float32),
        next_observations=np.asarray(
            observations[source_indices + 1], dtype=np.float32
        ),
        next_actions=np.asarray(actions[source_indices + 1], dtype=np.float32),
        rewards=np.asarray(rewards[source_indices], dtype=np.float32),
        source_indices=source_indices,
        stats=stats,
    )


def load_bootstrap_transitions(
    path: Path, *, max_episode_steps: int = 1000
) -> BootstrapTransitions:
    import h5py

    with h5py.File(path, "r") as handle:
        observations = np.asarray(handle["observations"], dtype=np.float32)
        actions = np.asarray(handle["actions"], dtype=np.float32)
        rewards = np.asarray(handle["rewards"], dtype=np.float32)
        terminals = np.asarray(handle["terminals"], dtype=np.float32)
        timeouts = (
            np.asarray(handle["timeouts"], dtype=np.float32)
            if "timeouts" in handle
            else None
        )
    return align_bootstrap_transitions(
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        max_episode_steps=max_episode_steps,
    )


def _mean_action_squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"action shapes do not match: {x.shape} versus {y.shape}")
    if x.ndim < 2:
        raise ValueError("action arrays must include sample and action dimensions")
    return np.mean(np.square(x - y), axis=-1, dtype=np.float64)


def compute_exposure_metrics(
    *,
    target_next_actions: np.ndarray,
    online_next_actions: np.ndarray,
    online_current_actions: np.ndarray,
    next_dataset_actions: np.ndarray,
    current_dataset_actions: np.ndarray,
    standard_noises: np.ndarray,
    policy_noise: float,
    noise_clip: float,
    max_action: float,
) -> dict[str, np.ndarray]:
    """Compute per-state exposure metrics with common standard-normal noises."""

    target_next_actions = np.asarray(target_next_actions, dtype=np.float32)
    online_next_actions = np.asarray(online_next_actions, dtype=np.float32)
    online_current_actions = np.asarray(online_current_actions, dtype=np.float32)
    next_dataset_actions = np.asarray(next_dataset_actions, dtype=np.float32)
    current_dataset_actions = np.asarray(current_dataset_actions, dtype=np.float32)
    standard_noises = np.asarray(standard_noises, dtype=np.float32)
    expected_shape = target_next_actions.shape
    for name, value in (
        ("online_next_actions", online_next_actions),
        ("online_current_actions", online_current_actions),
        ("next_dataset_actions", next_dataset_actions),
        ("current_dataset_actions", current_dataset_actions),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {value.shape}; expected {expected_shape}"
            )
    if standard_noises.ndim != 3 or standard_noises.shape[1:] != expected_shape:
        raise ValueError(
            "standard_noises must have shape "
            f"(smoothing_samples, {expected_shape[0]}, {expected_shape[1]})"
        )
    if standard_noises.shape[0] < 1:
        raise ValueError("at least one smoothing sample is required")
    if policy_noise < 0.0 or noise_clip < 0.0 or max_action <= 0.0:
        raise ValueError("noise scales must be nonnegative and max_action positive")

    clipped_noises = np.clip(
        standard_noises * float(policy_noise),
        -float(noise_clip),
        float(noise_clip),
    )
    smoothed_actions = np.clip(
        target_next_actions[None, :, :] + clipped_noises,
        -float(max_action),
        float(max_action),
    ).astype(np.float32)
    smoothing_distances = _mean_action_squared_distance(
        smoothed_actions,
        np.broadcast_to(next_dataset_actions, smoothed_actions.shape),
    )
    deterministic_distance = _mean_action_squared_distance(
        target_next_actions, next_dataset_actions
    )
    polyak_lag = _mean_action_squared_distance(
        target_next_actions, online_next_actions
    )
    current_proxy = _mean_action_squared_distance(
        online_current_actions, current_dataset_actions
    )
    return {
        "target_deterministic_to_next_data_sq_mean_metric": deterministic_distance,
        "target_smoothed_to_next_data_sq_mean_metric": np.mean(
            smoothing_distances, axis=0, dtype=np.float64
        ),
        "polyak_lag_sq_mean_metric": polyak_lag,
        "current_sample_anchor_proxy_sq_mean_metric": current_proxy,
        "smoothing_repeat_sq_mean_metric": smoothing_distances,
        "smoothed_target_action_mean": np.mean(
            smoothed_actions, axis=0, dtype=np.float64
        ).astype(np.float32),
        "smoothed_action_saturation_fraction": np.mean(
            np.abs(smoothed_actions) >= float(max_action), axis=(0, 2)
        ),
    }


def _metric_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values))),
    }


def audit_checkpoint_cell(
    *,
    payload: Mapping[str, Any],
    transitions: BootstrapTransitions,
    standard_noises: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Evaluate one checkpoint cell without mutating checkpoint or dataset."""

    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (transitions.observations.shape[-1],) or std.shape != mean.shape:
        raise ValueError("checkpoint normalization shape does not match observations")
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("checkpoint normalization std must be finite and positive")
    # These fields define the action used by update_critic(). Requiring them
    # avoids silently auditing a reconstructed convention instead of the one
    # stored by the run.
    max_action = float(payload["max_action"])
    policy_noise = float(payload["policy_noise"])
    noise_clip = float(payload["noise_clip"])
    actor = Actor(
        action_dim=int(transitions.actions.shape[-1]), max_action=max_action
    )
    actor_apply = jax.jit(actor.apply)
    current_states = (
        np.asarray(transitions.observations, dtype=np.float32) - mean
    ) / std
    next_states = (
        np.asarray(transitions.next_observations, dtype=np.float32) - mean
    ) / std
    online_current_actions = np.asarray(
        actor_apply(payload["actor_params"], current_states), dtype=np.float32
    )
    online_next_actions = np.asarray(
        actor_apply(payload["actor_params"], next_states), dtype=np.float32
    )
    target_next_actions = np.asarray(
        actor_apply(payload["target_actor_params"], next_states), dtype=np.float32
    )
    metrics = compute_exposure_metrics(
        target_next_actions=target_next_actions,
        online_next_actions=online_next_actions,
        online_current_actions=online_current_actions,
        next_dataset_actions=transitions.next_actions,
        current_dataset_actions=transitions.actions,
        standard_noises=standard_noises,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        max_action=max_action,
    )
    raw = {
        "source_indices": transitions.source_indices.astype(np.int64),
        "observations": np.asarray(transitions.observations, dtype=np.float32),
        "dataset_actions": np.asarray(transitions.actions, dtype=np.float32),
        "next_observations": np.asarray(
            transitions.next_observations, dtype=np.float32
        ),
        "next_dataset_actions": np.asarray(
            transitions.next_actions, dtype=np.float32
        ),
        "rewards": np.asarray(transitions.rewards, dtype=np.float32),
        "not_dones": np.ones(
            (transitions.rewards.shape[0],), dtype=np.float32
        ),
        "online_current_actions": online_current_actions,
        "online_next_actions": online_next_actions,
        "target_next_actions": target_next_actions,
        **metrics,
    }
    summary: dict[str, float | int] = {
        "n_states": int(transitions.observations.shape[0]),
        "action_dim": int(transitions.actions.shape[-1]),
        "max_action": max_action,
        "policy_noise": policy_noise,
        "noise_clip": noise_clip,
        "smoothing_samples": int(standard_noises.shape[0]),
        "target_deterministic_saturation_fraction": float(
            np.mean(np.abs(target_next_actions) >= max_action)
        ),
        "smoothed_action_saturation_fraction_mean": float(
            np.mean(metrics["smoothed_action_saturation_fraction"])
        ),
    }
    for metric in METRICS:
        summary.update(_metric_summary(metrics[metric], metric))
    return raw, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=os.environ.get("RESULTS_ROOT", str(_ROOT))
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR", str(_ROOT / "data"))
    )
    parser.add_argument(
        "--out-dir", default=str(_ROOT / "target_policy_exposure")
    )
    parser.add_argument(
        "--envs", nargs="+", choices=DEFAULT_ENVS, default=list(DEFAULT_ENVS)
    )
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS)
    )
    parser.add_argument("--checkpoint-step", type=int, default=1_000_000)
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--smoothing-samples", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260829)
    parser.add_argument("--noise-seed", type=int, default=20260830)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail instead of recording and skipping missing checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_states < 1 or args.smoothing_samples < 1:
        raise ValueError("n-states and smoothing-samples must be positive")
    if args.checkpoint_step < 0:
        raise ValueError("checkpoint-step must be nonnegative")

    root = Path(args.root).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    common_root = out_root / "common_batches"
    raw_root = out_root / "raw"
    common_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    common_batches: dict[str, Any] = {}
    normalization_fingerprints: dict[str, dict[str, str]] = {}

    for env_name in args.envs:
        path = dataset_path(env_name, data_dir)
        if not path.is_file():
            path = download_dataset(env_name, data_dir)
        transitions = load_bootstrap_transitions(
            path, max_episode_steps=args.max_episode_steps
        )
        n_available = int(transitions.observations.shape[0])
        sample_rng = np.random.default_rng(
            _stable_seed(env_name, args.sample_seed)
        )
        candidate_indices = sample_rng.choice(
            n_available,
            size=min(int(args.n_states), n_available),
            replace=False,
        ).astype(np.int64)
        selected = transitions.take(candidate_indices)
        action_dim = int(selected.actions.shape[-1])
        noise_rng = np.random.default_rng(_stable_seed(env_name, args.noise_seed))
        standard_noises = noise_rng.standard_normal(
            (int(args.smoothing_samples), selected.actions.shape[0], action_dim),
            dtype=np.float32,
        )
        candidate_path = common_root / f"{env_name}_indices.npy"
        source_path = common_root / f"{env_name}_hdf5_indices.npy"
        noise_path = common_root / f"{env_name}_standard_noises.npy"
        np.save(candidate_path, candidate_indices)
        np.save(source_path, selected.source_indices.astype(np.int64))
        np.save(noise_path, standard_noises)
        common_batches[env_name] = {
            "dataset": str(path),
            "candidate_indices": str(candidate_path),
            "hdf5_source_indices": str(source_path),
            "standard_noises": str(noise_path),
            "sample_seed": int(_stable_seed(env_name, args.sample_seed)),
            "noise_seed": int(_stable_seed(env_name, args.noise_seed)),
            "selected_states": int(selected.observations.shape[0]),
            "action_dim": action_dim,
            **transitions.stats,
        }
        normalization_fingerprints[env_name] = {}

        for tau in args.taus:
            for seed in args.seeds:
                for method in args.methods:
                    result_dir, _ = METHODS[method]
                    tag = _run_tag(method, env_name, tau, seed)
                    checkpoint = (
                        root
                        / result_dir
                        / tag
                        / f"params_{int(args.checkpoint_step)}.pkl"
                    )
                    if not checkpoint.is_file():
                        missing.append(str(checkpoint))
                        print(f"[missing] {checkpoint}", flush=True)
                        if args.strict_missing:
                            raise FileNotFoundError(checkpoint)
                        continue
                    print(f"[target-exposure] {tag}", flush=True)
                    payload = load_checkpoint(checkpoint)
                    raw, cell_summary = audit_checkpoint_cell(
                        payload=payload,
                        transitions=selected,
                        standard_noises=standard_noises,
                    )
                    fingerprint = hashlib.sha256(
                        np.asarray(payload["mean"], dtype=np.float32).tobytes()
                        + np.asarray(payload["std"], dtype=np.float32).tobytes()
                    ).hexdigest()
                    normalization_fingerprints[env_name][tag] = fingerprint
                    raw_dir = raw_root / env_name
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = raw_dir / f"{tag}_step{args.checkpoint_step}.npz"
                    np.savez_compressed(
                        raw_path,
                        **raw,
                        checkpoint_mean=np.asarray(
                            payload["mean"], dtype=np.float32
                        ),
                        checkpoint_std=np.asarray(
                            payload["std"], dtype=np.float32
                        ),
                    )
                    row: dict[str, Any] = {
                        "env": env_name,
                        "tau": float(tau),
                        "seed": int(seed),
                        "method": method,
                        "checkpoint_step": int(args.checkpoint_step),
                        "checkpoint": str(checkpoint),
                        "raw_npz": str(raw_path),
                        "normalization_sha256": fingerprint,
                    }
                    row.update(cell_summary)
                    rows.append(row)

    summary_path = out_root / "target_policy_exposure_summary.csv"
    _write_csv(summary_path, rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "actual_next_state_target_policy_displacement",
        "post_hoc_only": True,
        "cpu_only": True,
        "cpu_environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "OMP_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "root": str(root),
        "data_dir": str(data_dir),
        "out_dir": str(out_root),
        "methods": list(args.methods),
        "envs": list(args.envs),
        "taus": [float(value) for value in args.taus],
        "seeds": [int(value) for value in args.seeds],
        "checkpoint_step": int(args.checkpoint_step),
        "requested_n_states": int(args.n_states),
        "smoothing_samples": int(args.smoothing_samples),
        "sample_seed_base": int(args.sample_seed),
        "noise_seed_base": int(args.noise_seed),
        "max_episode_steps": int(args.max_episode_steps),
        "transition_convention": {
            "timeout": "excluded exactly as train_td3bc.qlearning_from_hdf5",
            "terminal": "retained by training loader, excluded from this bootstrap-only audit",
            "next_observation": "observations[hdf5_index + 1]",
            "next_dataset_action": "actions[hdf5_index + 1]",
        },
        "metric_units": "mean squared displacement per action coordinate",
        "interpretation_limits": {
            "paired_next_action": (
                "a single observed action at the paired next state, not the "
                "conditional-mean behavior action; absolute distances include "
                "conditional action variance"
            ),
            "causality": (
                "post-hoc associative diagnostic; it does not identify target "
                "policy displacement as the cause of critic or return changes"
            ),
        },
        "smoothing_estimand": (
            "Monte Carlo mean over fixed common standard-normal noises of "
            "distance after stored policy_noise scaling, stored noise_clip, "
            "and stored max_action clipping"
        ),
        "metrics": list(METRICS),
        "summary_csv": str(summary_path),
        "completed_cells": len(rows),
        "missing_checkpoints": missing,
        "common_batches": common_batches,
        "normalization_fingerprints": normalization_fingerprints,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    manifest_path = out_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[done] cells={len(rows)} missing={len(missing)} "
        f"summary={summary_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
