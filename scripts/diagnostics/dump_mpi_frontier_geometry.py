#!/usr/bin/env python3
"""Post-hoc geometry diagnostics for matched TD3+BC / MPI frontier cells.

The dump uses one deterministic transition subset per environment for every
method, tau, and seed.  Each actor path is evaluated both with its own jointly
trained critic and with a common, low-tau TD3+BC critic.  The latter is a
cross-method diagnostic only; it does not turn the original training run into
a frozen-critic control experiment.

The critic is ReLU, so the reported curvature values are finite path secants,
not pointwise Hessians.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train_td3bc import (  # noqa: E402
    Actor,
    TwinCritic,
    load_checkpoint,
    load_transition,
)


DEFAULT_ENVS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-v2",
    "walker2d-expert-v2",
)
DEFAULT_TAUS = (4.0, 7.0, 10.0)
METHODS = {
    "td3": ("results_qnorm", "", 1, "d4rl_score"),
    "mpi2": ("results_mpi2", "mpi2", 2, "d4rl_pi2"),
    "mpi3": ("results_mpi3", "mpi3", 3, "d4rl_pi3"),
    "expl2": ("results_expl2_matched", "expl2", 2, "d4rl_pi2"),
    "expl3": ("results_expl3_matched", "expl3", 3, "d4rl_pi3"),
}
EPS = 1e-8


def _tau_token(value: float) -> str:
    return f"{float(value):g}"


def _run_tag(method: str, env_name: str, tau: float, seed: int) -> str:
    _, method_tag, _, _ = METHODS[method]
    middle = f"_{method_tag}" if method_tag else ""
    return f"{env_name}_tau{_tau_token(tau)}{middle}_seed{seed}"


def _stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**63)


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def _cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + EPS
    return np.sum(x * y, axis=-1) / denom


def _read_final_eval(run_dir: Path, score_key: str) -> dict[str, float | int]:
    path = run_dir / "eval.csv"
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {}
    final = rows[-1]
    result: dict[str, float | int] = {
        "eval_step": int(final["step"]),
        "final_critic_loss": float(final["critic_loss"]),
        "max_critic_loss": max(float(row["critic_loss"]) for row in rows),
    }
    if final.get(score_key):
        result["d4rl_score"] = float(final[score_key])
    return result


def _tree_params(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise KeyError(f"checkpoint is missing {key!r}")
    return payload[key]


def _actor_path(
    payload: Mapping[str, Any],
    method: str,
    actor: Actor,
    states: np.ndarray,
    dataset_actions: np.ndarray,
) -> list[np.ndarray]:
    keys = ["actor_params"]
    if method in ("mpi2", "mpi3", "expl2", "expl3"):
        keys.append("actor2_params")
    if method in ("mpi3", "expl3"):
        keys.append("actor3_params")
    apply_actor = jax.jit(actor.apply)
    actions = [np.asarray(dataset_actions, dtype=np.float32)]
    for key in keys:
        actions.append(
            np.asarray(apply_actor(_tree_params(payload, key), states))
        )
    return actions


def _critic_path(
    critic: TwinCritic,
    critic_params: Any,
    states: np.ndarray,
    actions: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    def q_apply(params, obs, act):
        q1, q2 = critic.apply(params, obs, act)
        return jnp.squeeze(q1, -1), jnp.squeeze(q2, -1)

    def q1_grad(params, obs, act):
        def scalar(a):
            q1, _ = critic.apply(params, obs[None], a[None])
            return jnp.squeeze(q1)

        return jax.grad(scalar)(act)

    q_jit = jax.jit(q_apply)
    grad_jit = jax.jit(jax.vmap(q1_grad, in_axes=(None, 0, 0)))
    q1s, q2s, grads = [], [], []
    for action in actions:
        q1, q2 = q_jit(critic_params, states, action)
        q1s.append(np.asarray(q1))
        q2s.append(np.asarray(q2))
        grads.append(np.asarray(grad_jit(critic_params, states, action)))
    return q1s, q2s, grads


def _hop_row(
    *,
    env_name: str,
    tau: float,
    seed: int,
    method: str,
    critic_scope: str,
    hop: int,
    total_hops: int,
    actions: list[np.ndarray],
    q1s: list[np.ndarray],
    q2s: list[np.ndarray],
    grads: list[np.ndarray],
    max_action: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    previous = actions[hop - 1]
    current = actions[hop]
    delta = current - previous
    total_delta = current - actions[0]
    delta_sq = np.sum(delta * delta, axis=-1) + EPS
    transport_cost = np.sum(delta * delta, axis=-1) / float(
        current.shape[-1]
    )
    q1_gain = q1s[hop] - q1s[hop - 1]
    linear_gain = np.sum(grads[hop - 1] * delta, axis=-1)
    linearization_error = np.abs(q1_gain - linear_gain) / (
        np.abs(q1_gain) + np.abs(linear_gain) + EPS
    )
    gradient_change = np.linalg.norm(
        grads[hop] - grads[hop - 1], axis=-1
    ) / (
        np.linalg.norm(grads[hop], axis=-1)
        + np.linalg.norm(grads[hop - 1], axis=-1)
        + EPS
    )
    secant_grad = np.sum(
        delta * (grads[hop] - grads[hop - 1]), axis=-1
    ) / delta_sq
    secant_q = 2.0 * (
        q1s[hop] - q1s[hop - 1] - linear_gain
    ) / delta_sq
    tau_step = float(tau) / float(total_hops)
    # Hop 1 follows update_actor(), whose normalization uses Q(pi). Later
    # JKO hops normalize at the frozen reference action.
    q_scale_values = q1s[hop] if hop == 1 else q1s[hop - 1]
    q_scale = float(np.mean(np.abs(q_scale_values))) + 1e-6
    action_dim = int(current.shape[-1])
    effective_step = action_dim * tau_step / q_scale
    q_weight = 2.0 * tau_step / q_scale
    # This is a post-hoc validation-batch margin under the selected frozen
    # critic. It is not the original stochastic train-minibatch margin, which
    # cannot be reconstructed from checkpoints.
    proximal_margin = q_weight * q1_gain - transport_cost
    effective_secant_curvature = np.abs(effective_step * secant_q)
    scaled_grad = effective_step * grads[hop]
    prox_residual = np.linalg.norm(delta - scaled_grad, axis=-1) / (
        np.linalg.norm(delta, axis=-1)
        + np.linalg.norm(scaled_grad, axis=-1)
        + EPS
    )
    saturation = np.mean(
        np.abs(current) >= 0.99 * float(max_action), axis=-1
    )
    twin_gap = np.abs(q1s[hop] - q2s[hop])
    row: dict[str, Any] = {
        "env": env_name,
        "tau": float(tau),
        "seed": int(seed),
        "method": method,
        "critic_scope": critic_scope,
        "hop": int(hop),
        "total_hops": int(total_hops),
        "tau_step": tau_step,
        "n_states": int(current.shape[0]),
        "action_dim": action_dim,
        "q_scale": q_scale,
        "effective_action_step": effective_step,
        "q_weight": q_weight,
        "q1_gain_hop_mean": float(np.mean(q1_gain)),
        "q1_gain_total_mean": float(np.mean(q1s[hop] - q1s[0])),
        "q2_gain_hop_mean": float(np.mean(q2s[hop] - q2s[hop - 1])),
        "q2_gain_total_mean": float(np.mean(q2s[hop] - q2s[0])),
        "linear_gain_mean": float(np.mean(linear_gain)),
        "transport_cost_mean": float(np.mean(transport_cost)),
        "prox_margin_val_mean": float(np.mean(proximal_margin)),
        "prox_margin_val_nonnegative_fraction": float(
            np.mean(proximal_margin >= 0.0)
        ),
        "linearization_error_mean": float(np.mean(linearization_error)),
        "gradient_change_ratio_mean": float(np.mean(gradient_change)),
        "effective_secant_curvature_mean": float(
            np.mean(effective_secant_curvature)
        ),
        "secant_grad_mean": float(np.mean(secant_grad)),
        "secant_grad_p10": float(np.percentile(secant_grad, 10)),
        "secant_grad_p50": float(np.percentile(secant_grad, 50)),
        "secant_grad_p90": float(np.percentile(secant_grad, 90)),
        "secant_q_mean": float(np.mean(secant_q)),
        "secant_q_p10": float(np.percentile(secant_q, 10)),
        "secant_q_p50": float(np.percentile(secant_q, 50)),
        "secant_q_p90": float(np.percentile(secant_q, 90)),
        "secant_q_positive_fraction": float(np.mean(secant_q > 0.0)),
        "prox_residual_mean": float(np.mean(prox_residual)),
        "saturation_fraction_mean": float(np.mean(saturation)),
        "twin_gap_mean": float(np.mean(twin_gap)),
        "grad_norm_mean": float(np.mean(np.linalg.norm(grads[hop], axis=-1))),
        "cos_step_grad_mean": float(np.mean(_cosine(delta, grads[hop]))),
    }
    row.update(_quantiles(np.linalg.norm(delta, axis=-1), "step_norm"))
    row.update(_quantiles(np.linalg.norm(total_delta, axis=-1), "total_norm"))
    row.update(_quantiles(proximal_margin, "prox_margin_val"))
    row.update(_quantiles(linearization_error, "linearization_error"))
    row.update(_quantiles(gradient_change, "gradient_change_ratio"))
    row.update(
        _quantiles(
            effective_secant_curvature,
            "effective_secant_curvature",
        )
    )
    if hop > 1:
        previous_delta = actions[hop - 1] - actions[hop - 2]
        row["cos_consecutive_steps_mean"] = float(
            np.mean(_cosine(previous_delta, delta))
        )
    raw = {
        "delta": delta,
        "total_delta": total_delta,
        "q1": q1s[hop],
        "q2": q2s[hop],
        "grad_q1": grads[hop],
        "q1_gain": q1_gain,
        "linear_gain": linear_gain,
        "transport_cost": transport_cost,
        "prox_margin_val": proximal_margin,
        "linearization_error": linearization_error,
        "gradient_change_ratio": gradient_change,
        "effective_secant_curvature": effective_secant_curvature,
        "secant_grad": secant_grad,
        "secant_q": secant_q,
        "prox_residual": prox_residual,
        "saturation": saturation,
    }
    return row, raw


def _training_diagnostics(
    payload: Mapping[str, Any],
    critic: TwinCritic,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    not_dones: np.ndarray,
) -> dict[str, float]:
    critic_params = _tree_params(payload, "critic_params")
    target_critic_params = _tree_params(payload, "target_critic_params")
    target_actor_params = _tree_params(payload, "target_actor_params")
    actor = Actor(action_dim=actions.shape[-1], max_action=float(payload.get("max_action", 1.0)))

    def q_apply(params, obs, act):
        q1, q2 = critic.apply(params, obs, act)
        return jnp.squeeze(q1, -1), jnp.squeeze(q2, -1)

    next_actions = np.asarray(jax.jit(actor.apply)(target_actor_params, next_states))
    tq1, tq2 = jax.jit(q_apply)(target_critic_params, next_states, next_actions)
    target = np.asarray(rewards).reshape(-1) + 0.99 * np.asarray(not_dones).reshape(-1) * np.minimum(
        np.asarray(tq1), np.asarray(tq2)
    )
    q1, q2 = jax.jit(q_apply)(critic_params, states, actions)
    q1, q2 = np.asarray(q1), np.asarray(q2)
    residual = np.square(q1 - target) + np.square(q2 - target)
    result = {
        "q_data_abs_mean": float(np.mean(np.abs(q1))),
        "q_data_abs_p99": float(np.percentile(np.abs(q1), 99)),
    }
    result.update(_quantiles(residual, "td_residual"))
    result.update(_quantiles(np.abs(target), "td_target_abs"))
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=os.environ.get("RESULTS_ROOT", str(_ROOT))
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR", str(_ROOT / "data"))
    )
    parser.add_argument("--out-dir", default=str(_ROOT / "geometry_mpi_frontier"))
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260824)
    parser.add_argument("--canonical-critic-tau", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    critic = TwinCritic()
    hop_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    dataset_cache: dict[str, Any] = {}
    canonical_cache: dict[tuple[str, int], Mapping[str, Any]] = {}

    for env_name in args.envs:
        data, _, _ = load_transition(env_name, Path(args.data_dir), normalize=True)
        dataset_cache[env_name] = data
        n_available = int(data.observations.shape[0])
        rng = np.random.default_rng(_stable_seed(env_name, args.sample_seed))
        indices = rng.choice(
            n_available,
            size=min(int(args.n_states), n_available),
            replace=False,
        )
        env_out = out_root / "common_batches"
        env_out.mkdir(parents=True, exist_ok=True)
        np.save(env_out / f"{env_name}_indices.npy", indices.astype(np.int64))
        states = np.asarray(data.observations[indices], dtype=np.float32)
        dataset_actions = np.asarray(data.actions[indices], dtype=np.float32)
        next_states = np.asarray(data.next_observations[indices], dtype=np.float32)
        rewards = np.asarray(data.rewards[indices], dtype=np.float32)
        not_dones = np.asarray(data.not_dones[indices], dtype=np.float32)

        for seed in args.seeds:
            canonical_tag = _run_tag(
                "td3", env_name, args.canonical_critic_tau, seed
            )
            canonical_path = (
                root
                / METHODS["td3"][0]
                / canonical_tag
                / "params_1000000.pkl"
            )
            if not canonical_path.is_file():
                raise FileNotFoundError(canonical_path)
            canonical_cache[(env_name, seed)] = load_checkpoint(canonical_path)

        for tau in args.taus:
            for seed in args.seeds:
                canonical_payload = canonical_cache[(env_name, seed)]
                for method in args.methods:
                    result_dir_name, _, total_hops, score_key = METHODS[method]
                    tag = _run_tag(method, env_name, tau, seed)
                    run_dir = root / result_dir_name / tag
                    checkpoint = run_dir / "params_1000000.pkl"
                    if not checkpoint.is_file():
                        missing.append(str(checkpoint))
                        print(f"[missing] {checkpoint}", flush=True)
                        continue
                    print(f"[geometry] {tag}", flush=True)
                    payload = load_checkpoint(checkpoint)
                    max_action = float(payload.get("max_action", 1.0))
                    actor = Actor(
                        action_dim=int(dataset_actions.shape[-1]),
                        max_action=max_action,
                    )
                    actions = _actor_path(
                        payload, method, actor, states, dataset_actions
                    )
                    run_row: dict[str, Any] = {
                        "env": env_name,
                        "tau": float(tau),
                        "seed": int(seed),
                        "method": method,
                        "total_hops": int(total_hops),
                        "checkpoint": str(checkpoint),
                    }
                    run_row.update(_read_final_eval(run_dir, score_key))
                    run_row.update(
                        _training_diagnostics(
                            payload,
                            critic,
                            states,
                            dataset_actions,
                            next_states,
                            rewards,
                            not_dones,
                        )
                    )
                    step_norms = [
                        np.linalg.norm(actions[i] - actions[i - 1], axis=-1)
                        for i in range(1, len(actions))
                    ]
                    path_length = np.sum(np.stack(step_norms, axis=1), axis=1)
                    chord = np.linalg.norm(actions[-1] - actions[0], axis=-1)
                    run_row.update(_quantiles(path_length, "path_length"))
                    run_row.update(_quantiles(chord, "chord_length"))
                    action_dim = int(dataset_actions.shape[-1])
                    critic_displacement = np.sum(
                        np.square(actions[1] - actions[0]), axis=-1
                    ) / float(action_dim)
                    final_displacement = np.sum(
                        np.square(actions[-1] - actions[0]), axis=-1
                    ) / float(action_dim)
                    run_row.update(
                        _quantiles(
                            critic_displacement,
                            "critic_displacement_sq_mean_metric",
                        )
                    )
                    run_row.update(
                        _quantiles(
                            final_displacement,
                            "final_displacement_sq_mean_metric",
                        )
                    )
                    run_row["decoupling_displacement_ratio"] = float(
                        np.sqrt(
                            np.mean(final_displacement)
                            / (np.mean(critic_displacement) + EPS)
                        )
                    )
                    run_row["path_tortuosity_mean"] = float(
                        np.mean(path_length / (chord + EPS))
                    )
                    run_rows.append(run_row)

                    own_params = _tree_params(payload, "critic_params")
                    canonical_params = _tree_params(
                        canonical_payload, "critic_params"
                    )
                    raw_payload: dict[str, np.ndarray] = {
                        "indices": indices.astype(np.int64),
                        "states": states,
                        "dataset_actions": dataset_actions,
                    }
                    for hop, action in enumerate(actions):
                        raw_payload[f"action_hop{hop}"] = action
                    for critic_scope, critic_params in (
                        ("own", own_params),
                        ("canonical_td3_tau1", canonical_params),
                    ):
                        q1s, q2s, grads = _critic_path(
                            critic, critic_params, states, actions
                        )
                        for hop in range(1, len(actions)):
                            row, raw = _hop_row(
                                env_name=env_name,
                                tau=tau,
                                seed=seed,
                                method=method,
                                critic_scope=critic_scope,
                                hop=hop,
                                total_hops=total_hops,
                                actions=actions,
                                q1s=q1s,
                                q2s=q2s,
                                grads=grads,
                                max_action=max_action,
                            )
                            row.update(_read_final_eval(run_dir, score_key))
                            hop_rows.append(row)
                            for key, value in raw.items():
                                raw_payload[
                                    f"{critic_scope}_hop{hop}_{key}"
                                ] = value
                    raw_dir = out_root / "raw" / env_name
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(raw_dir / f"{tag}.npz", **raw_payload)

    _write_csv(out_root / "hop_geometry.csv", hop_rows)
    _write_csv(out_root / "run_diagnostics.csv", run_rows)
    manifest = {
        "methods": args.methods,
        "envs": args.envs,
        "taus": args.taus,
        "seeds": args.seeds,
        "n_states": int(args.n_states),
        "sample_seed": int(args.sample_seed),
        "canonical_critic": (
            "TD3+BC checkpoint from the same env/seed at "
            f"tau={args.canonical_critic_tau:g}"
        ),
        "critic_activation": "relu",
        "curvature": "finite path secants, not pointwise Hessians",
        "reference_action": "matched dataset transition action",
        "prox_margin_val": (
            "post-hoc fixed-batch margin under each selected critic; not the "
            "unrecoverable original train-minibatch margin"
        ),
        "effective_action_step": (
            "d * (tau / K) / q_scale for raw-Q gradients"
        ),
        "missing_checkpoints": missing,
        "hop_rows": len(hop_rows),
        "run_rows": len(run_rows),
    }
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] runs={len(run_rows)} hop_rows={len(hop_rows)} "
        f"missing={len(missing)} out={out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
