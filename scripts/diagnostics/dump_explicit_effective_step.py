#!/usr/bin/env python3
"""Post-hoc effective-step audit for the trained explicit MPI actors.

The audit reconstructs the Euler target from the final checkpoint on one
deterministic dataset subset per environment.  It records the raw target, the
componentwise clipped target used by training, the learned actor output, and
own/common-critic geometry.  This is a final-checkpoint diagnostic; it does
not reconstruct targets from earlier training iterations.
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
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train_td3bc import Actor, TwinCritic, load_checkpoint, load_transition  # noqa: E402


METHODS = {
    "exp2": ("results_expl2", "expl2", 2, "d4rl_pi2"),
    "exp3": ("results_expl3", "expl3", 3, "d4rl_pi3"),
    "exp2m": ("results_expl2_matched", "expl2", 2, "d4rl_pi2"),
    "exp3m": ("results_expl3_matched", "expl3", 3, "d4rl_pi3"),
}
DEFAULT_ENVS = (
    "hopper-medium-v2",
    "hopper-expert-v2",
    "walker2d-medium-v2",
    "walker2d-expert-v2",
    "halfcheetah-expert-v2",
)
EPS = 1e-8


def _tau_token(value: float) -> str:
    return f"{float(value):g}"


def _run_tag(method: str, env_name: str, tau: float, seed: int) -> str:
    _, tag, _, _ = METHODS[method]
    return f"{env_name}_tau{_tau_token(tau)}_{tag}_seed{seed}"


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
        f"{prefix}_max": float(np.max(values)),
    }


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


def _actor_path(
    payload: Mapping[str, Any],
    actor: Actor,
    states: np.ndarray,
    dataset_actions: np.ndarray,
    total_hops: int,
) -> list[np.ndarray]:
    keys = ["actor_params", "actor2_params", "actor3_params"][:total_hops]
    apply_actor = jax.jit(actor.apply)
    actions = [np.asarray(dataset_actions, dtype=np.float32)]
    for key in keys:
        actions.append(np.asarray(apply_actor(payload[key], states)))
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
    q1s: list[np.ndarray] = []
    q2s: list[np.ndarray] = []
    grads: list[np.ndarray] = []
    for action in actions:
        q1, q2 = q_jit(critic_params, states, action)
        q1s.append(np.asarray(q1))
        q2s.append(np.asarray(q2))
        grads.append(np.asarray(grad_jit(critic_params, states, action)))
    return q1s, q2s, grads


def _ratio_cosine(delta: np.ndarray, grad: np.ndarray) -> float:
    numerator = float(np.mean(np.sum(delta * grad, axis=-1)))
    denominator = np.sqrt(
        float(np.mean(np.sum(delta * delta, axis=-1)))
        * float(np.mean(np.sum(grad * grad, axis=-1)))
    )
    return numerator / (denominator + EPS)


def _geometry(
    previous: np.ndarray,
    current: np.ndarray,
    q_prev: np.ndarray,
    q_next: np.ndarray,
    grad_prev: np.ndarray,
    grad_next: np.ndarray,
) -> dict[str, float]:
    delta = current - previous
    delta_sq = np.sum(delta * delta, axis=-1)
    linear = np.sum(grad_prev * delta, axis=-1)
    nonlinear = q_next - q_prev - linear
    denominator = float(np.mean(delta_sq)) + EPS
    return {
        "q_gain_mean": float(np.mean(q_next - q_prev)),
        "linear_gain_mean": float(np.mean(linear)),
        "nonlinear_correction_mean": float(np.mean(nonlinear)),
        "step_sq_mean": float(np.mean(delta_sq)),
        "kappa_q_ratio_of_means": 2.0 * float(np.mean(nonlinear)) / denominator,
        "kappa_grad_ratio_of_means": float(
            np.mean(np.sum(delta * (grad_next - grad_prev), axis=-1))
        )
        / denominator,
        "grad_prev_norm_mean": float(np.mean(np.linalg.norm(grad_prev, axis=-1))),
        "grad_next_norm_mean": float(np.mean(np.linalg.norm(grad_next, axis=-1))),
    }


def _training_diagnostics(
    payload: Mapping[str, Any],
    critic: TwinCritic,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    not_dones: np.ndarray,
) -> dict[str, float]:
    actor = Actor(
        action_dim=actions.shape[-1],
        max_action=float(payload.get("max_action", 1.0)),
    )

    def q_apply(params, obs, act):
        q1, q2 = critic.apply(params, obs, act)
        return jnp.squeeze(q1, -1), jnp.squeeze(q2, -1)

    next_actions = np.asarray(
        jax.jit(actor.apply)(payload["target_actor_params"], next_states)
    )
    tq1, tq2 = jax.jit(q_apply)(
        payload["target_critic_params"], next_states, next_actions
    )
    target = rewards.reshape(-1) + 0.99 * not_dones.reshape(-1) * np.minimum(
        np.asarray(tq1), np.asarray(tq2)
    )
    q1, q2 = jax.jit(q_apply)(payload["critic_params"], states, actions)
    q1, q2 = np.asarray(q1), np.asarray(q2)
    residual = np.square(q1 - target) + np.square(q2 - target)
    result = _quantiles(residual, "td_residual")
    result.update(_quantiles(np.abs(q1), "q_data_abs"))
    result.update(_quantiles(np.abs(target), "td_target_abs"))
    result.update(_quantiles(np.abs(next_actions), "target_actor_abs"))
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--data-dir", default="/raid/ext_csv/datasets/d4rl")
    parser.add_argument(
        "--out-dir", default=str(_ROOT / "audit_report_20260828" / "explicit_geometry")
    )
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--taus", nargs="+", type=float, default=[4.0, 10.0, 20.0])
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

    for env_name in args.envs:
        data, _, _ = load_transition(env_name, Path(args.data_dir), normalize=True)
        n_available = int(data.observations.shape[0])
        rng = np.random.default_rng(_stable_seed(env_name, args.sample_seed))
        indices = rng.choice(
            n_available, size=min(int(args.n_states), n_available), replace=False
        )
        batch_dir = out_root / "common_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        np.save(batch_dir / f"{env_name}_indices.npy", indices.astype(np.int64))
        states = np.asarray(data.observations[indices], dtype=np.float32)
        dataset_actions = np.asarray(data.actions[indices], dtype=np.float32)
        next_states = np.asarray(data.next_observations[indices], dtype=np.float32)
        rewards = np.asarray(data.rewards[indices], dtype=np.float32)
        not_dones = np.asarray(data.not_dones[indices], dtype=np.float32)

        canonical_payloads: dict[int, Mapping[str, Any]] = {}
        for seed in args.seeds:
            canonical_path = (
                root
                / "results_qnorm"
                / f"{env_name}_tau{_tau_token(args.canonical_critic_tau)}_seed{seed}"
                / "params_1000000.pkl"
            )
            if not canonical_path.is_file():
                raise FileNotFoundError(canonical_path)
            canonical_payloads[seed] = load_checkpoint(canonical_path)

        for tau in args.taus:
            for seed in args.seeds:
                for method in args.methods:
                    result_dir, _, total_hops, score_key = METHODS[method]
                    tag = _run_tag(method, env_name, tau, seed)
                    run_dir = root / result_dir / tag
                    checkpoint = run_dir / "params_1000000.pkl"
                    if not checkpoint.is_file():
                        missing.append(str(checkpoint))
                        print(f"[missing] {checkpoint}", flush=True)
                        continue
                    print(f"[audit] {tag}", flush=True)
                    payload = load_checkpoint(checkpoint)
                    config_path = run_dir / "config.json"
                    if config_path.is_file():
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                    else:
                        config = dict(payload.get("config", {}))
                    scale_norm = bool(config.get("q_scale_norm", True))
                    q_term = bool(config.get("explicit_q_term", False))
                    max_action = float(payload.get("max_action", 1.0))
                    actor = Actor(
                        action_dim=int(dataset_actions.shape[-1]),
                        max_action=max_action,
                    )
                    actions = _actor_path(
                        payload, actor, states, dataset_actions, total_hops
                    )
                    own_q1, own_q2, own_grads = _critic_path(
                        critic, payload["critic_params"], states, actions
                    )
                    canonical_q1, canonical_q2, canonical_grads = _critic_path(
                        critic,
                        canonical_payloads[seed]["critic_params"],
                        states,
                        actions,
                    )
                    run_row: dict[str, Any] = {
                        "env": env_name,
                        "tau": float(tau),
                        "seed": int(seed),
                        "method": method,
                        "total_hops": int(total_hops),
                        "checkpoint": str(checkpoint),
                        "explicit_q_term": q_term,
                        "q_scale_norm": scale_norm,
                        "config_option_inferred": (
                            "explicit_q_term" not in config or "q_scale_norm" not in config
                        ),
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
                    run_rows.append(run_row)

                    raw_payload: dict[str, np.ndarray] = {
                        "indices": indices.astype(np.int64),
                        "states": states,
                        "dataset_actions": dataset_actions,
                    }
                    for hop, action in enumerate(actions):
                        raw_payload[f"action_hop{hop}"] = action

                    eta = float(tau) / float(total_hops)
                    for hop in range(1, len(actions)):
                        previous = actions[hop - 1]
                        current = actions[hop]
                        delta = current - previous
                        q_scale = float(np.mean(np.abs(own_q1[hop - 1]))) + 1e-6
                        normalized_grad = (
                            own_grads[hop - 1] / q_scale
                            if scale_norm
                            else own_grads[hop - 1]
                        )
                        action_dim = int(current.shape[-1])
                        target_coefficient = float(action_dim) * eta
                        target_raw = (
                            previous + target_coefficient * normalized_grad
                        )
                        target_clip = np.clip(target_raw, -max_action, max_action)
                        target_step = target_clip - previous
                        raw_target_step = target_raw - previous
                        delta_norm = np.linalg.norm(delta, axis=-1)
                        grad_norm = np.linalg.norm(normalized_grad, axis=-1)
                        raw_error = np.linalg.norm(
                            delta - target_coefficient * normalized_grad, axis=-1
                        )
                        clip_error = np.linalg.norm(current - target_clip, axis=-1)
                        numerator = float(
                            np.mean(np.sum(delta * normalized_grad, axis=-1))
                        )
                        grad_sq_mean = float(
                            np.mean(np.sum(normalized_grad * normalized_grad, axis=-1))
                        )
                        h_eff = numerator / (grad_sq_mean + EPS)
                        clip_numerator = float(
                            np.mean(np.sum(target_step * normalized_grad, axis=-1))
                        )
                        h_eff_clip_target = clip_numerator / (grad_sq_mean + EPS)
                        component_clip = np.abs(target_raw) > max_action
                        raw_target_sq_mean = float(
                            np.mean(
                                np.sum(raw_target_step * raw_target_step, axis=-1)
                            )
                        )
                        clip_target_sq_mean = float(
                            np.mean(np.sum(target_step * target_step, axis=-1))
                        )
                        actual_step_sq_mean = float(
                            np.mean(np.sum(delta * delta, axis=-1))
                        )
                        projection_attenuation = np.sqrt(
                            clip_target_sq_mean / (raw_target_sq_mean + EPS)
                        )
                        actor_fit_realization = np.sqrt(
                            actual_step_sq_mean / (clip_target_sq_mean + EPS)
                        )
                        target_fit_error = float(
                            np.mean(np.square(current - target_clip))
                        ) / (
                            float(np.mean(np.square(previous - target_clip)))
                            + EPS
                        )
                        directional_realization = float(
                            np.mean(np.sum(delta * target_step, axis=-1))
                        ) / (clip_target_sq_mean + EPS)
                        row: dict[str, Any] = {
                            "env": env_name,
                            "tau": float(tau),
                            "seed": int(seed),
                            "method": method,
                            "hop": int(hop),
                            "total_hops": int(total_hops),
                            "n_states": int(states.shape[0]),
                            "action_dim": int(current.shape[-1]),
                            "eta_tau_over_k": eta,
                            "q_scale_own_ref": q_scale,
                            "explicit_q_term": q_term,
                            "q_scale_norm": scale_norm,
                            "h_eff": h_eff,
                            "h_eff_over_eta": h_eff / (eta + EPS),
                            "target_coefficient_d_eta": target_coefficient,
                            "h_eff_over_target_coefficient": h_eff
                            / (target_coefficient + EPS),
                            "clipped_target_h_eff": h_eff_clip_target,
                            "clipped_target_h_eff_over_eta": h_eff_clip_target / (eta + EPS),
                            "clipped_target_h_eff_over_target_coefficient": h_eff_clip_target
                            / (target_coefficient + EPS),
                            "explicit_raw_fidelity": float(np.mean(raw_error))
                            / (
                                float(np.mean(delta_norm))
                                + target_coefficient * float(np.mean(grad_norm))
                                + EPS
                            ),
                            "projection_attenuation": float(
                                projection_attenuation
                            ),
                            "actor_fit_realization": float(actor_fit_realization),
                            "target_fit_error": target_fit_error,
                            "directional_realization": directional_realization,
                            "cos_step_grad_ratio": _ratio_cosine(delta, normalized_grad),
                            "raw_target_component_clip_fraction": float(np.mean(component_clip)),
                            "raw_target_state_clip_fraction": float(
                                np.mean(np.any(component_clip, axis=-1))
                            ),
                            "raw_to_clip_norm_mean": float(
                                np.mean(np.linalg.norm(target_raw - target_clip, axis=-1))
                            ),
                            "actor_to_clipped_target_norm_mean": float(np.mean(clip_error)),
                            "actor_saturation_component_fraction": float(
                                np.mean(np.abs(current) > 0.99 * max_action)
                            ),
                            "actor_saturation_state_fraction": float(
                                np.mean(np.any(np.abs(current) > 0.99 * max_action, axis=-1))
                            ),
                        }
                        row.update(_quantiles(delta_norm, "actual_step_norm"))
                        row.update(_quantiles(np.linalg.norm(target_step, axis=-1), "clip_target_step_norm"))
                        for prefix, q1s, grads in (
                            ("own", own_q1, own_grads),
                            ("canonical", canonical_q1, canonical_grads),
                        ):
                            for key, value in _geometry(
                                previous,
                                current,
                                q1s[hop - 1],
                                q1s[hop],
                                grads[hop - 1],
                                grads[hop],
                            ).items():
                                row[f"{prefix}_{key}"] = value
                        row.update(_read_final_eval(run_dir, score_key))
                        hop_rows.append(row)

                        raw_payload[f"hop{hop}_grad_q_own_prev"] = own_grads[hop - 1]
                        raw_payload[f"hop{hop}_grad_q_own_next"] = own_grads[hop]
                        raw_payload[f"hop{hop}_q1_own_prev"] = own_q1[hop - 1]
                        raw_payload[f"hop{hop}_q1_own_next"] = own_q1[hop]
                        raw_payload[f"hop{hop}_q2_own_prev"] = own_q2[hop - 1]
                        raw_payload[f"hop{hop}_q2_own_next"] = own_q2[hop]
                        raw_payload[f"hop{hop}_target_raw"] = target_raw
                        raw_payload[f"hop{hop}_target_clip"] = target_clip
                        raw_payload[f"hop{hop}_q1_canonical_prev"] = canonical_q1[hop - 1]
                        raw_payload[f"hop{hop}_q1_canonical_next"] = canonical_q1[hop]
                        raw_payload[f"hop{hop}_grad_q_canonical_prev"] = canonical_grads[hop - 1]
                        raw_payload[f"hop{hop}_grad_q_canonical_next"] = canonical_grads[hop]

                    raw_dir = out_root / "raw" / env_name
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(raw_dir / f"{tag}.npz", **raw_payload)

    _write_csv(out_root / "hop_effective_step.csv", hop_rows)
    _write_csv(out_root / "run_diagnostics.csv", run_rows)
    manifest = {
        "methods": args.methods,
        "envs": args.envs,
        "taus": args.taus,
        "seeds": args.seeds,
        "n_states": int(args.n_states),
        "sample_seed": int(args.sample_seed),
        "canonical_critic": (
            "TD3+BC checkpoint from same env/seed at "
            f"tau={args.canonical_critic_tau:g}"
        ),
        "diagnostic_time": "final checkpoint only",
        "target_reconstruction": (
            "matched d*(tau/K)*grad(Q)/q_scale target under own final critic; "
            "fixed validation batch, while training used minibatch q-scale"
        ),
        "legacy_config_inference": (
            "missing q_scale_norm -> True; missing explicit_q_term -> False"
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
