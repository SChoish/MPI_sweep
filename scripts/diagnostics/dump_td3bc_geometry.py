#!/usr/bin/env python3
"""Dump critic-geometry logs for finished TD3+BC tau-sweep checkpoints.

Uses leftover GPU memory (intended CUDA_VISIBLE_DEVICES=1). Does not touch
PathBridger trainers. Critic is ReLU, so this stores finite-scale secant
quantities (g, Q, line scans), not pointwise Hessians.

Each (env, tau, seed) has its own jointly trained (Q_α, π_α).
a_0 is the dataset action a_D (SPI-from-π(D) interpretation; no separate BC actor).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

from train_td3bc import Actor, TwinCritic, load_checkpoint, load_transition

import jax
import jax.numpy as jnp
import numpy as np

TAUS = [
    0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7,
    1, 1.5, 2, 2.5, 3, 4, 5, 7, 8, 10,
]
ENVS = [
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-expert-v2",
]
LINE_T = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float32)
EPS = 1e-8


def _tag(env: str, tau: float, seed: int) -> str:
    return f"{env}_tau{tau:g}_seed{seed}"


def _d4rl_final(run_dir: Path) -> float | None:
    path = run_dir / "eval.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows or rows[-1].get("step") != "1000000":
        return None
    return float(rows[-1]["d4rl_score"])


def _quantiles(x: np.ndarray, qs=(10, 50, 90, 99)) -> dict:
    out = {"mean": float(np.mean(x))}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(x, q))
    return out


def dump_cell(
    env: str,
    tau: float,
    seed: int,
    results_root: Path,
    data_dir: Path,
    out_root: Path,
    n_states: int,
    data_cache: dict,
) -> dict | None:
    tag = _tag(env, tau, seed)
    run_dir = results_root / tag
    ckpt = run_dir / "params_1000000.pkl"
    if not ckpt.is_file():
        return None
    out_dir = out_root / tag
    raw_path = out_dir / "geometry_raw.npz"
    sum_path = out_dir / "geometry_summary.json"
    if raw_path.is_file() and sum_path.is_file():
        return json.loads(sum_path.read_text())

    if env not in data_cache:
        data, mean, std = load_transition(env, data_dir, normalize=True)
        data_cache[env] = (data, mean, std)
    data, mean, std = data_cache[env]
    n = int(data.observations.shape[0])
    rng = np.random.default_rng(seed + 4096)
    idx = rng.choice(n, size=min(n_states, n), replace=False)
    states = np.asarray(data.observations[idx], dtype=np.float32)
    a_d = np.asarray(data.actions[idx], dtype=np.float32)

    payload = load_checkpoint(ckpt)
    actor_params = payload["actor_params"]
    critic_params = payload["critic_params"]
    max_action = float(payload.get("max_action", 1.0))
    da = int(a_d.shape[-1])
    actor = Actor(action_dim=da, max_action=max_action)
    critic = TwinCritic()

    def apply_actor(params, obs):
        return actor.apply(params, obs)

    def apply_q(params, obs, act):
        q1, q2 = critic.apply(params, obs, act)
        return jnp.squeeze(q1, -1), jnp.squeeze(q2, -1)

    def grad_q1(params, obs, act):
        def f(a):
            q1, _ = critic.apply(params, obs[None], a[None])
            return jnp.squeeze(q1)

        return jax.grad(f)(act)

    def grad_q2(params, obs, act):
        def f(a):
            _, q2 = critic.apply(params, obs[None], a[None])
            return jnp.squeeze(q2)

        return jax.grad(f)(act)

    a_alpha = np.asarray(jax.jit(apply_actor)(actor_params, states))
    a0 = a_d  # π(D) sample; no separate BC actor in this sweep
    q1_0, q2_0 = jax.jit(apply_q)(critic_params, states, a0)
    q1_a, q2_a = jax.jit(apply_q)(critic_params, states, a_alpha)
    q1_0 = np.asarray(q1_0)
    q2_0 = np.asarray(q2_0)
    q1_a = np.asarray(q1_a)
    q2_a = np.asarray(q2_a)
    g0_1 = np.asarray(jax.jit(jax.vmap(grad_q1, in_axes=(None, 0, 0)))(critic_params, states, a0))
    ga_1 = np.asarray(jax.jit(jax.vmap(grad_q1, in_axes=(None, 0, 0)))(critic_params, states, a_alpha))
    g0_2 = np.asarray(jax.jit(jax.vmap(grad_q2, in_axes=(None, 0, 0)))(critic_params, states, a0))
    ga_2 = np.asarray(jax.jit(jax.vmap(grad_q2, in_axes=(None, 0, 0)))(critic_params, states, a_alpha))

    delta = a_alpha - a0
    r = np.linalg.norm(delta, axis=-1)
    r2 = np.sum(delta * delta, axis=-1) + EPS
    mean_abs_q = float(np.mean(np.abs(q1_a)))
    alpha = 2.0 * float(tau)
    lam = alpha / (mean_abs_q + 1e-6)
    tau_eff_mean = 0.5 * lam
    tau_eff_sum = 0.5 * lam * da

    kappa_grad = np.sum(delta * (ga_1 - g0_1), axis=-1) / r2
    lin = np.sum(g0_1 * delta, axis=-1)
    kappa_q = 2.0 * (q1_a - q1_0 - lin) / r2
    prox_num = np.linalg.norm(delta - tau_eff_sum * ga_1, axis=-1)
    prox_den = r + tau_eff_sum * np.linalg.norm(ga_1, axis=-1) + EPS
    r_prox = prox_num / prox_den
    sat = np.mean(np.abs(a_alpha) > 0.99 * max_action, axis=-1)

    q1_line = []
    q2_line = []
    apply_q_jit = jax.jit(apply_q)
    for t in LINE_T:
        at = a0 + float(t) * delta
        q1t, q2t = apply_q_jit(critic_params, states, at)
        q1_line.append(np.asarray(q1t))
        q2_line.append(np.asarray(q2t))
    q1_line = np.stack(q1_line, axis=1)
    q2_line = np.stack(q2_line, axis=1)

    d4rl = _d4rl_final(run_dir)
    g0n = np.linalg.norm(g0_1, axis=-1) * np.linalg.norm(g0_2, axis=-1) + EPS
    cos_g = np.sum(g0_1 * g0_2, axis=-1) / g0n

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        states=states,
        a_d=a_d,
        a_0=a0,
        a_alpha=a_alpha,
        q1_a0=q1_0,
        q1_aalpha=q1_a,
        q2_a0=q2_0,
        q2_aalpha=q2_a,
        g0_q1=g0_1,
        galpha_q1=ga_1,
        g0_q2=g0_2,
        galpha_q2=ga_2,
        delta=delta,
        r_alpha=r,
        kappa_grad=kappa_grad,
        kappa_q=kappa_q,
        r_prox=r_prox,
        sat_frac=sat,
        line_t=LINE_T,
        q1_line=q1_line,
        q2_line=q2_line,
        idx=idx.astype(np.int64),
    )

    rq = _quantiles(r)
    summary = {
        "env": env,
        "tau": float(tau),
        "alpha": alpha,
        "seed": int(seed),
        "n": int(states.shape[0]),
        "d_a": da,
        "max_action": max_action,
        "critic_activation": "relu",
        "same_frozen_critic": False,
        "bc_reduction": "mean_over_batch_and_action_dims",
        "a0_source": "dataset_action",
        "mean_abs_q1_pi": mean_abs_q,
        "lambda_alpha": lam,
        "tau_eff_if_mean_sq": tau_eff_mean,
        "tau_eff_if_sum_sq": tau_eff_sum,
        "d4rl_score": d4rl,
        "r_mean": rq["mean"],
        "r_p50": rq["p50"],
        "r_p90": rq["p90"],
        "r_p99": rq["p99"],
        "kappa_grad_mean": float(np.mean(kappa_grad)),
        "kappa_grad_p10": float(np.percentile(kappa_grad, 10)),
        "kappa_grad_p50": float(np.percentile(kappa_grad, 50)),
        "kappa_grad_p90": float(np.percentile(kappa_grad, 90)),
        "kappa_q_mean": float(np.mean(kappa_q)),
        "kappa_q_p10": float(np.percentile(kappa_q, 10)),
        "kappa_q_p50": float(np.percentile(kappa_q, 50)),
        "kappa_q_p90": float(np.percentile(kappa_q, 90)),
        "pr_kappa_q_pos": float(np.mean(kappa_q > 0)),
        "pr_tau_eff_kappa_q_gt_0p5": float(np.mean(tau_eff_sum * kappa_q > 0.5)),
        "r_prox_mean": float(np.mean(r_prox)),
        "delta_q1_mean": float(np.mean(q1_a - q1_0)),
        "delta_q2_mean": float(np.mean(q2_a - q2_0)),
        "u_alpha_mean": float(np.mean(np.abs(q1_a - q2_a))),
        "cos_g0_q1q2_mean": float(np.mean(cos_g)),
        "sat_frac_mean": float(np.mean(sat)),
        "linear_gain_mean": float(np.mean(lin)),
    }
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/ext_csv/mpi_sweep_lab/results_qnorm")
    parser.add_argument("--data-dir", default="/raid/ext_csv/datasets/d4rl")
    parser.add_argument("--out-dir", default="/home/ext_csv/mpi_sweep_lab/geometry_logs")
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--seeds", default="0 1")
    args = parser.parse_args()
    results = Path(args.results_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split()]
    cache = {}
    summary_csv = out_root / "geometry_summary.csv"
    rows = []
    if summary_csv.is_file():
        rows = list(csv.DictReader(summary_csv.open()))
    done = {(r["env"], float(r["tau"]), int(r["seed"])) for r in rows}

    meta = {
        "same_frozen_critic": False,
        "critic_activation": "relu",
        "actor_loss": "L = -lambda * mean(Q1(s,pi)) + mean((pi-a_D)^2)",
        "lambda": "alpha / mean(|Q1(s,pi)|) with alpha=2*tau",
        "bc_reduction": "jnp.mean over batch and action dims (not sum)",
        "a0": "dataset action a_D (no separate BC actor in this sweep)",
        "n_states": args.n_states,
    }
    (out_root / "META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_ok = 0
    for env in ENVS:
        for tau in TAUS:
            for seed in seeds:
                key = (env, float(tau), int(seed))
                tag = _tag(env, tau, seed)
                print(f"[geom] {tag}", flush=True)
                summary = dump_cell(
                    env, tau, seed, results, Path(args.data_dir), out_root,
                    args.n_states, cache,
                )
                if summary is None:
                    print(f"[skip] missing ckpt {tag}", flush=True)
                    continue
                n_ok += 1
                if key not in done:
                    rows.append({k: summary[k] for k in summary})
                    done.add(key)
                    fieldnames = list(rows[0].keys())
                    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
    print(f"[geom] done cells={n_ok} csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
