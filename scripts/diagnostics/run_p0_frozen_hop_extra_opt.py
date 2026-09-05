#!/usr/bin/env python3
"""GPU extra-opt: freeze 1M critic + μ_{k-1}, optimize only hop actor k.

Hopper-medium/expert T=10 seeds {0,1} hops {2,3,4}. Uses current JKO loss
(train_td3bc.update_jko). Readout is dataset ΔL vs staying at μ_{k-1}, same
formula as the first90 hop-actor audit.

Hops are independent: hop 3 still references the original 1M μ2, not an
extra-opted μ2. Does not launch shared-driver. Does not mix first90/tail90.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

# Limit JAX memory so this diagnostic can share the box with the live AMO pack.
# Do not force CPU. Do not preallocate a large GPU slice.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.pop("JAX_PLATFORMS", None)
os.environ.pop("JAX_PLATFORM_NAME", None)


def _early_gpu() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    args, _ = parser.parse_known_args()
    gpu = str(args.gpu).strip()
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


_early_gpu()

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()
sys.path.insert(0, str(_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from _p0_hop_metrics import hop_objective_delta  # noqa: E402
from p0_frozen_hop_extra_opt import (  # noqa: E402
    BATCH_SIZE,
    DATASET_N,
    DATASET_SAMPLE_SEED,
    DEFAULT_CACHE,
    EVAL_SEED_BASE,
    FIRST90_CELLS,
    HOPS,
    LR,
    SEEDS,
    TASKS,
    T_VALUE,
    cache_ckpt_path,
    copy_instructions,
    planned_jobs,
    resolve_all,
    sha256_file,
    tau_step,
)
from train_td3bc import (  # noqa: E402
    EVAL_ENV,
    Actor,
    Transition,
    TwinCritic,
    _domain,
    d4rl_normalized_score,
    install_stop_handler,
    load_checkpoint,
    qlearning_from_hdf5,
    sample_batch,
    update_jko,
)
from d4rl_data import dataset_path  # noqa: E402

KST = timezone(timedelta(hours=9))
OUT_DEFAULT = _ROOT / "sweep_results/diagnostics/p0_frozen_hop_extra_opt"


def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--ckpt-root", type=Path, default=_ROOT / DEFAULT_CACHE)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--hops", type=int, nargs="+", default=list(HOPS))
    parser.add_argument("--extra-steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--updates-per-dispatch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _actor_list(payload: Mapping[str, Any]) -> list[Any]:
    if "actors_params" in payload:
        return list(payload["actors_params"])
    keys = ["actor_params", "actor2_params", "actor3_params", "actor4_params"]
    params = [payload[key] for key in keys if key in payload]
    if len(params) < 4:
        raise KeyError("MART K=4 checkpoint must carry four hop actors")
    return params


def _probe_raw(env_name: str, data_dir: Path) -> np.ndarray:
    raw = qlearning_from_hdf5(dataset_path(env_name, data_dir))["observations"]
    rng = np.random.default_rng(DATASET_SAMPLE_SEED)
    index = rng.choice(raw.shape[0], size=min(DATASET_N, raw.shape[0]), replace=False)
    index.sort()
    return np.asarray(raw[index], dtype=np.float32)


def _transition(env_name: str, data_dir: Path, mean: np.ndarray, std: np.ndarray) -> Transition:
    raw = qlearning_from_hdf5(dataset_path(env_name, data_dir))
    obs = (raw["observations"] - mean) / std
    next_obs = (raw["next_observations"] - mean) / std
    return Transition(
        observations=jnp.asarray(obs, dtype=jnp.float32),
        actions=jnp.asarray(raw["actions"], dtype=jnp.float32),
        rewards=jnp.asarray(raw["rewards"], dtype=jnp.float32),
        next_observations=jnp.asarray(next_obs, dtype=jnp.float32),
        not_dones=jnp.asarray(raw["not_dones"], dtype=jnp.float32),
    )


def _save_extra(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _latest_extra(run_dir: Path) -> Path | None:
    files = [
        path
        for path in run_dir.glob("params_extra_*.pkl")
        if path.stem.split("_")[-1].isdigit()
    ]
    if not files:
        return None
    return max(files, key=lambda path: int(path.stem.split("_")[-1]))


def _eval_episodes(
    policy: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    n_episodes: int,
) -> dict[str, float]:
    if n_episodes <= 0:
        return {
            "n_episodes": 0,
            "mean_normalized": float("nan"),
            "mean_raw": float("nan"),
            "mean_length": float("nan"),
        }
    import gymnasium as gym

    env = gym.make(EVAL_ENV[_domain(env_name)])
    raws = []
    lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        done = False
        ep_ret = 0.0
        length = 0
        while not done:
            state = (np.asarray(obs, dtype=np.float32) - mean) / std
            action = np.asarray(policy(state), dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            length += 1
        raws.append(ep_ret)
        lengths.append(length)
    env.close()
    mean_raw = float(np.mean(raws))
    return {
        "n_episodes": n_episodes,
        "mean_normalized": float(d4rl_normalized_score(env_name, mean_raw)),
        "mean_raw": mean_raw,
        "mean_length": float(np.mean(lengths)),
    }


def _probe_terms(
    actor_apply,
    critic_apply,
    hop_params,
    ref_params,
    critic_params,
    states: jax.Array,
    hop_tau: float,
    scale_norm: bool,
) -> dict[str, float]:
    pi = actor_apply(hop_params, states)
    ref = actor_apply(ref_params, states)
    q_k, _ = critic_apply(critic_params, states, pi)
    q_ref, _ = critic_apply(critic_params, states, ref)
    return hop_objective_delta(
        np.asarray(q_k).reshape(-1),
        np.asarray(q_ref).reshape(-1),
        np.asarray(pi),
        np.asarray(ref),
        tau_step=hop_tau,
        scale_norm=scale_norm,
    )


def extra_opt_one_job(
    job: Mapping[str, Any],
    ckpt: Path,
    *,
    data_dir: Path,
    out_dir: Path,
    extra_steps: int,
    log_every: int,
    save_every: int,
    updates_per_dispatch: int,
    batch_size: int,
    lr: float,
    eval_episodes: int,
) -> dict[str, Any]:
    payload = load_checkpoint(ckpt)
    actors = _actor_list(payload)
    hop = int(job["hop"])
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    cfg = payload.get("config") or {}
    scale_norm = bool(cfg.get("q_scale_norm", True))
    hop_tau = tau_step(float(job["T"]), 4)
    parent_sha = sha256_file(ckpt)

    run_dir = (
        out_dir
        / f"{job['environment']}_tau{int(job['T'])}_seed{int(job['seed'])}_hop{hop}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stop = install_stop_handler(run_dir)

    action_dim = int(np.asarray(actors[0]["params"]["Dense_2"]["kernel"]).shape[-1])
    actor_model = Actor(action_dim=action_dim, max_action=max_action)
    critic_model = TwinCritic()
    hop_actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actors[hop - 1],
        tx=optax.adam(lr),
    )
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=payload["critic_params"],
        tx=optax.set_to_zero(),
    )
    ref_params = actors[hop - 2]
    start_step = 0
    resume = _latest_extra(run_dir)
    if resume is not None:
        extra = load_checkpoint(resume)
        if extra.get("parent_sha256") != parent_sha:
            raise ValueError(f"extra-opt resume parent hash mismatch: {resume}")
        if int(extra["hop"]) != hop:
            raise ValueError(f"extra-opt resume hop mismatch: {resume}")
        hop_actor = hop_actor.replace(
            params=extra["actor_params"],
            opt_state=extra["actor_opt_state"],
            step=extra.get("actor_opt_step", hop_actor.step),
        )
        start_step = int(extra["step"])
        print(f"[resume] {resume} step={start_step}", flush=True)

    data = _transition(job["environment"], data_dir, mean, std)
    probe_raw = _probe_raw(job["environment"], data_dir)
    probe_states = jnp.asarray((probe_raw - mean) / std, dtype=jnp.float32)
    rng = jax.random.PRNGKey(int(job["seed"]) * 1000 + hop + start_step)

    @jax.jit
    def dispatch(hop_actor, rng, start_it, n_updates):
        def body(i, carry):
            hop_actor, rng, last_loss = carry
            rng, b_rng = jax.random.split(rng)
            batch = sample_batch(data, b_rng, batch_size)
            ref = hop_actor.apply_fn(ref_params, batch.observations)
            hop_actor, loss = update_jko(
                hop_actor,
                hop_actor.apply_fn,
                critic,
                batch,
                ref,
                hop_tau,
                scale_norm,
            )
            return hop_actor, rng, loss

        return jax.lax.fori_loop(
            0, n_updates, body, (hop_actor, rng, jnp.asarray(0.0))
        )

    actor_apply = jax.jit(actor_model.apply)
    critic_apply = jax.jit(critic_model.apply)

    def policy_fn(state: np.ndarray) -> np.ndarray:
        return np.asarray(
            actor_apply(hop_actor.params, jnp.asarray(state, dtype=jnp.float32)),
            dtype=np.float32,
        )

    def snapshot(step: int, train_loss: float) -> dict[str, Any]:
        terms = _probe_terms(
            actor_apply,
            critic_apply,
            hop_actor.params,
            ref_params,
            critic.params,
            probe_states,
            hop_tau,
            scale_norm,
        )
        row = {
            "environment": job["environment"],
            "T": int(job["T"]),
            "seed": int(job["seed"]),
            "hop": hop,
            "step": int(step),
            "train_loss": float(train_loss),
            "checkpoint_sha256": parent_sha,
            **terms,
        }
        return row

    def persist(step: int, *, reason: str) -> None:
        _save_extra(
            run_dir / f"params_extra_{int(step)}.pkl",
            {
                "step": int(step),
                "hop": hop,
                "environment": job["environment"],
                "seed": int(job["seed"]),
                "T": int(job["T"]),
                "parent_sha256": parent_sha,
                "parent_checkpoint": str(ckpt),
                "actor_params": jax.device_get(hop_actor.params),
                "actor_opt_state": jax.device_get(hop_actor.opt_state),
                "actor_opt_step": jax.device_get(hop_actor.step),
                "critic_params": jax.device_get(critic.params),
                "ref_actor_params": jax.device_get(ref_params),
                "reason": reason,
            },
        )

    metrics_path = run_dir / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    if start_step == 0 or not metrics_path.is_file():
        row0 = snapshot(start_step, train_loss=float("nan"))
        rows.append(row0)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row0, sort_keys=True, default=str) + "\n")
        print(
            f"[probe] {job['environment']} seed={job['seed']} hop={hop} "
            f"step={start_step} delta_L={row0['delta_L']:.6g} "
            f"obj_up={row0['objective_improved']}",
            flush=True,
        )

    eval_start = _eval_episodes(
        policy_fn, job["environment"], mean, std, eval_episodes
    )
    step = start_step
    last_loss = 0.0
    while step < extra_steps:
        if stop["flag"]:
            persist(step, reason="signal")
            print(f"[signal] emergency save at step={step}", flush=True)
            break
        n = min(updates_per_dispatch, extra_steps - step)
        hop_actor, rng, loss = dispatch(hop_actor, rng, step, n)
        last_loss = float(loss)
        step += n
        on_schedule = step % log_every == 0 or step >= extra_steps
        if on_schedule:
            row = snapshot(step, last_loss)
            rows.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            print(
                f"[opt] {job['environment']} seed={job['seed']} hop={hop} "
                f"step={step} train_L={last_loss:.6g} delta_L={row['delta_L']:.6g} "
                f"obj_up={row['objective_improved']}",
                flush=True,
            )
        if save_every and step % save_every == 0:
            persist(step, reason="schedule")
    else:
        persist(step, reason="final")

    eval_end = _eval_episodes(policy_fn, job["environment"], mean, std, eval_episodes)
    summary = {
        "environment": job["environment"],
        "T": int(job["T"]),
        "seed": int(job["seed"]),
        "hop": hop,
        "parent_checkpoint": str(ckpt),
        "parent_sha256": parent_sha,
        "extra_steps_requested": extra_steps,
        "extra_steps_done": step,
        "stopped_by_signal": bool(stop["flag"]),
        "delta_L_start": rows[0]["delta_L"] if rows else None,
        "delta_L_end": rows[-1]["delta_L"] if rows else None,
        "objective_improved_start": rows[0]["objective_improved"] if rows else None,
        "objective_improved_end": rows[-1]["objective_improved"] if rows else None,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "run_dir": str(run_dir),
        "finished_at": now_kst(),
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    return summary


def write_report(out_dir: Path, summaries: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# P0 frozen-critic hop extra-opt (Hopper first90)",
        "",
        f"Finished: {now_kst()}",
        "",
        "Freeze the 1M critic and μ_{k-1}. Extra-optimize only μ_k on the current",
        "JKO loss. Hops are independent (hop 3 still uses original μ2).",
        "",
        f"Backend: `{meta.get('backend')}`  GPU: `{meta.get('gpu')}`",
        f"Extra steps: {meta.get('extra_steps')}",
        "",
        "| task | seed | hop | ΔL start | ΔL end | obj_up start | obj_up end | J start | J end |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in summaries:
        j0 = row["eval_start"].get("mean_normalized")
        j1 = row["eval_end"].get("mean_normalized")
        lines.append(
            f"| {row['environment']} | {row['seed']} | {row['hop']} | "
            f"{row['delta_L_start']:.6g} | {row['delta_L_end']:.6g} | "
            f"{row['objective_improved_start']} | {row['objective_improved_end']} | "
            f"{j0:.3f} | {j1:.3f} |"
        )
    n_end_up = sum(1 for row in summaries if row.get("objective_improved_end"))
    lines.extend(
        [
            "",
            f"Hops with ΔL<0 after extra-opt: {n_end_up}/{len(summaries)}.",
            "Negative ΔL means the hop loss is better than staying at μ_{k-1}.",
            "",
        ]
    )
    (out_dir / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    backend = jax.default_backend()
    print(
        json.dumps(
            {
                "backend": backend,
                "devices": [str(d) for d in jax.devices()],
                "gpu": args.gpu,
                "started_at": now_kst(),
            }
        ),
        flush=True,
    )
    if args.require_gpu and backend != "gpu" and not args.allow_cpu:
        raise RuntimeError(
            f"refusing non-GPU JAX backend {backend!r}; pass --allow-cpu to override"
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.ckpt_root
    jobs = planned_jobs(args.tasks, args.seeds, args.hops)
    found, missing = resolve_all(jobs, cache_root=cache_root, extra_roots=())
    if missing:
        text = copy_instructions(missing, cache_root=cache_root)
        (out_dir / "COPY_INSTRUCTIONS.md").write_text(text, encoding="utf-8")
        _write_json(
            out_dir / "STATUS.json",
            {
                "status": "blocked_missing_first90_ckpts",
                "backend": backend,
                "gpu": args.gpu,
                "missing": [
                    {
                        "environment": cell["environment"],
                        "seed": cell["seed"],
                        "expected_sha256": cell["expected_sha256"],
                        "inventory_path": cell["inventory_path"],
                        "cache_path": str(
                            cache_ckpt_path(cache_root, cell["environment"], int(cell["seed"]))
                        ),
                    }
                    for cell in missing
                ],
                "n_jobs": len(jobs),
                "started_at": now_kst(),
            },
        )
        print(text, flush=True)
        return 2

    summaries = []
    for job in jobs:
        ckpt = found[(job["environment"], int(job["seed"]))]
        print(
            f"[job] {job['environment']} seed={job['seed']} hop={job['hop']} ckpt={ckpt}",
            flush=True,
        )
        summaries.append(
            extra_opt_one_job(
                job,
                ckpt,
                data_dir=args.data_dir,
                out_dir=out_dir,
                extra_steps=args.extra_steps,
                log_every=args.log_every,
                save_every=args.save_every,
                updates_per_dispatch=args.updates_per_dispatch,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_episodes=args.eval_episodes,
            )
        )
        if summaries[-1]["stopped_by_signal"]:
            break

    with (out_dir / "summaries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "environment",
                "T",
                "seed",
                "hop",
                "delta_L_start",
                "delta_L_end",
                "objective_improved_start",
                "objective_improved_end",
                "eval_start_normalized",
                "eval_end_normalized",
                "extra_steps_done",
                "parent_sha256",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "environment": row["environment"],
                    "T": row["T"],
                    "seed": row["seed"],
                    "hop": row["hop"],
                    "delta_L_start": row["delta_L_start"],
                    "delta_L_end": row["delta_L_end"],
                    "objective_improved_start": row["objective_improved_start"],
                    "objective_improved_end": row["objective_improved_end"],
                    "eval_start_normalized": row["eval_start"]["mean_normalized"],
                    "eval_end_normalized": row["eval_end"]["mean_normalized"],
                    "extra_steps_done": row["extra_steps_done"],
                    "parent_sha256": row["parent_sha256"],
                }
            )

    meta = {
        "status": "complete" if all(not r["stopped_by_signal"] for r in summaries) else "signaled",
        "backend": backend,
        "gpu": args.gpu,
        "extra_steps": args.extra_steps,
        "n_jobs": len(summaries),
        "cells": [dict(cell) for cell in FIRST90_CELLS],
        "finished_at": now_kst(),
    }
    _write_json(out_dir / "STATUS.json", meta)
    write_report(out_dir, summaries, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
