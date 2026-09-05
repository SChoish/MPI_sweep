#!/usr/bin/env python3
"""CPU eval of every stored P0 K=4 actor. No training, no GPU.

Re-evaluates MART actors 1-4 and two-actor target/deployment under one
protocol. Does not mix archived first/endpoint scores into this table.
Missing ext_csh checkpoints are skipped, not replaced.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EIGEN_NUM_THREADS", "1")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _ckpt_compat import normalize_checkpoint  # noqa: E402
from _p0_hop_common import (  # noqa: E402
    ARCHIVE,
    EPISODE_FIELDS,
    EVAL_SEED_BASE,
    ROLLOUT_N_EACH,
    ROLLOUT_SAMPLE_SEED,
    actor_roles,
    now_kst,
    write_inventory,
)
from train_td3bc import (  # noqa: E402
    EVAL_ENV,
    Actor,
    _domain,
    d4rl_normalized_score,
    load_checkpoint,
)


def _infer_action_dim(params: Any, mean: np.ndarray) -> int:
    kernel = np.asarray(params["params"]["Dense_0"]["kernel"])
    output_kernel = np.asarray(params["params"]["Dense_2"]["kernel"])
    if kernel.shape[0] != mean.shape[0]:
        raise ValueError("actor input dim does not match checkpoint mean")
    return int(output_kernel.shape[-1])


def _make_policy(actor: Actor, params: Any):
    apply = jax.jit(actor.apply)

    def policy(state: np.ndarray) -> np.ndarray:
        x = jnp.asarray(state, dtype=jnp.float32)
        if x.ndim == 1:
            x = x[None, :]
            return np.asarray(apply(params, x)[0], dtype=np.float32)
        return np.asarray(apply(params, x), dtype=np.float32)

    return policy


def _actor_params(payload: dict[str, Any], actor_index: int):
    actors = payload.get("actors_params")
    if actors is not None:
        return actors[actor_index - 1]
    normalized = normalize_checkpoint(payload)
    key = "actor_params" if actor_index == 1 else f"actor{actor_index}_params"
    return normalized[key]


def evaluate_actor_episodes(
    policy,
    env_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    episodes: int,
    collect_raw_states: bool,
    n_states: int,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    import gymnasium as gym

    env = gym.make(EVAL_ENV[_domain(env_name)])
    collected: list[np.ndarray] = []
    rng = np.random.default_rng(sample_seed)
    rows = []
    for ep in range(episodes):
        eval_seed = EVAL_SEED_BASE + ep
        obs, _ = env.reset(seed=eval_seed)
        done = False
        ep_ret = 0.0
        length = 0
        while not done:
            raw = np.asarray(obs, dtype=np.float32)
            if collect_raw_states and len(collected) < n_states:
                if rng.random() < 0.08 or len(collected) == 0:
                    collected.append(raw.copy())
            state = (raw - mean) / std
            action = np.asarray(policy(state), dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            length += 1
        rows.append(
            {
                "evaluation_seed": eval_seed,
                "raw_return": ep_ret,
                "normalized_score": float(d4rl_normalized_score(env_name, ep_ret)),
                "episode_length": length,
            }
        )
    env.close()
    states = np.stack(collected[:n_states]) if collected else None
    return rows, states


def _done_keys(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (
                row["task"],
                row["T"],
                row["training_seed"],
                row["method"],
                row["actor_index"],
                row["evaluation_seed"],
            )
            for row in csv.DictReader(handle)
        }


def _append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()


def eval_one(
    record: dict[str, Any],
    actor_spec: dict[str, Any],
    episodes: int,
    output_csv: Path,
    raw_dir: Path,
) -> None:
    ckpt = Path(record["checkpoint_path"])
    payload = load_checkpoint(ckpt)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    params = _actor_params(payload, actor_spec["actor_index"])
    actor = Actor(action_dim=_infer_action_dim(params, mean), max_action=max_action)
    import hashlib

    collect = actor_spec["actor_role"] in ("mu4", "deployment")
    seed_key = (
        f"{record['environment']}|{record['T']}|{record['seed']}|{actor_spec['actor_role']}"
    )
    sample_seed = ROLLOUT_SAMPLE_SEED + (
        int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], "little") % 10_000
    )
    t0 = time.time()
    ep_rows, states = evaluate_actor_episodes(
        _make_policy(actor, params),
        record["environment"],
        mean,
        std,
        episodes,
        collect,
        ROLLOUT_N_EACH,
        sample_seed,
    )
    elapsed = time.time() - t0
    out_rows = []
    for ep in ep_rows:
        out_rows.append(
            {
                "task": record["environment"],
                "T": str(record["T"]),
                "training_seed": str(record["seed"]),
                "method": record["method"],
                "actor_index": str(actor_spec["actor_index"]),
                "actor_role": actor_spec["actor_role"],
                "checkpoint_hash": record["checkpoint_hash"],
                "checkpoint_step": str(record["checkpoint"]["step"]),
                "evaluation_seed": str(ep["evaluation_seed"]),
                "raw_return": f"{ep['raw_return']}",
                "normalized_score": f"{ep['normalized_score']}",
                "episode_length": str(ep["episode_length"]),
                "host_run_dir": record["host_run_dir"],
            }
        )
    _append_rows(output_csv, out_rows)
    if states is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_dir
            / (
                f"{record['method']}_{record['environment']}_T{record['T']}_"
                f"s{record['seed']}_{actor_spec['actor_role']}.npz"
            ),
            raw_observations=states,
            sampling_seed=np.asarray(sample_seed),
            actor_role=np.asarray(actor_spec["actor_role"]),
        )
    print(
        f"{record['method']} {record['environment']} T={record['T']} "
        f"seed={record['seed']} {actor_spec['actor_role']} "
        f"mean_norm={np.mean([ep['normalized_score'] for ep in ep_rows]):.2f} "
        f"{elapsed:.1f}s",
        flush=True,
    )


def _jobs(inventory: dict[str, Any], limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    jobs = []
    for record in inventory["rows"]:
        if record["status"] != "evalable":
            continue
        for spec in actor_roles(record["condition"]):
            jobs.append((record, spec))
            if limit and len(jobs) >= limit:
                return jobs
    return jobs


def _remaining(jobs, output_csv: Path, episodes: int):
    done = _done_keys(output_csv)
    leftover = []
    for record, spec in jobs:
        needed = [
            (
                record["environment"],
                str(record["T"]),
                str(record["seed"]),
                record["method"],
                str(spec["actor_index"]),
                str(EVAL_SEED_BASE + ep),
            )
            for ep in range(episodes)
        ]
        if not all(key in done for key in needed):
            leftover.append((record, spec))
    return leftover


def _worker(worker_id: int, jobs, output_csv: str, raw_dir: str, episodes: int, cpu0: int) -> None:
    try:
        os.sched_setaffinity(0, {cpu0 + worker_id})
    except OSError:
        pass
    out = Path(output_csv)
    raw = Path(raw_dir)
    for record, spec in jobs:
        eval_one(record, spec, episodes, out, raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu0", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"refusing non-CPU JAX backend {jax.default_backend()!r}")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    inventory = write_inventory()
    jobs = _jobs(inventory, args.limit)
    if args.smoke:
        jobs = [
            job
            for job in jobs
            if job[0]["environment"] == "hopper-medium-v2"
            and job[0]["T"] == 4
            and job[0]["seed"] == 0
        ]
    output_csv = ARCHIVE / "episode_scores.csv"
    raw_dir = ARCHIVE / "raw" / "rollout_states"
    leftover = _remaining(jobs, output_csv, args.episodes)
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "n_inventory_evalable": inventory["coverage"]["n_evalable"],
                "n_jobs": len(jobs),
                "n_remaining": len(leftover),
                "workers": min(args.workers, max(1, len(leftover))),
                "started_at": now_kst(),
            },
            indent=2,
        ),
        flush=True,
    )
    if not leftover:
        return 0
    workers = max(1, min(int(args.workers), len(leftover)))
    if workers == 1:
        for record, spec in leftover:
            eval_one(record, spec, args.episodes, output_csv, raw_dir)
        return 0
    shards: list[list] = [[] for _ in range(workers)]
    for index, job in enumerate(leftover):
        shards[index % workers].append(job)
    ctx = mp.get_context("spawn")
    procs = []
    for worker_id, shard in enumerate(shards):
        if not shard:
            continue
        proc = ctx.Process(
            target=_worker,
            args=(
                worker_id,
                shard,
                str(output_csv),
                str(raw_dir),
                args.episodes,
                args.cpu0,
            ),
        )
        proc.start()
        procs.append(proc)
    failures = 0
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failures += 1
    if failures:
        raise RuntimeError(f"{failures} CPU eval workers failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
