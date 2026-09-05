#!/usr/bin/env python3
"""CPU diagnostics: inventory, all-actor env eval, and common-state actions.

Does not train or modify checkpoints. Forces JAX onto CPU. Limit ``--workers``
so a live trainer is not starved. Resume is per-policy JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from _ckpt_compat import normalize_checkpoint  # noqa: E402
from d4rl_data import dataset_path  # noqa: E402
from scripts.diagnostics.p0_intermediate_actors import (  # noqa: E402
    ACTORS_BY_CONDITION,
    BOUND_FRAC,
    COSINE_MIN_L2,
    DATASET_SAMPLE_SEED,
    DEFAULT_OUT_DIR,
    DEFAULT_P0_ROOT,
    DEFAULT_TAIL_CHECKPOINTS,
    DEFAULT_TAIL_MANIFEST,
    EVAL_EPISODE_SEEDS,
    FROZEN_SOURCE_REVISION,
    METHOD_LABEL,
    N_DATASET_STATES,
    N_ROLLOUT_PER_POLICY,
    Q_SCALE_EPS,
    ROLLOUT_SAMPLE_SEED,
    ActorSpec,
    bound_fraction,
    coverage_stats,
    cosine_with_near_zero_mask,
    episode_part_path,
    hop_tau,
    load_json,
    pair_key,
    planned_runs,
    rms_action_difference,
    run_key,
    sha256_file,
    vector_l2,
    write_csv,
    write_json,
)
from train_td3bc import (  # noqa: E402
    EVAL_ENV,
    Actor,
    TwinCritic,
    _domain,
    d4rl_normalized_score,
    load_checkpoint,
    qlearning_from_hdf5,
)


def _kernel_shapes(tree: Any, prefix: str = "") -> list[tuple[str, tuple[int, ...]]]:
    out: list[tuple[str, tuple[int, ...]]] = []
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            out.extend(_kernel_shapes(value, f"{prefix}/{key}"))
    elif hasattr(tree, "shape") and prefix.endswith("kernel"):
        out.append((prefix, tuple(int(x) for x in tree.shape)))
    return out


def infer_action_dim(params: Any) -> int:
    shapes = _kernel_shapes(params)
    if not shapes:
        raise ValueError("cannot infer action dimension from actor params")
    return int(shapes[-1][1][-1])


def load_payload(checkpoint: Path) -> dict[str, Any]:
    return normalize_checkpoint(load_checkpoint(checkpoint))


def actor_params_list(payload: Mapping[str, Any]) -> list[Any]:
    if "actors_params" in payload:
        return list(payload["actors_params"])
    keys = ["actor_params", "actor2_params", "actor3_params", "actor4_params"]
    params = [payload[key] for key in keys if key in payload]
    if not params:
        raise KeyError("checkpoint has neither actors_params nor actor_params")
    return params


def build_inventory(
    *,
    checkpoints_path: Path,
    manifest_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    recorded = load_json(checkpoints_path) if checkpoints_path.is_file() else {}
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    diverged = set(manifest.get("optimizer_diverged_runs", []))
    rows = []
    for plan in planned_runs():
        key = plan["run_key"]
        rec = recorded.get(key)
        row = dict(plan)
        row.update(
            {
                "host_shard": "ext_csh_tail90" if rec else None,
                "checkpoint": rec["checkpoint"] if rec else None,
                "recorded_file_sha256": rec.get("file_sha256") if rec else None,
                "recorded_weights_sha256": rec.get("weights_sha256") if rec else None,
                "recorded_normalization_sha256": rec.get("normalization_sha256")
                if rec
                else None,
                "frozen_source_revision": FROZEN_SOURCE_REVISION,
                "optimizer_diverged": key in diverged,
                "eval_eligible": False,
                "missing_reason": None,
            }
        )
        if rec is None:
            row["missing_reason"] = (
                "not in ext_csh frozen tail90 CHECKPOINTS.json; "
                "head shard lives on ext_csv and is not substituted"
            )
            rows.append(row)
            continue
        ckpt = Path(rec["checkpoint"])
        row["checkpoint"] = str(ckpt)
        if not ckpt.is_file():
            row["missing_reason"] = f"recorded checkpoint missing on disk: {ckpt}"
            rows.append(row)
            continue
        cfg_path = ckpt.parent / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        expected = len(ACTORS_BY_CONDITION[plan["condition"]])
        file_hash = rec.get("file_sha256") or sha256_file(ckpt)
        step = int(rec.get("step", 1_000_000))
        method_ok = (
            (plan["condition"] == "bar_p4" and cfg.get("method") == "bar" and int(cfg.get("mpi_steps", 4)) == 4)
            or (plan["condition"] == "two_actor_p4" and cfg.get("method") == "mcep" and int(cfg.get("mpi_steps", 4)) == 4)
            or not cfg
        )
        row.update(
            {
                "eval_eligible": step == 1_000_000 and method_ok,
                "file_sha256": file_hash,
                "step": step,
                "n_actors_in_ckpt": expected,
                "expected_n_actors": expected,
                "config_method": cfg.get("method"),
                "config_mpi_steps": cfg.get("mpi_steps"),
                "config_tau": cfg.get("tau"),
                "config_normalize": cfg.get("normalize"),
                "config_q_scale_norm": cfg.get("q_scale_norm"),
                "config_integrator": cfg.get("integrator"),
                "max_action": 1.0,
                "mean_sha256": rec.get("normalization_sha256"),
                "std_sha256": rec.get("normalization_sha256"),
                "eval_env": EVAL_ENV[_domain(plan["environment"])],
                "dataset_path": str(dataset_path(plan["environment"], data_dir)),
                "restore_ok": None,
            }
        )
        if not method_ok:
            row["missing_reason"] = f"config method/K mismatch: {cfg.get('method')} mpi_steps={cfg.get('mpi_steps')}"
        elif step != 1_000_000:
            row["missing_reason"] = f"checkpoint step is {step}, not 1000000"
        rows.append(row)
    stats = coverage_stats(rows)
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frozen_source_revision": FROZEN_SOURCE_REVISION,
        "checkpoints_path": str(checkpoints_path),
        "manifest_path": str(manifest_path),
        "eval_episode_seeds": list(EVAL_EPISODE_SEEDS),
        "eval_stack": {
            "env_map": EVAL_ENV,
            "episodes": 10,
            "shared_episode_seeds_by_task": True,
            "deterministic": True,
            "do_not_mix_with_historical_first_endpoint_scores": True,
        },
        "coverage": stats,
        "runs": rows,
    }


def evaluate_one_policy(job: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym
    import jax

    payload = load_payload(Path(job["checkpoint"]))
    params_list = actor_params_list(payload)
    actor_index = int(job["actor_index"]) - 1
    params = params_list[actor_index]
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    actor = Actor(action_dim=infer_action_dim(params), max_action=max_action)
    apply = jax.jit(actor.apply)
    env_name = job["environment"]
    env = gym.make(EVAL_ENV[_domain(env_name)])
    episodes = []
    collected_raw = []
    for eval_seed in job["eval_seeds"]:
        obs, _ = env.reset(seed=int(eval_seed))
        done = False
        ep_ret = 0.0
        length = 0
        raw_steps = []
        while not done:
            raw = np.asarray(obs, dtype=np.float32)
            raw_steps.append(raw)
            state = (raw - mean) / std
            action = np.asarray(apply(params, state), dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            length += 1
        collected_raw.append(np.stack(raw_steps, axis=0))
        episodes.append(
            {
                "run_key": job["run_key"],
                "environment": env_name,
                "T": job["T"],
                "seed": job["seed"],
                "condition": job["condition"],
                "method": job["method_label"],
                "actor_index": job["actor_index"],
                "actor_id": job["actor_id"],
                "checkpoint_sha256": job["file_sha256"],
                "eval_seed": int(eval_seed),
                "raw_return": ep_ret,
                "normalized_score": float(d4rl_normalized_score(env_name, ep_ret)),
                "episode_length": length,
            }
        )
    env.close()
    rng = np.random.default_rng(ROLLOUT_SAMPLE_SEED + job["actor_index"] + 17 * job["seed"])
    concat = np.concatenate(collected_raw, axis=0)
    take = min(N_ROLLOUT_PER_POLICY, int(concat.shape[0]))
    idx = rng.choice(concat.shape[0], size=take, replace=False)
    rollout = concat[np.sort(idx)]
    return {"episodes": episodes, "rollout_raw": rollout.astype(np.float32)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eval_jobs_from_inventory(inventory: Mapping[str, Any], *, pilot: bool) -> list[dict[str, Any]]:
    eligible = [row for row in inventory["runs"] if row.get("eval_eligible")]
    if pilot and eligible:
        first = eligible[0]
        target = pair_key(first["environment"], first["T"], first["seed"])
        eligible = [
            row
            for row in eligible
            if pair_key(row["environment"], row["T"], row["seed"]) == target
        ]
    jobs = []
    for row in eligible:
        for spec in ACTORS_BY_CONDITION[row["condition"]]:
            jobs.append(
                {
                    "run_key": row["run_key"],
                    "environment": row["environment"],
                    "T": row["T"],
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "method_label": row["method_label"],
                    "actor_index": spec.actor_index,
                    "actor_id": spec.actor_id,
                    "checkpoint": row["checkpoint"],
                    "file_sha256": row["file_sha256"],
                    "eval_seeds": list(EVAL_EPISODE_SEEDS),
                }
            )
    return jobs


def run_eval(inventory: Mapping[str, Any], out_dir: Path, workers: int, pilot: bool) -> None:
    jobs = eval_jobs_from_inventory(inventory, pilot=pilot)
    pending = []
    for job in jobs:
        part = episode_part_path(out_dir, job["run_key"], job["actor_id"])
        existing = _read_jsonl(part)
        if len(existing) == len(EVAL_EPISODE_SEEDS):
            continue
        pending.append(job)
    print(
        f"[eval] jobs={len(jobs)} pending={len(pending)} workers={workers} pilot={pilot}",
        flush=True,
    )
    if not pending:
        merge_episodes(out_dir)
        return
    raw_dir = out_dir / "raw" / "rollout_states"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _commit(job: dict[str, Any], result: dict[str, Any]) -> None:
        part = episode_part_path(out_dir, job["run_key"], job["actor_id"])
        _write_jsonl(part, result["episodes"])
        np.savez_compressed(
            raw_dir / f"{part.stem}.npz",
            raw_observations=result["rollout_raw"],
            run_key=np.asarray(job["run_key"]),
            actor_id=np.asarray(job["actor_id"]),
            sampling_seed=np.asarray(ROLLOUT_SAMPLE_SEED),
        )

    if workers <= 1:
        for i, job in enumerate(pending, start=1):
            result = evaluate_one_policy(job)
            _commit(job, result)
            print(f"[eval] finished {i}/{len(pending)} last={job['run_key']}|{job['actor_id']}", flush=True)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(evaluate_one_policy, job): job for job in pending}
            done = 0
            for fut in as_completed(futures):
                job = futures[fut]
                result = fut.result()
                _commit(job, result)
                done += 1
                if done == 1 or done % 10 == 0 or done == len(pending):
                    print(
                        f"[eval] finished {done}/{len(pending)} last={job['run_key']}|{job['actor_id']}",
                        flush=True,
                    )
    merge_episodes(out_dir)


def merge_episodes(out_dir: Path) -> None:
    rows = []
    for path in sorted((out_dir / "episodes").glob("*.jsonl")):
        rows.extend(_read_jsonl(path))
    fieldnames = [
        "environment",
        "T",
        "seed",
        "condition",
        "method",
        "actor_index",
        "actor_id",
        "run_key",
        "checkpoint_sha256",
        "eval_seed",
        "raw_return",
        "normalized_score",
        "episode_length",
    ]
    write_csv(out_dir / "episodes.csv", rows, fieldnames)
    print(f"[eval] wrote {len(rows)} episode rows", flush=True)


def _sample_dataset_states(env_name: str, data_dir: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    raw = qlearning_from_hdf5(dataset_path(env_name, data_dir))
    obs = np.asarray(raw["observations"], dtype=np.float32)
    rng = np.random.default_rng(DATASET_SAMPLE_SEED)
    take = min(n, obs.shape[0])
    idx = np.sort(rng.choice(obs.shape[0], size=take, replace=False))
    return idx, obs[idx]


def _apply_batch(apply, params, states: np.ndarray, chunk: int = 512) -> np.ndarray:
    import jax.numpy as jnp

    out = []
    for start in range(0, states.shape[0], chunk):
        batch = jnp.asarray(states[start : start + chunk])
        out.append(np.asarray(apply(params, batch)))
    return np.concatenate(out, axis=0)


def _q1_and_grad(critic, critic_params, states: np.ndarray, actions: np.ndarray):
    import jax
    import jax.numpy as jnp

    def q1_values(obs, act):
        q1, _ = critic.apply(critic_params, obs, act)
        return jnp.squeeze(q1, axis=-1)

    def q1_grad(obs, act):
        def scalar(a):
            q1, _ = critic.apply(critic_params, obs[None], a[None])
            return jnp.squeeze(q1)

        return jax.grad(scalar)(act)

    q_jit = jax.jit(q1_values)
    g_jit = jax.jit(jax.vmap(q1_grad, in_axes=(0, 0)))
    q = np.asarray(q_jit(jnp.asarray(states), jnp.asarray(actions)))
    grads = np.asarray(g_jit(jnp.asarray(states), jnp.asarray(actions)))
    return q, grads


def _normalize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (raw - mean) / std


def _action_pack_for_ckpt(
    payload: Mapping[str, Any],
    specs: tuple[ActorSpec, ...],
    raw_states: np.ndarray,
    total_t: float,
) -> dict[str, Any]:
    params_list = actor_params_list(payload)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    states = _normalize(raw_states, mean, std)
    actor = Actor(action_dim=infer_action_dim(params_list[0]), max_action=max_action)
    import jax

    apply = jax.jit(actor.apply)
    actions = {spec.actor_id: _apply_batch(apply, params_list[spec.actor_index - 1], states) for spec in specs}
    critic = TwinCritic()
    critic_params = payload["critic_params"]
    q_scale_norm = bool((payload.get("config") or {}).get("q_scale_norm", True))
    hop_stats = []
    for spec in specs:
        act = actions[spec.actor_id]
        if spec.actor_id.startswith("mart_mu") and spec.actor_index > 1:
            ref = actions[f"mart_mu{spec.actor_index - 1}"]
        else:
            ref = act
        q_ref, grads_at_pi = _q1_and_grad(critic, critic_params, states, act)
        q_for_scale, _ = _q1_and_grad(critic, critic_params, states, ref)
        tau_h = hop_tau(total_t, spec.tau_multiplier)
        if q_scale_norm:
            q_scale = float(np.mean(np.abs(q_for_scale)) + Q_SCALE_EPS)
            lam = 2.0 * tau_h / q_scale
        else:
            q_scale = float("nan")
            lam = 2.0 * tau_h
        hop_stats.append(
            {
                "actor_id": spec.actor_id,
                "tau_hop": tau_h,
                "q_scale": q_scale,
                "lambda": lam,
                "action_grad_norm_mean": float(np.mean(np.linalg.norm(grads_at_pi, axis=-1))),
                "q_at_pi_mean": float(np.mean(q_ref)),
                "bound_fraction": bound_fraction(act, max_action),
                "q_scale_formula": spec.q_scale_ref if q_scale_norm else "unnormalized_2T",
            }
        )
    return {
        "actions": actions,
        "max_action": max_action,
        "hop_stats": hop_stats,
        "q_scale_norm": q_scale_norm,
    }


def _distance_rows(
    *,
    state_set: str,
    environment: str,
    tau: int,
    seed: int,
    n_states: int,
    mart: dict[str, Any] | None,
    two: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(metric: str, values: np.ndarray, extra: dict[str, Any] | None = None) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        row = {
            "state_set": state_set,
            "environment": environment,
            "T": tau,
            "seed": seed,
            "n_states": n_states,
            "metric": metric,
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "n_finite": int(finite.size),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    if mart is not None:
        acts = mart["actions"]
        for prev, nxt in ((1, 2), (2, 3), (3, 4)):
            a = acts[f"mart_mu{prev}"]
            b = acts[f"mart_mu{nxt}"]
            add(f"mart_adjacent_rms_mu{prev}_to_mu{nxt}", rms_action_difference(a, b))
            delta = np.asarray(b) - np.asarray(a)
            add(f"mart_adjacent_l2_mu{prev}_to_mu{nxt}", vector_l2(a, b))
            add(f"mart_delta_mu{nxt}_minus_mu{prev}_rms", rms_action_difference(a, b))
        mu1 = acts["mart_mu1"]
        for nxt in (2, 3, 4):
            add(f"mart_from_mu1_rms_mu{nxt}", rms_action_difference(mu1, acts[f"mart_mu{nxt}"]))
        d2 = np.asarray(acts["mart_mu2"]) - np.asarray(acts["mart_mu1"])
        d3 = np.asarray(acts["mart_mu3"]) - np.asarray(acts["mart_mu2"])
        d4 = np.asarray(acts["mart_mu4"]) - np.asarray(acts["mart_mu3"])
        for name, vec in (("delta2", d2), ("delta3", d3), ("delta4", d4)):
            add(f"mart_{name}_rms", np.sqrt(np.mean(vec * vec, axis=-1)))
            add(f"mart_{name}_l2", np.linalg.norm(vec, axis=-1))
        for left, right, label in (
            (d2, d3, "delta2_delta3"),
            (d3, d4, "delta3_delta4"),
            (d2, d4, "delta2_delta4"),
        ):
            cosine, valid = cosine_with_near_zero_mask(left, right, COSINE_MIN_L2)
            add(f"mart_cosine_{label}", cosine[valid], extra={"n_near_zero_excluded": int((~valid).sum())})
        for spec_id, act in acts.items():
            add(f"bound_frac_{spec_id}", np.asarray([bound_fraction(act, mart["max_action"])]))
    if mart is not None and two is not None:
        add(
            "mart_mu4_vs_two_deploy_rms",
            rms_action_difference(mart["actions"]["mart_mu4"], two["actions"]["two_actor_deploy"]),
        )
        add(
            "mart_mu1_vs_two_target_rms",
            rms_action_difference(mart["actions"]["mart_mu1"], two["actions"]["two_actor_target"]),
        )
        add(
            "two_target_vs_deploy_rms",
            rms_action_difference(two["actions"]["two_actor_target"], two["actions"]["two_actor_deploy"]),
        )
        for spec_id, act in two["actions"].items():
            add(f"bound_frac_{spec_id}", np.asarray([bound_fraction(act, two["max_action"])]))
    return rows


def run_actions(inventory: Mapping[str, Any], out_dir: Path, data_dir: Path) -> None:
    eligible = [row for row in inventory["runs"] if row.get("eval_eligible")]
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in eligible:
        by_pair.setdefault(pair_key(row["environment"], row["T"], row["seed"]), {})[row["condition"]] = row

    ds_dir = out_dir / "raw" / "dataset_states"
    ds_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache: dict[str, np.ndarray] = {}
    distance_rows: list[dict[str, Any]] = []
    hop_rows: list[dict[str, Any]] = []

    for environment in sorted({row["environment"] for row in eligible}):
        idx, raw = _sample_dataset_states(environment, data_dir, N_DATASET_STATES)
        dataset_cache[environment] = raw
        np.savez_compressed(
            ds_dir / f"{environment}.npz",
            indices=idx,
            raw_observations=raw,
            sampling_seed=np.asarray(DATASET_SAMPLE_SEED),
            source=np.asarray("dataset_transitions"),
        )

    for pkey, methods in sorted(by_pair.items()):
        mart_row = methods.get("bar_p4")
        two_row = methods.get("two_actor_p4")
        environment = (mart_row or two_row)["environment"]
        tau = int((mart_row or two_row)["T"])
        seed = int((mart_row or two_row)["seed"])
        ds_raw = dataset_cache[environment]
        mart_pack = two_pack = None
        if mart_row is not None:
            mart_pack = _action_pack_for_ckpt(
                load_payload(Path(mart_row["checkpoint"])),
                ACTORS_BY_CONDITION["bar_p4"],
                ds_raw,
                tau,
            )
            for item in mart_pack["hop_stats"]:
                hop_rows.append(
                    {"state_set": "dataset", "run_key": mart_row["run_key"], "environment": environment, "T": tau, "seed": seed, **item}
                )
        if two_row is not None:
            two_pack = _action_pack_for_ckpt(
                load_payload(Path(two_row["checkpoint"])),
                ACTORS_BY_CONDITION["two_actor_p4"],
                ds_raw,
                tau,
            )
            for item in two_pack["hop_stats"]:
                hop_rows.append(
                    {"state_set": "dataset", "run_key": two_row["run_key"], "environment": environment, "T": tau, "seed": seed, **item}
                )
        distance_rows.extend(
            _distance_rows(
                state_set="dataset",
                environment=environment,
                tau=tau,
                seed=seed,
                n_states=int(ds_raw.shape[0]),
                mart=mart_pack,
                two=two_pack,
            )
        )

        rollout_raw = _load_pair_rollout_states(out_dir, mart_row, two_row)
        if rollout_raw is not None:
            mart_r = two_r = None
            if mart_row is not None:
                mart_r = _action_pack_for_ckpt(
                    load_payload(Path(mart_row["checkpoint"])),
                    ACTORS_BY_CONDITION["bar_p4"],
                    rollout_raw,
                    tau,
                )
                for item in mart_r["hop_stats"]:
                    hop_rows.append(
                        {"state_set": "rollout_common", "run_key": mart_row["run_key"], "environment": environment, "T": tau, "seed": seed, **item}
                    )
            if two_row is not None:
                two_r = _action_pack_for_ckpt(
                    load_payload(Path(two_row["checkpoint"])),
                    ACTORS_BY_CONDITION["two_actor_p4"],
                    rollout_raw,
                    tau,
                )
                for item in two_r["hop_stats"]:
                    hop_rows.append(
                        {"state_set": "rollout_common", "run_key": two_row["run_key"], "environment": environment, "T": tau, "seed": seed, **item}
                    )
            distance_rows.extend(
                _distance_rows(
                    state_set="rollout_common",
                    environment=environment,
                    tau=tau,
                    seed=seed,
                    n_states=int(rollout_raw.shape[0]),
                    mart=mart_r,
                    two=two_r,
                )
            )
        print(f"[actions] {pkey} dataset={ds_raw.shape[0]} rollout={None if rollout_raw is None else rollout_raw.shape[0]}", flush=True)

    write_csv(
        out_dir / "action_distances.csv",
        distance_rows,
        ["state_set", "environment", "T", "seed", "n_states", "metric", "mean", "std", "n_finite", "n_near_zero_excluded"],
    )
    write_csv(
        out_dir / "hop_q_scale.csv",
        hop_rows,
        ["state_set", "run_key", "environment", "T", "seed", "actor_id", "tau_hop", "q_scale", "lambda", "action_grad_norm_mean", "q_at_pi_mean", "bound_fraction", "q_scale_formula"],
    )


def _load_pair_rollout_states(
    out_dir: Path,
    mart_row: dict[str, Any] | None,
    two_row: dict[str, Any] | None,
) -> np.ndarray | None:
    chunks = []
    raw_dir = out_dir / "raw" / "rollout_states"
    for row, actor_id in (
        (mart_row, "mart_mu4"),
        (two_row, "two_actor_deploy"),
    ):
        if row is None:
            continue
        path = raw_dir / f"{episode_part_path(out_dir, row['run_key'], actor_id).stem}.npz"
        if not path.is_file():
            return None
        payload = np.load(path, allow_pickle=False)
        obs = np.asarray(payload["raw_observations"], dtype=np.float32)
        take = min(N_ROLLOUT_PER_POLICY, obs.shape[0])
        chunks.append(obs[:take])
    if len(chunks) != 2:
        return None
    return np.concatenate(chunks, axis=0)


def _dataset_actions(env_name: str, data_dir: Path, indices: np.ndarray) -> np.ndarray:
    raw = qlearning_from_hdf5(dataset_path(env_name, data_dir))
    return np.asarray(raw["actions"], dtype=np.float32)[np.asarray(indices, dtype=np.int64)]


def run_objectives(inventory: Mapping[str, Any], out_dir: Path, data_dir: Path) -> None:
    from scripts.diagnostics.p0_survival import delta_loss_terms

    eligible = [row for row in inventory["runs"] if row.get("eval_eligible")]
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in eligible:
        by_pair.setdefault(pair_key(row["environment"], row["T"], row["seed"]), {})[row["condition"]] = row
    ds_dir = out_dir / "raw" / "dataset_states"
    rows_out: list[dict[str, Any]] = []

    def pack_and_q(ckpt: Path, specs, raw_states: np.ndarray, tau: float):
        payload = load_payload(ckpt)
        pack = _action_pack_for_ckpt(payload, specs, raw_states, tau)
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        states = _normalize(raw_states, mean, std)
        critic = TwinCritic()
        q_by_id = {}
        for spec in specs:
            q, _ = _q1_and_grad(critic, payload["critic_params"], states, pack["actions"][spec.actor_id])
            q_by_id[spec.actor_id] = q
        return payload, pack, q_by_id, states

    for pkey, methods in sorted(by_pair.items()):
        mart_row = methods.get("bar_p4")
        two_row = methods.get("two_actor_p4")
        environment = (mart_row or two_row)["environment"]
        tau = int((mart_row or two_row)["T"])
        seed = int((mart_row or two_row)["seed"])
        ds_path = ds_dir / f"{environment}.npz"
        if not ds_path.is_file():
            print(f"[objectives] missing dataset states {environment}", flush=True)
            continue
        ds = np.load(ds_path, allow_pickle=False)
        ds_raw = np.asarray(ds["raw_observations"], dtype=np.float32)
        ds_act = _dataset_actions(environment, data_dir, ds["indices"])
        rollout_raw = _load_pair_rollout_states(out_dir, mart_row, two_row)
        state_sets = [("dataset", ds_raw, ds_act)]
        if rollout_raw is not None:
            state_sets.append(("rollout_common", rollout_raw, None))

        for state_set, raw_states, dataset_actions in state_sets:
            packs = {}
            qs = {}
            if mart_row is not None:
                _, packs["bar_p4"], qs["bar_p4"], _ = pack_and_q(
                    Path(mart_row["checkpoint"]), ACTORS_BY_CONDITION["bar_p4"], raw_states, tau
                )
            if two_row is not None:
                _, packs["two_actor_p4"], qs["two_actor_p4"], _ = pack_and_q(
                    Path(two_row["checkpoint"]), ACTORS_BY_CONDITION["two_actor_p4"], raw_states, tau
                )
            hop_specs = []
            if "bar_p4" in packs:
                hop_specs.extend(
                    [
                        ("bar_p4", 1, "dataset_action", "mart_mu1", mart_row),
                        ("bar_p4", 2, "mart_mu1", "mart_mu2", mart_row),
                        ("bar_p4", 3, "mart_mu2", "mart_mu3", mart_row),
                        ("bar_p4", 4, "mart_mu3", "mart_mu4", mart_row),
                    ]
                )
            if "two_actor_p4" in packs:
                hop_specs.extend(
                    [
                        ("two_actor_p4", 1, "dataset_action", "two_actor_target", two_row),
                        ("two_actor_p4", 1, "dataset_action", "two_actor_deploy", two_row),
                    ]
                )
            for condition, hop_index, ref_id, new_id, run in hop_specs:
                pack = packs[condition]
                if new_id not in pack["actions"]:
                    continue
                new_a = pack["actions"][new_id]
                if ref_id == "dataset_action":
                    if dataset_actions is None:
                        continue
                    ref_a = dataset_actions
                    critic = TwinCritic()
                    payload = load_payload(Path(run["checkpoint"]))
                    mean = np.asarray(payload["mean"], dtype=np.float32)
                    std = np.asarray(payload["std"], dtype=np.float32)
                    states = _normalize(raw_states, mean, std)
                    q_ref, _ = _q1_and_grad(critic, payload["critic_params"], states, ref_a)
                else:
                    ref_a = pack["actions"][ref_id]
                    q_ref = qs[condition][ref_id]
                q_new = qs[condition][new_id]
                c_k = float(next(s["lambda"] for s in pack["hop_stats"] if s["actor_id"] == new_id))
                terms = delta_loss_terms(q_new, q_ref, new_a, ref_a, c_k)
                rows_out.append(
                    {
                        "state_set": state_set,
                        "environment": environment,
                        "T": tau,
                        "seed": seed,
                        "condition": condition,
                        "run_key": run["run_key"],
                        "hop_index": hop_index,
                        "ref_actor": ref_id,
                        "new_actor": new_id,
                        "n_states": int(raw_states.shape[0]),
                        "optimizer_diverged": bool(run.get("optimizer_diverged")),
                        **terms,
                    }
                )
        print(f"[objectives] {pkey}", flush=True)

    write_csv(
        out_dir / "hop_objectives.csv",
        rows_out,
        [
            "state_set",
            "environment",
            "T",
            "seed",
            "condition",
            "run_key",
            "hop_index",
            "ref_actor",
            "new_actor",
            "n_states",
            "optimizer_diverged",
            "c_k",
            "mean_q_new",
            "mean_q_ref",
            "mean_q_gain",
            "transport_mean_sq",
            "delta_L",
            "q_term",
        ],
    )
    print(f"[objectives] wrote {len(rows_out)} rows", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "eval", "actions", "objectives", "all"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoints-json", type=Path, default=DEFAULT_TAIL_CHECKPOINTS)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_TAIL_MANIFEST)
    parser.add_argument("--p0-root", type=Path, default=DEFAULT_P0_ROOT)
    parser.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="Evaluate the first eligible run only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    gitignore = out_dir / "raw" / ".gitignore"
    if not gitignore.is_file():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    inventory_path = out_dir / "INVENTORY.json"
    if args.command in ("inventory", "all") or not inventory_path.is_file():
        print("[inventory] scanning frozen P0 tail90 + planned 180-run grid", flush=True)
        inventory = build_inventory(
            checkpoints_path=args.checkpoints_json,
            manifest_path=args.manifest_json,
            data_dir=args.data_dir,
        )
        write_json(inventory_path, inventory)
        write_csv(
            out_dir / "inventory_runs.csv",
            inventory["runs"],
            [
                "run_key",
                "condition",
                "method_label",
                "environment",
                "T",
                "seed",
                "eval_eligible",
                "missing_reason",
                "checkpoint",
                "file_sha256",
                "step",
                "n_actors_in_ckpt",
                "config_method",
                "config_tau",
                "config_normalize",
                "config_q_scale_norm",
                "eval_env",
                "optimizer_diverged",
                "frozen_source_revision",
            ],
        )
        write_json(out_dir / "COVERAGE.json", inventory["coverage"])
        print("[inventory] coverage", json.dumps(inventory["coverage"], sort_keys=True), flush=True)
    else:
        inventory = load_json(inventory_path)

    if args.command in ("eval", "all"):
        run_eval(inventory, out_dir, workers=args.workers, pilot=args.pilot)
    if args.command in ("actions", "all"):
        run_actions(inventory, out_dir, args.data_dir)
    if args.command in ("objectives", "all"):
        run_objectives(inventory, out_dir, args.data_dir)
    write_json(
        out_dir / "STATUS.json",
        {
            "command": args.command,
            "pilot": args.pilot,
            "out_dir": str(out_dir),
            "python": sys.executable,
            "coverage": inventory.get("coverage"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
