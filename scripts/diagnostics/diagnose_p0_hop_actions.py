#!/usr/bin/env python3
"""Common-state action diagnostics for P0 hop actors. No training."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EIGEN_NUM_THREADS", "1")
_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
_platforms = os.environ.get("JAX_PLATFORMS", "").lower()
if _platforms != "cuda" and _cuda in (None, ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )

from _lab_import import ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _p0_hop_common import (  # noqa: E402
    ARCHIVE,
    DATA_DIR,
    DATASET_N,
    DATASET_SAMPLE_SEED,
    ENVIRONMENTS,
    NEAR_BOUND,
    NEAR_ZERO_RMS,
    actor_roles,
    now_kst,
)
from train_td3bc import (  # noqa: E402
    Actor,
    TwinCritic,
    load_checkpoint,
    normalized_q_weight,
    qlearning_from_hdf5,
    download_dataset,
)

EPS = 1e-8


def _infer_action_dim(params: Any, mean: np.ndarray) -> int:
    kernel = np.asarray(params["params"]["Dense_0"]["kernel"])
    output_kernel = np.asarray(params["params"]["Dense_2"]["kernel"])
    if kernel.shape[0] != mean.shape[0]:
        raise ValueError("actor input dim does not match checkpoint mean")
    return int(output_kernel.shape[-1])


def _actor_params(payload: dict[str, Any], actor_index: int):
    return payload["actors_params"][actor_index - 1]


def _rms(delta: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(delta), axis=-1))


def _cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + EPS
    return np.sum(x * y, axis=-1) / denom


def dataset_raw(env_name: str) -> np.ndarray:
    cache = ARCHIVE / "raw" / "dataset_states" / f"{env_name}.npz"
    if cache.is_file():
        return np.load(cache)["raw_observations"]
    raw = qlearning_from_hdf5(download_dataset(env_name, DATA_DIR))["observations"]
    rng = np.random.default_rng(DATASET_SAMPLE_SEED)
    index = rng.choice(raw.shape[0], size=min(DATASET_N, raw.shape[0]), replace=False)
    index.sort()
    states = np.asarray(raw[index], dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        raw_observations=states,
        sampling_seed=np.asarray(DATASET_SAMPLE_SEED),
        source=np.asarray("d4rl_qlearning_observations"),
        n=np.asarray(states.shape[0]),
    )
    return states


def rollout_union(environment: str, tau: int, seed: int) -> np.ndarray | None:
    raw_dir = ARCHIVE / "raw" / "rollout_states"
    parts = []
    for method, role in (("MART4", "mu4"), ("two_actor_p4", "deployment")):
        path = raw_dir / f"{method}_{environment}_T{tau}_s{seed}_{role}.npz"
        if path.is_file():
            parts.append(np.load(path)["raw_observations"])
    if len(parts) != 2:
        return None
    return np.concatenate(parts, axis=0)


def load_policies(record: dict[str, Any]):
    payload = load_checkpoint(Path(record["checkpoint_path"]))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    first = _actor_params(payload, 1)
    actor = Actor(action_dim=_infer_action_dim(first, mean), max_action=max_action)
    apply = jax.jit(actor.apply)
    actions = []
    for spec in actor_roles(record["condition"]):
        params = _actor_params(payload, spec["actor_index"])
        actions.append((spec, params, apply))
    critic = TwinCritic()
    q_apply = jax.jit(critic.apply)
    return payload, mean, std, max_action, actions, q_apply


def act_on_raw(apply, params, raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normed = (raw - mean) / std
    return np.asarray(apply(params, jnp.asarray(normed, dtype=jnp.float32)), dtype=np.float32)


def hop_tau(record: dict[str, Any], spec: dict[str, Any]) -> float | None:
    cfg = record.get("checkpoint") or {}
    total = float(record["T"])
    if record["condition"] == "bar_p4":
        k = int(cfg.get("mpi_steps") or 4)
        return total / k
    if spec["actor_role"] == "target_value":
        k = int(cfg.get("mpi_steps") or cfg.get("reference_depth") or 4)
        return total / k
    if spec["actor_role"] == "deployment":
        return total
    return None


def summarize_actions(
    record: dict[str, Any],
    state_set: str,
    raw: np.ndarray,
    payload: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    max_action: float,
    packed,
    q_apply,
    q_scale_norm: bool | None,
) -> list[dict[str, Any]]:
    named = {}
    for spec, params, apply in packed:
        named[spec["actor_role"]] = act_on_raw(apply, params, raw, mean, std)
    rows = []
    roles = [spec["actor_role"] for spec, _, _ in packed]
    actions = [named[role] for role in roles]
    for index, role in enumerate(roles):
        act = actions[index]
        bound_frac = float(np.mean(np.max(np.abs(act), axis=-1) >= NEAR_BOUND * max_action))
        tau_h = hop_tau(record, packed[index][0])
        q_w = None
        grad_norm = None
        ref_act = actions[index - 1] if index > 0 and record["condition"] == "bar_p4" else act
        if q_scale_norm is not None and tau_h is not None and "critic_params" in payload:
            q_at = ref_act if packed[index][0]["objective"] == "jko_recenter" else act
            q1, _ = q_apply(
                payload["critic_params"],
                jnp.asarray((raw - mean) / std, dtype=jnp.float32),
                jnp.asarray(q_at, dtype=jnp.float32),
            )
            q1 = jnp.squeeze(q1, -1)
            q_w = float(normalized_q_weight(q1, tau_h, scale_norm=bool(q_scale_norm)))

            def q_sum(a):
                q, _ = q_apply(
                    payload["critic_params"],
                    jnp.asarray((raw - mean) / std, dtype=jnp.float32),
                    a,
                )
                return jnp.sum(jnp.squeeze(q, -1))

            grads = np.asarray(jax.grad(q_sum)(jnp.asarray(act, dtype=jnp.float32)))
            grad_norm = float(np.mean(np.linalg.norm(grads, axis=-1)))
        rows.append(
            {
                "task": record["environment"],
                "T": record["T"],
                "training_seed": record["seed"],
                "method": record["method"],
                "state_set": state_set,
                "metric": "bound_frac",
                "actor_a": role,
                "actor_b": "",
                "value": bound_frac,
                "n": int(raw.shape[0]),
                "n_excluded": 0,
                "q_weight": q_w,
                "action_grad_norm": grad_norm,
                "paired": bool(record.get("paired")),
                "checkpoint_hash": record.get("checkpoint_hash"),
            }
        )
    if record["condition"] == "bar_p4" and all(f"mu{k}" in named for k in (1, 2, 3, 4)):
        hops = [("mu1", "mu2"), ("mu2", "mu3"), ("mu3", "mu4")]
        deltas = []
        for a, b in hops:
            delta = named[b] - named[a]
            rms = _rms(delta)
            rows.append(
                {
                    "task": record["environment"],
                    "T": record["T"],
                    "training_seed": record["seed"],
                    "method": record["method"],
                    "state_set": state_set,
                    "metric": "adjacent_rms",
                    "actor_a": a,
                    "actor_b": b,
                    "value": float(np.mean(rms)),
                    "n": int(raw.shape[0]),
                    "n_excluded": 0,
                    "q_weight": None,
                    "action_grad_norm": None,
                    "paired": bool(record.get("paired")),
                    "checkpoint_hash": record.get("checkpoint_hash"),
                }
            )
            deltas.append(delta)
        for later in ("mu2", "mu3", "mu4"):
            rms = _rms(named[later] - named["mu1"])
            rows.append(
                {
                    "task": record["environment"],
                    "T": record["T"],
                    "training_seed": record["seed"],
                    "method": record["method"],
                    "state_set": state_set,
                    "metric": "from_mu1_rms",
                    "actor_a": "mu1",
                    "actor_b": later,
                    "value": float(np.mean(rms)),
                    "n": int(raw.shape[0]),
                    "n_excluded": 0,
                    "q_weight": None,
                    "action_grad_norm": None,
                    "paired": bool(record.get("paired")),
                    "checkpoint_hash": record.get("checkpoint_hash"),
                }
            )
        for left, right, name in (
            (deltas[0], deltas[1], "cos_d2_d3"),
            (deltas[1], deltas[2], "cos_d3_d4"),
            (deltas[0], deltas[2], "cos_d2_d4"),
        ):
            rms_l = _rms(left)
            rms_r = _rms(right)
            keep = (rms_l >= NEAR_ZERO_RMS) & (rms_r >= NEAR_ZERO_RMS)
            n_ex = int((~keep).sum())
            if keep.any():
                value = float(np.mean(_cosine(left[keep], right[keep])))
            else:
                value = float("nan")
            rows.append(
                {
                    "task": record["environment"],
                    "T": record["T"],
                    "training_seed": record["seed"],
                    "method": record["method"],
                    "state_set": state_set,
                    "metric": name,
                    "actor_a": "delta",
                    "actor_b": "delta",
                    "value": value,
                    "n": int(keep.sum()),
                    "n_excluded": n_ex,
                    "q_weight": None,
                    "action_grad_norm": None,
                    "paired": bool(record.get("paired")),
                    "checkpoint_hash": record.get("checkpoint_hash"),
                }
            )
    return rows, named


def process_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    q_scale = record.get("checkpoint", {}).get("q_scale_norm")
    payload, mean, std, max_action, packed, q_apply = load_policies(record)
    rows, _ = summarize_actions(
        record,
        "dataset",
        dataset_raw(record["environment"]),
        payload,
        mean,
        std,
        max_action,
        packed,
        q_apply,
        q_scale,
    )
    raw_b = rollout_union(record["environment"], record["T"], record["seed"])
    if raw_b is not None:
        extra, _ = summarize_actions(
            record,
            "rollout_union",
            raw_b,
            payload,
            mean,
            std,
            max_action,
            packed,
            q_apply,
            q_scale,
        )
        rows.extend(extra)
    return rows


def _worker(worker_id: int, records: list[dict[str, Any]], out_pkl: str, cpu0: int) -> None:
    try:
        os.sched_setaffinity(0, {cpu0 + worker_id})
    except OSError:
        pass
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(process_record(record))
        print(
            f"w{worker_id} {record['method']} {record['environment']} "
            f"T={record['T']} seed={record['seed']}",
            flush=True,
        )
    Path(out_pkl).write_bytes(pickle.dumps(rows))


def paired_distance_rows(inventory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for record in inventory_rows:
        if record["status"] != "evalable":
            continue
        key = (record["environment"], record["T"], record["seed"])
        by_key.setdefault(key, {})[record["method"]] = record
    rows = []
    for key, methods in by_key.items():
        if "MART4" not in methods or "two_actor_p4" not in methods:
            continue
        mart = methods["MART4"]
        control = methods["two_actor_p4"]
        _, m_mean, m_std, _, m_packed, _ = load_policies(mart)
        _, c_mean, c_std, _, c_packed, _ = load_policies(control)
        raw_b = rollout_union(key[0], key[1], key[2])
        for state_set, raw in (
            ("dataset", dataset_raw(key[0])),
            ("rollout_union", raw_b),
        ):
            if raw is None:
                continue
            mu4 = act_on_raw(m_packed[-1][2], m_packed[-1][1], raw, m_mean, m_std)
            deploy = act_on_raw(c_packed[-1][2], c_packed[-1][1], raw, c_mean, c_std)
            rows.append(
                {
                    "task": key[0],
                    "T": key[1],
                    "training_seed": key[2],
                    "method": "paired",
                    "state_set": state_set,
                    "metric": "mu4_vs_deploy_rms",
                    "actor_a": "mu4",
                    "actor_b": "deployment",
                    "value": float(np.mean(_rms(mu4 - deploy))),
                    "n": int(raw.shape[0]),
                    "n_excluded": 0,
                    "q_weight": None,
                    "action_grad_norm": None,
                    "paired": True,
                    "checkpoint_hash": (
                        f"{mart.get('checkpoint_hash')}|{control.get('checkpoint_hash')}"
                    ),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu0", type=int, default=96)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"refusing non-CPU JAX backend {jax.default_backend()!r}")
    inventory = json.loads((ARCHIVE / "INVENTORY.json").read_text(encoding="utf-8"))
    records = [row for row in inventory["rows"] if row["status"] == "evalable"]
    for env in sorted({row["environment"] for row in records}):
        dataset_raw(env)
    workers = max(1, min(int(args.workers), len(records) or 1))
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "n_records": len(records),
                "workers": workers,
                "cpu0": args.cpu0,
                "started_at": now_kst(),
            },
            indent=2,
        ),
        flush=True,
    )
    all_rows: list[dict[str, Any]] = []
    if workers == 1:
        for record in records:
            all_rows.extend(process_record(record))
    else:
        shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
        for index, record in enumerate(records):
            shards[index % workers].append(record)
        shard_dir = ARCHIVE / "raw" / "action_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        ctx = mp.get_context("spawn")
        procs = []
        paths = []
        for worker_id, shard in enumerate(shards):
            if not shard:
                continue
            path = shard_dir / f"w{worker_id}.pkl"
            paths.append(path)
            proc = ctx.Process(
                target=_worker,
                args=(worker_id, shard, str(path), args.cpu0),
            )
            proc.start()
            procs.append(proc)
        failures = 0
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                failures += 1
        if failures:
            raise RuntimeError(f"{failures} action-diagnostic workers failed")
        for path in paths:
            all_rows.extend(pickle.loads(path.read_bytes()))
    all_rows.extend(paired_distance_rows(inventory["rows"]))

    out = ARCHIVE / "action_diagnostics.csv"
    fields = [
        "task",
        "T",
        "training_seed",
        "method",
        "state_set",
        "metric",
        "actor_a",
        "actor_b",
        "value",
        "n",
        "n_excluded",
        "q_weight",
        "action_grad_norm",
        "paired",
        "checkpoint_hash",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (ARCHIVE / "ACTION_DIAGNOSTICS_RECEIPT.json").write_text(
        json.dumps(
            {
                "built_at": now_kst(),
                "n_rows": len(all_rows),
                "dataset_n": DATASET_N,
                "dataset_sample_seed": DATASET_SAMPLE_SEED,
                "near_zero_rms": NEAR_ZERO_RMS,
                "workers": 4,
                "cpu_pin": "96-99",
                "note": (
                    "Distances are on shared raw states with per-checkpoint "
                    "normalization. Do not treat per-policy visit statistics as "
                    "cross-policy action distances."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} rows={len(all_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
