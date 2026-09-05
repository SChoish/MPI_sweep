#!/usr/bin/env python3
"""Frozen-critic JKO hop objective on P0 MART checkpoints. No training."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EIGEN_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _p0_hop_common import ARCHIVE, now_kst  # noqa: E402
from _p0_hop_metrics import classify_obj_vs_return, hop_objective_delta  # noqa: E402
from diagnose_p0_hop_actions import (  # noqa: E402
    act_on_raw,
    dataset_raw,
    hop_tau,
    load_policies,
)

HOPS = (("mu1", "mu2", 2), ("mu2", "mu3", 3), ("mu3", "mu4", 4))
DEFAULT_TASKS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
)


def q1_on_actions(q_apply, critic_params, raw, mean, std, actions) -> np.ndarray:
    q1, _ = q_apply(
        critic_params,
        jnp.asarray((raw - mean) / std, dtype=jnp.float32),
        jnp.asarray(actions, dtype=jnp.float32),
    )
    return np.asarray(jnp.squeeze(q1, -1), dtype=np.float64)


def process_record(record: dict[str, Any], cell_means: list[dict[str, str]]) -> list[dict]:
    payload, mean, std, _max_action, packed, q_apply = load_policies(record)
    raw = dataset_raw(record["environment"])
    named = {}
    q_named = {}
    for spec, params, apply in packed:
        act = act_on_raw(apply, params, raw, mean, std)
        named[spec["actor_role"]] = act
        q_named[spec["actor_role"]] = q1_on_actions(
            q_apply, payload["critic_params"], raw, mean, std, act
        )
    rows = []
    q_scale = bool(record.get("checkpoint", {}).get("q_scale_norm", True))
    returns = {
        (row["method"], row["actor_role"], int(row["training_seed"])): float(
            row["mean_normalized"]
        )
        for row in cell_means
        if row["task"] == record["environment"]
        and int(row["T"]) == int(record["T"])
        and int(row["training_seed"]) == int(record["seed"])
    }
    for prev_role, role, hop in HOPS:
        spec = next(s for s, _, _ in packed if s["actor_role"] == role)
        tau_step = hop_tau(record, spec)
        terms = hop_objective_delta(
            q_named[role],
            q_named[prev_role],
            named[role],
            named[prev_role],
            tau_step=float(tau_step),
            scale_norm=q_scale,
        )
        j_k = returns.get(("MART4", role, int(record["seed"])))
        j_prev = returns.get(("MART4", prev_role, int(record["seed"])))
        delta_j = None if j_k is None or j_prev is None else j_k - j_prev
        row = {
            "task": record["environment"],
            "T": record["T"],
            "training_seed": record["seed"],
            "hop": hop,
            "actor_a": prev_role,
            "actor_b": role,
            "state_set": "dataset",
            "n": int(raw.shape[0]),
            "tau_step": float(tau_step),
            "scale_norm": q_scale,
            "checkpoint_hash": record.get("checkpoint_hash"),
            "delta_normalized": delta_j,
            "obj_vs_return": (
                None
                if delta_j is None
                else classify_obj_vs_return(terms["delta_L"], delta_j)
            ),
            **terms,
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[10])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"refusing non-CPU JAX backend {jax.default_backend()!r}")
    inventory = json.loads((ARCHIVE / "INVENTORY.json").read_text(encoding="utf-8"))
    cell_means = []
    means_path = ARCHIVE / "cell_means.csv"
    if means_path.is_file():
        import csv as _csv

        with means_path.open(newline="", encoding="utf-8") as handle:
            cell_means = list(_csv.DictReader(handle))
    records = [
        row
        for row in inventory["rows"]
        if row["status"] == "evalable"
        and row["condition"] == "bar_p4"
        and row["environment"] in set(args.tasks)
        and int(row["T"]) in set(args.T)
        and int(row["seed"]) in set(args.seeds)
    ]
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "n_records": len(records),
                "tasks": args.tasks,
                "T": args.T,
                "started_at": now_kst(),
            },
            indent=2,
        ),
        flush=True,
    )
    all_rows: list[dict[str, Any]] = []
    for record in records:
        rows = process_record(record, cell_means)
        all_rows.extend(rows)
        print(
            f"{record['environment']} T={record['T']} seed={record['seed']} "
            + " ".join(
                f"h{r['hop']} dL={r['delta_L']:+.4g} dQ={r['delta_q']:+.4g}"
                for r in rows
            ),
            flush=True,
        )
    out = ARCHIVE / "hop_objective.csv"
    fields = [
        "task",
        "T",
        "training_seed",
        "hop",
        "actor_a",
        "actor_b",
        "state_set",
        "n",
        "tau_step",
        "scale_norm",
        "q_weight",
        "mean_q_prev",
        "mean_q_k",
        "delta_q",
        "w2",
        "loss_stay",
        "loss_k",
        "delta_L",
        "objective_improved",
        "delta_normalized",
        "obj_vs_return",
        "checkpoint_hash",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    receipt = {
        "built_at": now_kst(),
        "n_rows": len(all_rows),
        "formula": (
            "delta_L = -c_k * mean(Q(mu_k)-Q(mu_{k-1})) + mean((mu_k-mu_{k-1})^2); "
            "c_k = 2*(T/K) / mean(|Q(mu_{k-1})|); Q is critic head 1"
        ),
        "state_set": "dataset sample from diagnose_p0_hop_actions",
        "note": (
            "Negative delta_L is an improvement of the hop objective. "
            "Q improvement is not treated as policy improvement. The critic "
            "is trained on actor-1 continuation."
        ),
    }
    (ARCHIVE / "HOP_OBJECTIVE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out} rows={len(all_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
