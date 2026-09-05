#!/usr/bin/env python3
"""Frozen-critic extra JKO steps on Hopper-medium/expert hop actors.

Understood as: on existing 1M MART snapshots, freeze the critic and μ_{k-1},
run additional Adam steps on μ_k only, and test whether snapshot ΔL>0 shrinks.
Not a new training grid. Not a shared-driver run. Device comes from the caller.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EIGEN_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from _lab_import import REPO_ROOT as ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from _p0_hop_common import ARCHIVE, DATA_DIR, now_kst  # noqa: E402
from _p0_hop_metrics import hop_objective_delta  # noqa: E402
from diagnose_p0_hop_actions import (  # noqa: E402
    act_on_raw,
    dataset_raw,
    hop_tau,
    load_policies,
)
from diagnose_p0_hop_objective import q1_on_actions  # noqa: E402
from train_td3bc import (  # noqa: E402
    Actor,
    download_dataset,
    normalized_q_weight,
    qlearning_from_hdf5,
)

OUT = ROOT / "sweep_results/diagnostics/p0_frozen_hop_extra_opt"
SOURCE_CKPTS = OUT / "source_ckpts"
TASKS = ("hopper-medium-v2", "hopper-expert-v2")
HOPS = (("mu1", "mu2", 2), ("mu2", "mu3", 3), ("mu3", "mu4", 4))
DEFAULT_LOG_STEPS = (0, 500, 2000, 10000)
_OBS_CACHE: dict[tuple[str, bytes, bytes], np.ndarray] = {}


def _tree_l2(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return float(jnp.sqrt(total))


def full_dataset_normed(env_name: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    key = (env_name, mean.tobytes(), std.tobytes())
    cached = _OBS_CACHE.get(key)
    if cached is not None:
        return cached
    raw = qlearning_from_hdf5(download_dataset(env_name, DATA_DIR))["observations"]
    out = np.asarray((raw - mean) / std, dtype=np.float32)
    _OBS_CACHE[key] = out
    return out


def resolve_checkpoint(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    name = Path(row.get("host_run_dir") or row.get("checkpoint_path") or "").name
    if name == "params_1000000.pkl":
        name = Path(row.get("host_run_dir") or "").name
    staged = SOURCE_CKPTS / name / "params_1000000.pkl"
    if staged.is_file():
        row["checkpoint_path"] = str(staged)
    path = Path(row.get("checkpoint_path") or "")
    if not path.is_file():
        raise FileNotFoundError(f"missing 1M checkpoint for {row.get('key')}: {path}")
    return row


def make_actor_state(params, lr: float) -> TrainState:
    return TrainState.create(
        apply_fn=lambda p, x: None,
        params=params,
        tx=optax.adam(lr),
    )


def make_update(
    apply,
    critic_apply,
    critic_params,
    ref_params,
    tau_step: float,
    scale_norm: bool,
    data_obs: jax.Array,
    batch_size: int,
):
    critic = SimpleNamespace(apply_fn=critic_apply, params=critic_params)

    @jax.jit
    def block(actor: TrainState, rng: jax.Array, n_steps: int) -> tuple[TrainState, jax.Array]:
        def body(_i, carry):
            actor, rng = carry
            rng, key = jax.random.split(rng)
            idx = jax.random.randint(key, (batch_size,), 0, data_obs.shape[0])
            obs = data_obs[idx]
            ref = jax.lax.stop_gradient(apply(ref_params, obs))
            q_ref, _ = critic.apply_fn(critic.params, obs, ref)
            q_weight = normalized_q_weight(q_ref, tau_step, scale_norm=scale_norm)

            def loss_fn(params):
                pi = apply(params, obs)
                q1, _ = critic.apply_fn(critic.params, obs, pi)
                return -q_weight * jnp.mean(q1) + jnp.mean(jnp.square(pi - ref))

            loss, grads = jax.value_and_grad(loss_fn)(actor.params)
            actor = actor.apply_gradients(grads=grads)
            return actor, rng

        actor, rng = jax.lax.fori_loop(0, n_steps, body, (actor, rng))
        return actor, rng

    return block


def eval_hop(
    apply,
    q_apply,
    critic_params,
    hop_params,
    ref_params,
    raw_eval: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    tau_step: float,
    scale_norm: bool,
) -> dict[str, float]:
    act_k = act_on_raw(apply, hop_params, raw_eval, mean, std)
    act_ref = act_on_raw(apply, ref_params, raw_eval, mean, std)
    q_k = q1_on_actions(q_apply, critic_params, raw_eval, mean, std, act_k)
    q_ref = q1_on_actions(q_apply, critic_params, raw_eval, mean, std, act_ref)
    terms = hop_objective_delta(
        q_k, q_ref, act_k, act_ref, tau_step=tau_step, scale_norm=scale_norm
    )
    normed = (raw_eval - mean) / std
    ref_a = apply(ref_params, jnp.asarray(normed, dtype=jnp.float32))
    q_ref_j, _ = q_apply(
        critic_params,
        jnp.asarray(normed, dtype=jnp.float32),
        ref_a,
    )
    q_weight = normalized_q_weight(jnp.squeeze(q_ref_j, -1), tau_step, scale_norm=scale_norm)

    def loss_fn(params):
        pi = apply(params, jnp.asarray(normed, dtype=jnp.float32))
        q1, _ = q_apply(
            critic_params,
            jnp.asarray(normed, dtype=jnp.float32),
            pi,
        )
        return -q_weight * jnp.mean(q1) + jnp.mean(jnp.square(pi - jax.lax.stop_gradient(ref_a)))

    grads = jax.grad(loss_fn)(hop_params)
    terms["grad_l2"] = _tree_l2(grads)
    terms["action_rms_from_snapshot"] = float(
        np.sqrt(np.mean(np.square(act_k - act_on_raw(apply, hop_params, raw_eval, mean, std))))
    )
    return terms


def snapshot_action_rms(apply, params_a, params_b, raw, mean, std) -> float:
    a = act_on_raw(apply, params_a, raw, mean, std)
    b = act_on_raw(apply, params_b, raw, mean, std)
    return float(np.sqrt(np.mean(np.square(a - b))))


def process_hop(
    record: dict[str, Any],
    prev_role: str,
    role: str,
    hop: int,
    extra_steps: int,
    log_steps: tuple[int, ...],
    batch_size: int,
    lr: float,
    rng_seed: int,
) -> list[dict[str, Any]]:
    payload, mean, std, _max_a, packed, q_apply = load_policies(record)
    named_params = {spec["actor_role"]: params for spec, params, _apply in packed}
    apply = packed[0][2]
    spec = next(s for s, _, _ in packed if s["actor_role"] == role)
    tau_step = float(hop_tau(record, spec))
    scale_norm = bool(record.get("checkpoint", {}).get("q_scale_norm", True))
    raw_eval = dataset_raw(record["environment"])
    data_obs = jnp.asarray(full_dataset_normed(record["environment"], mean, std))
    hop_params0 = named_params[role]
    ref_params = named_params[prev_role]
    actor = make_actor_state(hop_params0, lr)
    block = make_update(
        apply,
        q_apply,
        payload["critic_params"],
        ref_params,
        tau_step,
        scale_norm,
        data_obs,
        batch_size,
    )
    rng = jax.random.PRNGKey(rng_seed)
    rows = []
    done = 0
    for target in log_steps:
        if target > extra_steps:
            break
        n = target - done
        if n > 0:
            actor, rng = block(actor, rng, n)
            done = target
        terms = eval_hop(
            apply,
            q_apply,
            payload["critic_params"],
            actor.params,
            ref_params,
            raw_eval,
            mean,
            std,
            tau_step,
            scale_norm,
        )
        terms["action_rms_from_snapshot"] = snapshot_action_rms(
            apply, hop_params0, actor.params, raw_eval, mean, std
        )
        rows.append(
            {
                "task": record["environment"],
                "T": record["T"],
                "training_seed": record["seed"],
                "hop": hop,
                "actor_a": prev_role,
                "actor_b": role,
                "extra_steps": target,
                "batch_size": batch_size,
                "lr": lr,
                "tau_step": tau_step,
                "n_eval": int(raw_eval.shape[0]),
                "checkpoint_hash": record.get("checkpoint_hash"),
                **{k: terms[k] for k in (
                    "q_weight",
                    "mean_q_k",
                    "mean_q_prev",
                    "delta_q",
                    "w2",
                    "delta_L",
                    "q_term",
                    "q_gain_move_ratio",
                    "objective_improved",
                    "grad_l2",
                    "action_rms_from_snapshot",
                )},
            }
        )
        print(
            f"{record['environment']} T={record['T']} seed={record['seed']} "
            f"{prev_role}->{role} step={target} dL={terms['delta_L']:+.4g} "
            f"ratio={terms['q_gain_move_ratio']:.3f} grad={terms['grad_l2']:.3g}",
            flush=True,
        )
    return rows


def plot_curves(rows: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        for seed in (0, 1):
            for hop, color in ((2, "#2c7fb8"), (3, "#fdae61"), (4, "#d73027")):
                sub = [
                    r
                    for r in rows
                    if r["task"] == task
                    and int(r["training_seed"]) == seed
                    and int(r["hop"]) == hop
                ]
                if not sub:
                    continue
                sub = sorted(sub, key=lambda r: int(r["extra_steps"]))
                ax.plot(
                    [int(r["extra_steps"]) for r in sub],
                    [float(r["delta_L"]) for r in sub],
                    marker="o",
                    color=color,
                    linestyle="-" if seed == 0 else "--",
                    linewidth=1.4,
                    label=f"hop{hop} seed{seed}",
                )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(task.replace("-v2", ""))
        ax.set_xlabel("extra Adam steps (frozen Q, frozen ref)")
        ax.set_ylabel("ΔL on diagnostic dataset states")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Hopper T=10: extra hop-actor steps vs frozen-critic ΔL")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_deltaL_vs_extra_steps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_report(rows: list[dict[str, Any]], extra_steps: int, figure: str) -> None:
    last = [r for r in rows if int(r["extra_steps"]) == extra_steps]
    start = [r for r in rows if int(r["extra_steps"]) == 0]
    lines = [
        "# Frozen-critic extra hop-actor optimization",
        "",
        f"Built: {now_kst()}",
        "",
        "Understood as: freeze the 1M critic and μ_{k-1} on Hopper-medium/expert",
        "T=10 MART snapshots, extra-optimize μ_k with the JKO loss, and test",
        "whether snapshot ΔL>0 shrinks. Not a return experiment.",
        "",
        "## Protocol",
        "",
        f"- Extra Adam steps: {extra_steps} at lr=3e-4, batch 256, fresh optimizer.",
        "- Critic and previous actor are not updated.",
        "- ΔL is measured on the same 4096 diagnostic dataset states as hop_objective.csv.",
        "- Optimization samples the full D4RL observations, checkpoint-normalized.",
        "- One JAX process per GPU when launched via launch_p0_frozen_hop_extra_opt.sh.",
        "",
        "## Snapshot vs extra-opt",
        "",
    ]
    by_task: dict[str, list] = {}
    for row in last:
        by_task.setdefault(row["task"], []).append(row)
    for task, group in by_task.items():
        n_start_pos = sum(
            1
            for r in start
            if r["task"] == task and float(r["delta_L"]) >= 0
        )
        n_end_neg = sum(1 for r in group if float(r["delta_L"]) < 0)
        mean_start = float(
            np.mean(
                [float(r["delta_L"]) for r in start if r["task"] == task]
            )
        )
        mean_end = float(np.mean([float(r["delta_L"]) for r in group]))
        mean_ratio_end = float(
            np.mean([float(r["q_gain_move_ratio"]) for r in group])
        )
        lines += [
            f"### {task.replace('-v2', '')}",
            "",
            f"- hops with ΔL≥0 at step 0: {n_start_pos}/{len(group)}.",
            f"- hops with ΔL<0 after {extra_steps} steps: {n_end_neg}/{len(group)}.",
            f"- mean ΔL {mean_start:+.4g} → {mean_end:+.4g}.",
            f"- mean Q-gain/move-cost after extra steps: {mean_ratio_end:.3f}.",
            "",
        ]
        for row in group:
            match = next(
                r
                for r in start
                if r["task"] == task
                and r["training_seed"] == row["training_seed"]
                and r["hop"] == row["hop"]
            )
            lines.append(
                f"- seed {row['training_seed']} {row['actor_a']}→{row['actor_b']}: "
                f"ΔL {float(match['delta_L']):+.4g} → {float(row['delta_L']):+.4g}, "
                f"ratio {float(match['q_gain_move_ratio']):.3f} → "
                f"{float(row['q_gain_move_ratio']):.3f}, "
                f"grad {float(row['grad_l2']):.3g}, "
                f"move from snapshot RMS {float(row['action_rms_from_snapshot']):.4f}."
            )
        lines.append("")
    lines += [
        "## Read",
        "",
        "If extra steps drive ΔL below 0, snapshot non-improvement was optimizer",
        "slack on this frozen objective. If ΔL stays positive while the actor",
        "still moves, the hop is not a minimizer of the measured JKO loss.",
        "This does not reconstruct per-minibatch decreases during joint training.",
        "",
        "## Figures",
        "",
        f"- `{Path(figure).name}`",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--extra-steps", type=int, default=10000)
    parser.add_argument("--log-steps", type=int, nargs="+", default=list(DEFAULT_LOG_STEPS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()
    backend = jax.default_backend()
    if args.device == "cpu" and backend != "cpu":
        raise RuntimeError(f"refusing non-CPU JAX backend {backend!r}")
    if args.device == "cuda" and backend not in ("gpu", "cuda"):
        raise RuntimeError(
            f"expected CUDA JAX backend, got {backend!r} devices={jax.devices()!r}"
        )
    log_steps = tuple(sorted(set(int(s) for s in args.log_steps if 0 <= int(s) <= args.extra_steps)))
    if args.extra_steps not in log_steps:
        log_steps = tuple(sorted(log_steps + (args.extra_steps,)))
    inventory = json.loads((ARCHIVE / "INVENTORY.json").read_text(encoding="utf-8"))
    records = []
    for row in inventory["rows"]:
        if (
            row["status"] == "evalable"
            and row["condition"] == "bar_p4"
            and row["environment"] in set(args.tasks)
            and int(row["T"]) == int(args.T)
            and int(row["seed"]) in set(args.seeds)
        ):
            records.append(resolve_checkpoint(row))
    if not records:
        raise FileNotFoundError(
            f"no evalable MART checkpoints for tasks={args.tasks} T={args.T} seeds={args.seeds}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "understood_as": (
                    "Freeze critic and previous actor; extra-optimize hop actor; "
                    "test whether snapshot ΔL>0 shrinks on Hopper-medium/expert T=10."
                ),
                "backend": jax.default_backend(),
                "devices": [str(d) for d in jax.devices()],
                "n_records": len(records),
                "checkpoint_paths": [row["checkpoint_path"] for row in records],
                "extra_steps": args.extra_steps,
                "log_steps": log_steps,
                "started_at": now_kst(),
            },
            indent=2,
        ),
        flush=True,
    )
    all_rows: list[dict[str, Any]] = []
    for record in records:
        for prev_role, role, hop in HOPS:
            rng_seed = (
                20260906
                + 1000 * int(record["seed"])
                + 10 * hop
                + (0 if "medium-v2" in record["environment"] else 1)
            )
            all_rows.extend(
                process_hop(
                    record,
                    prev_role,
                    role,
                    hop,
                    args.extra_steps,
                    log_steps,
                    args.batch_size,
                    args.lr,
                    rng_seed,
                )
            )
    fields = [
        "task",
        "T",
        "training_seed",
        "hop",
        "actor_a",
        "actor_b",
        "extra_steps",
        "batch_size",
        "lr",
        "tau_step",
        "n_eval",
        "q_weight",
        "mean_q_k",
        "mean_q_prev",
        "delta_q",
        "w2",
        "delta_L",
        "q_term",
        "q_gain_move_ratio",
        "objective_improved",
        "grad_l2",
        "action_rms_from_snapshot",
        "checkpoint_hash",
    ]
    out_csv = Path(args.out_csv) if args.out_csv else OUT / "extra_opt_curves.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    if not args.skip_report:
        figure = plot_curves(all_rows)
        write_report(all_rows, args.extra_steps, figure)
        (OUT / "RECEIPT.json").write_text(
            json.dumps(
                {
                    "built_at": now_kst(),
                    "n_rows": len(all_rows),
                    "extra_steps": args.extra_steps,
                    "log_steps": log_steps,
                    "tasks": args.tasks,
                    "T": args.T,
                    "backend": jax.default_backend(),
                    "devices": [str(d) for d in jax.devices()],
                    "out_csv": str(out_csv),
                    "note": (
                        "Negative delta_L is an improvement of the hop objective. "
                        "q_gain_move_ratio = (c_k ΔQ) / w2; values > 1 mean ΔL < 0."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"wrote {out_csv} rows={len(all_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
