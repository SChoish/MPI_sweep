#!/usr/bin/env python3
"""Measure actor-path agreement: π_{s+t} vs (π_s → π_t) from a_D=π_0.

Notation
--------
* ``a_D`` / ``π_0``: offline dataset actions on a shared observation batch.
* ``π_τ``: final-hop actor of a checkpoint trained with total coefficient ``τ``.
* Multi-hop (mpi2/mpi3): intermediate actors ``μ_1,...,μ_K`` with hop size τ/K;
  the chain is ``a_D → μ_1(obs) → ... → μ_K(obs)`` only in the training loss;
  at evaluation each ``μ_k`` is still a state→action map, so hop agreement is
  measured by comparing action outputs on the same observations.

This audit reports three complementary distances on shared (obs, a_D):

1. **Cross-τ actor composition (live endpoints + frozen continuation)**
   Compare ``π_{s+t}(obs)`` with ``Φ_t(π_s(obs))`` under the critic of the
   ``π_s`` checkpoint (and the reverse order).  Also compare each actor to
   the frozen map started at ``a_D``: ``π_τ(obs)`` vs ``Φ_τ(a_D)``.

2. **Frozen multi-hop from a_D**
   ``Φ_{s+t}(a_D)`` vs ``Φ_t(Φ_s(a_D))`` under a chosen critic (default:
   critic of ``π_{s+t}``).

3. **Cross-K hop agreement at fixed total τ**
   Final actions of mpi1 / mpi2 / mpi3, and within mpiK the intermediate
   hop actors vs the mpi1 endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("EIGEN_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import REPO_ROOT as ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train_td3bc import (  # noqa: E402
    DATASET_FILES,
    Actor,
    TwinCritic,
    load_checkpoint,
    load_transition,
)


DEFAULT_TAUS = (0.05, 0.1, 0.2, 0.4, 0.7, 1.5, 2.5, 4, 7, 10, 12, 14, 17, 20)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "logs" / "actor_path_semigroup"),
    )
    parser.add_argument("--envs", nargs="+", default=list(DATASET_FILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mpi1", "mpi2", "mpi3"],
        help="mpiK tags. Default layout is results/<tag>_s23 unless --method-dir is set.",
    )
    parser.add_argument(
        "--method-dir",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override checkpoint root for one method (repeatable). "
        "Example: --method-dir mpi4=/path/results/mpi4_norm",
    )
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=("explicit", "implicit"),
        default=["explicit", "implicit"],
    )
    parser.add_argument("--n-states", type=int, default=512)
    parser.add_argument("--interior-margin", type=float, default=0.05)
    parser.add_argument("--sample-seed", type=int, default=20260831)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--implicit-damping", type=float, default=0.25)
    return parser.parse_args()


def tau_token(value: float) -> str:
    return f"{float(value):g}"


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base)) % (2**63)


def available_splits(taus: list[float]) -> list[tuple[float, float, float]]:
    lookup = {round(value, 10): value for value in taus}
    splits: list[tuple[float, float, float]] = []
    for index, s in enumerate(taus):
        for t in taus[index:]:
            total = lookup.get(round(s + t, 10))
            if total is not None:
                splits.append((s, t, total))
    return splits


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(arr))))


def masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return rms(selected) if selected.size else float("nan")


def quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(
        [value for value in values if np.isfinite(value)], dtype=np.float64
    )
    if finite.size == 0:
        return {"median": None, "p90": None, "max": None}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def relative(defect: float, scale: float) -> float:
    return defect / max(scale, EPS)


def parse_method_dirs(
    results_root: Path, overrides: list[str]
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--method-dir must be METHOD=PATH, got {item!r}")
        method, raw = item.split("=", 1)
        method = method.strip()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (results_root / path).resolve()
        else:
            path = path.resolve()
        mapping[method] = path
    return mapping


def method_dir(
    results_root: Path, method: str, overrides: dict[str, Path] | None = None
) -> Path:
    if overrides and method in overrides:
        return overrides[method]
    return results_root / f"{method}_s23"


def checkpoint_path(
    results_root: Path,
    env_name: str,
    tau: float,
    method: str,
    seed: int,
    overrides: dict[str, Path] | None = None,
) -> Path:
    return (
        method_dir(results_root, method, overrides)
        / f"{env_name}_tau{tau_token(tau)}_{method}_seed{seed}"
        / "params_1000000.pkl"
    )


def infer_action_dim(params: Any) -> int:
    kernels = [
        leaf
        for leaf in jax.tree_util.tree_leaves(params)
        if hasattr(leaf, "ndim") and leaf.ndim == 2 and leaf.shape[0] == 256
    ]
    if not kernels:
        raise ValueError("cannot infer action_dim")
    return int(min(kernels, key=lambda leaf: leaf.shape[1]).shape[1])


def make_operator(max_iterations: int, tolerance: float, implicit_damping: float):
    critic = TwinCritic()

    @jax.jit
    def q1_values(params, states, actions):
        q1, _ = critic.apply(params, states, actions)
        return jnp.squeeze(q1, axis=-1)

    @jax.jit
    def q1_grad(params, states, actions):
        def q_sum(candidate):
            return jnp.sum(q1_values(params, states, candidate))

        return jax.grad(q_sum)(actions)

    @jax.jit
    def explicit_map(params, states, reference, step, max_action):
        q_scale = jnp.mean(jnp.abs(q1_values(params, states, reference))) + 1e-6
        beta = reference.shape[-1] * step / q_scale
        raw = reference + beta * q1_grad(params, states, reference)
        result = jnp.clip(raw, -max_action, max_action)
        projected = jnp.any(jnp.abs(raw - result) > 1e-8, axis=-1)
        return (
            result,
            q_scale,
            jnp.ones(reference.shape[0], dtype=bool),
            jnp.asarray(1),
            jnp.zeros(reference.shape[0]),
            projected,
        )

    @jax.jit
    def implicit_map(params, states, reference, step, max_action):
        q_scale = jnp.mean(jnp.abs(q1_values(params, states, reference))) + 1e-6
        beta = reference.shape[-1] * step / q_scale

        def cond(state):
            iteration, _candidate, delta = state
            return jnp.logical_and(
                iteration < max_iterations,
                delta > tolerance * implicit_damping,
            )

        def body(state):
            iteration, candidate, _delta = state
            raw = reference + beta * q1_grad(params, states, candidate)
            fixed_point = jnp.clip(raw, -max_action, max_action)
            proposal = (1.0 - implicit_damping) * candidate + implicit_damping * fixed_point
            delta = jnp.max(jnp.abs(proposal - candidate))
            return iteration + 1, proposal, delta

        initial = (jnp.asarray(0), reference, jnp.asarray(jnp.inf))
        iterations, result, _delta = jax.lax.while_loop(cond, body, initial)
        raw = reference + beta * q1_grad(params, states, result)
        fixed_point = jnp.clip(raw, -max_action, max_action)
        residual = jnp.max(jnp.abs(result - fixed_point), axis=-1)
        projected = jnp.any(jnp.abs(raw - fixed_point) > 1e-8, axis=-1)
        return (
            result,
            q_scale,
            residual <= tolerance * 2,
            iterations,
            residual,
            projected,
        )

    return {"explicit": explicit_map, "implicit": implicit_map}


def load_run(
    checkpoint: Path, states: np.ndarray, action_dim: int
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint)
    max_action = float(payload.get("max_action", 1.0))
    actors_params = list(payload["actors_params"])
    actor = Actor(action_dim=action_dim, max_action=max_action)
    apply = jax.jit(actor.apply)
    hop_actions = [
        np.asarray(apply(params, jnp.asarray(states))) for params in actors_params
    ]
    return {
        "checkpoint": str(checkpoint),
        "payload": payload,
        "critic_params": payload["critic_params"],
        "max_action": max_action,
        "hop_actions": hop_actions,
        "final_actions": hop_actions[-1],
        "n_hops": len(hop_actions),
        "mean": np.asarray(payload["mean"]),
        "std": np.asarray(payload["std"]),
    }


def apply_map(
    operator,
    critic_params,
    states: np.ndarray,
    reference: np.ndarray,
    step: float,
    max_action: float,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float]:
    out = operator(
        critic_params,
        jnp.asarray(states),
        jnp.asarray(reference),
        jnp.asarray(step, dtype=jnp.asarray(states).dtype),
        jnp.asarray(max_action, dtype=jnp.asarray(states).dtype),
    )
    actions = np.asarray(out[0])
    mask = np.asarray(out[2], dtype=bool)
    return (
        actions,
        mask,
        float(np.asarray(out[1])),
        int(np.asarray(out[3])),
        float(np.max(np.asarray(out[4]))),
        float(np.mean(np.asarray(out[5], dtype=bool))),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metric: str,
    fraction_key: str | None = "solver_converged_sample_fraction",
) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        buckets[key].append(row)
    out = []
    for key, group in sorted(buckets.items()):
        valid = [
            row
            for row in group
            if row.get("all_numeric_outputs_finite", True)
            and (
                fraction_key is None
                or row.get(fraction_key, 1.0) > 0
            )
        ]
        entry = {name: value for name, value in zip(group_keys, key)}
        entry["rows"] = len(group)
        entry["valid_rows"] = len(valid)
        if fraction_key is not None:
            entry["mean_solver_converged_sample_fraction"] = float(
                np.mean([row[fraction_key] for row in group])
            )
        entry[metric] = quantiles([row[metric] for row in valid])
        out.append(entry)
    return out


def main() -> None:
    args = parse_args()
    if args.n_states < 16:
        raise ValueError("--n-states must be at least 16")
    if not 0 <= args.interior_margin < 1:
        raise ValueError("--interior-margin must lie in [0, 1)")
    if not 0 < args.implicit_damping <= 1:
        raise ValueError("--implicit-damping must lie in (0, 1]")
    if unknown := sorted(set(args.envs) - set(DATASET_FILES)):
        raise ValueError(f"unknown environments: {unknown}")

    taus = sorted({float(v) for v in args.taus})
    splits = available_splits(taus)
    if not splits:
        raise ValueError("tau grid has no s+t splits")

    results_root = Path(args.results_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    method_dirs = parse_method_dirs(results_root, args.method_dir)
    operators = make_operator(
        args.max_iterations, args.tolerance, args.implicit_damping
    )

    composition_rows: list[dict[str, Any]] = []
    frozen_from_data_rows: list[dict[str, Any]] = []
    cross_k_rows: list[dict[str, Any]] = []
    hop_path_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    selected_indices: dict[str, list[int]] = {}

    for env_name in args.envs:
        data, mean, std = load_transition(env_name, data_dir, normalize=True)
        observations = np.asarray(data.observations)
        dataset_actions = np.asarray(data.actions)
        action_dim = int(dataset_actions.shape[-1])
        interior = np.all(
            np.abs(dataset_actions) <= 1.0 - args.interior_margin, axis=-1
        )
        candidates = np.flatnonzero(interior)
        if candidates.size < args.n_states:
            raise RuntimeError(
                f"{env_name}: only {candidates.size} interior actions"
            )
        rng = np.random.default_rng(stable_seed(env_name, args.sample_seed))
        selected = rng.choice(candidates, size=args.n_states, replace=False)
        selected_indices[env_name] = [int(v) for v in selected]
        states = observations[selected]
        a_d = dataset_actions[selected]  # π_0

        for seed in args.seeds:
            cache: dict[tuple[str, float], dict[str, Any]] = {}

            def get_run(method: str, tau: float) -> dict[str, Any] | None:
                key = (method, float(tau))
                if key in cache:
                    return cache[key]
                path = checkpoint_path(
                    results_root, env_name, tau, method, seed, method_dirs
                )
                if not path.is_file():
                    return None
                run = load_run(path, states, action_dim)
                if not np.allclose(run["mean"], mean):
                    raise ValueError(f"mean mismatch: {path}")
                if not np.allclose(run["std"], std):
                    raise ValueError(f"std mismatch: {path}")
                cache[key] = run
                return run

            composed = 0
            for method in args.methods:
                for s, t, total in splits:
                    run_s = get_run(method, s)
                    run_t = get_run(method, t)
                    run_u = get_run(method, total)
                    if run_s is None or run_t is None or run_u is None:
                        continue
                    pi_s = run_s["final_actions"]
                    pi_t = run_t["final_actions"]
                    pi_u = run_u["final_actions"]
                    composed += 1

                    for integrator in args.integrators:
                        op = operators[integrator]
                        cont_st, mask_st, _, it_st, res_st, proj_st = apply_map(
                            op,
                            run_s["critic_params"],
                            states,
                            pi_s,
                            t,
                            run_s["max_action"],
                        )
                        cont_ts, mask_ts, _, it_ts, res_ts, proj_ts = apply_map(
                            op,
                            run_t["critic_params"],
                            states,
                            pi_t,
                            s,
                            run_t["max_action"],
                        )
                        mask = np.logical_and(mask_st, mask_ts)
                        frac = float(np.mean(mask))
                        gap_direct = masked_rms(pi_u - a_d, mask)
                        defect_st = masked_rms(cont_st - pi_u, mask)
                        defect_ts = masked_rms(cont_ts - pi_u, mask)
                        order = masked_rms(cont_st - cont_ts, mask)
                        actor_s_to_u = masked_rms(pi_u - pi_s, mask)
                        scale = max(gap_direct, actor_s_to_u, EPS)
                        phi_s, mask_phi_s, *_ = apply_map(
                            op, run_s["critic_params"], states, a_d, s, run_s["max_action"]
                        )
                        phi_u, mask_phi_u, *_ = apply_map(
                            op, run_u["critic_params"], states, a_d, total, run_u["max_action"]
                        )
                        phi_s_u, mask1, *_ = apply_map(
                            op, run_u["critic_params"], states, a_d, s, run_u["max_action"]
                        )
                        phi_st_u, mask2, *_ = apply_map(
                            op,
                            run_u["critic_params"],
                            states,
                            phi_s_u,
                            t,
                            run_u["max_action"],
                        )
                        mask_data = np.logical_and.reduce(
                            [mask, mask_phi_s, mask_phi_u, mask1, mask2]
                        )
                        frac_data = float(np.mean(mask_data))
                        actor_vs_phi_s = masked_rms(pi_s - phi_s, mask_data)
                        actor_vs_phi_u = masked_rms(pi_u - phi_u, mask_data)
                        frozen_semigroup = masked_rms(phi_st_u - phi_u, mask_data)
                        actor_vs_frozen_compose = masked_rms(pi_u - phi_st_u, mask_data)
                        data_scale = max(
                            masked_rms(phi_u - a_d, mask_data),
                            masked_rms(pi_u - a_d, mask_data),
                            EPS,
                        )
                        finite = all(
                            math.isfinite(v)
                            for v in (
                                defect_st,
                                defect_ts,
                                actor_vs_phi_u,
                                frozen_semigroup,
                                actor_vs_frozen_compose,
                            )
                        )
                        composition_rows.append(
                            {
                                "environment": env_name,
                                "seed": seed,
                                "method": method,
                                "integrator": integrator,
                                "s": float(s),
                                "t": float(t),
                                "total": float(total),
                                "pi_u_vs_continue_pi_s_rms": defect_st,
                                "pi_u_vs_continue_pi_t_rms": defect_ts,
                                "relative_pi_u_vs_continue_mean": relative(
                                    0.5 * (defect_st + defect_ts), scale
                                ),
                                "order_defect_rms": order,
                                "relative_order_defect": relative(order, scale),
                                "actor_gap_pi_s_to_pi_u_rms": actor_s_to_u,
                                "actor_gap_aD_to_pi_u_rms": gap_direct,
                                "pi_s_vs_Phi_s_aD_rms": actor_vs_phi_s,
                                "pi_u_vs_Phi_u_aD_rms": actor_vs_phi_u,
                                "relative_pi_u_vs_Phi_u_aD": relative(
                                    actor_vs_phi_u, data_scale
                                ),
                                "pi_u_vs_Phi_t_Phi_s_aD_rms": actor_vs_frozen_compose,
                                "relative_pi_u_vs_Phi_t_Phi_s_aD": relative(
                                    actor_vs_frozen_compose, data_scale
                                ),
                                "Phi_u_vs_Phi_t_Phi_s_aD_rms": frozen_semigroup,
                                "relative_frozen_semigroup_from_aD": relative(
                                    frozen_semigroup, data_scale
                                ),
                                "solver_converged_sample_fraction": min(frac, frac_data),
                                "all_solver_calls_converged": min(frac, frac_data) == 1.0,
                                "max_solver_iterations": max(it_st, it_ts),
                                "max_fixed_point_residual": max(res_st, res_ts),
                                "max_projection_fraction": max(proj_st, proj_ts),
                                "all_numeric_outputs_finite": finite,
                                "n_states": args.n_states,
                            }
                        )
                        frozen_from_data_rows.append(
                            {
                                "environment": env_name,
                                "seed": seed,
                                "method": method,
                                "integrator": integrator,
                                "s": float(s),
                                "t": float(t),
                                "total": float(total),
                                "Phi_u_vs_Phi_t_Phi_s_aD_rms": frozen_semigroup,
                                "relative_frozen_semigroup_from_aD": relative(
                                    frozen_semigroup, data_scale
                                ),
                                "pi_u_vs_Phi_u_aD_rms": actor_vs_phi_u,
                                "pi_u_vs_Phi_t_Phi_s_aD_rms": actor_vs_frozen_compose,
                                "solver_converged_sample_fraction": frac_data,
                                "all_numeric_outputs_finite": finite,
                            }
                        )

            print(
                f"[compose] {env_name} seed={seed} ready_splits={composed} "
                f"integrators={args.integrators}",
                flush=True,
            )

            # --- 3: cross-K + hop paths on the full requested τ grid
            # (not only s+t split endpoints), loading whichever methods exist.
            for tau in taus:
                runs = {
                    method: run
                    for method in args.methods
                    if (run := get_run(method, tau)) is not None
                }
                if not runs:
                    continue
                methods = sorted(runs)
                for i, method_a in enumerate(methods):
                    for method_b in methods[i + 1 :]:
                        gap = rms(
                            runs[method_a]["final_actions"]
                            - runs[method_b]["final_actions"]
                        )
                        scale = max(
                            rms(runs[method_a]["final_actions"] - a_d),
                            rms(runs[method_b]["final_actions"] - a_d),
                            EPS,
                        )
                        cross_k_rows.append(
                            {
                                "environment": env_name,
                                "seed": seed,
                                "tau": float(tau),
                                "method_a": method_a,
                                "method_b": method_b,
                                "final_action_rms": gap,
                                "relative_final_action_rms": relative(gap, scale),
                                "aD_to_final_a_rms": rms(
                                    runs[method_a]["final_actions"] - a_d
                                ),
                                "aD_to_final_b_rms": rms(
                                    runs[method_b]["final_actions"] - a_d
                                ),
                                "n_states": args.n_states,
                            }
                        )

                ref_method = "mpi1" if "mpi1" in runs else None
                pi_ref = runs[ref_method]["final_actions"] if ref_method else None
                scale_ref = max(rms(pi_ref - a_d), EPS) if pi_ref is not None else None
                for method, run in runs.items():
                    prev = a_d
                    for hop_idx, actions in enumerate(run["hop_actions"], start=1):
                        hop_vs_ref = (
                            rms(actions - pi_ref) if pi_ref is not None else float("nan")
                        )
                        hop_path_rows.append(
                            {
                                "environment": env_name,
                                "seed": seed,
                                "tau": float(tau),
                                "method": method,
                                "hop": hop_idx,
                                "n_hops": run["n_hops"],
                                "ref_method": ref_method or "",
                                "hop_vs_mpi1_final_rms": hop_vs_ref,
                                "relative_hop_vs_mpi1_final": (
                                    relative(hop_vs_ref, scale_ref)
                                    if scale_ref is not None
                                    else float("nan")
                                ),
                                "hop_vs_prev_rms": rms(actions - prev),
                                "hop_vs_aD_rms": rms(actions - a_d),
                                "is_final_hop": hop_idx == run["n_hops"],
                                "n_states": args.n_states,
                            }
                        )
                        prev = actions

            print(
                f"[cross-k] {env_name} seed={seed} taus={len(taus)} methods={args.methods}",
                flush=True,
            )

        del data, observations, dataset_actions

    if not composition_rows and not cross_k_rows:
        raise RuntimeError("no rows produced")

    summary = {
        "n_composition_rows": len(composition_rows),
        "n_cross_k_rows": len(cross_k_rows),
        "n_hop_path_rows": len(hop_path_rows),
        "composition_by_integrator": summarize_by(
            composition_rows,
            ("method", "integrator"),
            "relative_pi_u_vs_continue_mean",
        ),
        "composition_by_split": summarize_by(
            composition_rows,
            ("method", "integrator", "s", "t", "total"),
            "relative_pi_u_vs_continue_mean",
        ),
        "actor_vs_frozen_from_aD_by_integrator": summarize_by(
            composition_rows,
            ("method", "integrator"),
            "relative_pi_u_vs_Phi_u_aD",
        ),
        "frozen_semigroup_from_aD_by_split": summarize_by(
            composition_rows,
            ("method", "integrator", "s", "t", "total"),
            "relative_frozen_semigroup_from_aD",
        ),
        "cross_k_by_pair": summarize_by(
            cross_k_rows,
            ("method_a", "method_b"),
            "relative_final_action_rms",
            fraction_key=None,
        ),
        "cross_k_by_tau": summarize_by(
            cross_k_rows,
            ("tau", "method_a", "method_b"),
            "relative_final_action_rms",
            fraction_key=None,
        ),
        "hop_final_vs_mpi1": summarize_by(
            [row for row in hop_path_rows if row["is_final_hop"]],
            ("method",),
            "relative_hop_vs_mpi1_final",
            fraction_key=None,
        ),
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "actor_path_pi_s_plus_t_vs_pi_s_then_pi_t_from_aD",
        "pi_0": "dataset actions a_D on shared interior observations",
        "claim_scope": (
            "compares learned actor endpoints and frozen continuations; "
            "not an end-to-end return certificate"
        ),
        "comparisons": {
            "pi_u_vs_continue_pi_s": "RMS(π_{s+t}(obs) - Φ_t^{Q_s}(π_s(obs)))",
            "pi_u_vs_Phi_u_aD": "RMS(π_{s+t}(obs) - Φ_{s+t}^{Q_u}(a_D))",
            "Phi_u_vs_Phi_t_Phi_s_aD": "RMS(Φ_{s+t}(a_D) - Φ_t(Φ_s(a_D))) under Q_u",
            "cross_k": "RMS(final(mpiA) - final(mpiB)) / move-from-a_D",
            "hop_path": "each μ_k(obs) vs mpi1 final at same total τ",
        },
        "results_root": str(results_root),
        "environments": args.envs,
        "seeds": args.seeds,
        "taus": taus,
        "splits": [{"s": s, "t": t, "total": u} for s, t, u in splits],
        "methods": args.methods,
        "method_dirs": {k: str(v) for k, v in method_dirs.items()},
        "integrators": args.integrators,
        "n_states": args.n_states,
        "interior_margin": args.interior_margin,
        "sample_seed": args.sample_seed,
        "selected_indices": selected_indices,
        "missing_checkpoints": missing,
        "jax_backend": jax.default_backend(),
        "cli": vars(args),
    }

    write_csv(out_dir / "actor_composition.csv", composition_rows)
    write_csv(out_dir / "frozen_from_aD.csv", frozen_from_data_rows)
    write_csv(out_dir / "cross_k_finals.csv", cross_k_rows)
    write_csv(out_dir / "hop_paths.csv", hop_path_rows)
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # compact stdout
    compact = {
        "composition": {
            f"{row.get('method', '')}/{row['integrator']}": row[
                "relative_pi_u_vs_continue_mean"
            ]
            for row in summary["composition_by_integrator"]
        },
        "actor_vs_Phi_from_aD": {
            f"{row.get('method', '')}/{row['integrator']}": row[
                "relative_pi_u_vs_Phi_u_aD"
            ]
            for row in summary["actor_vs_frozen_from_aD_by_integrator"]
        },
        "cross_k": {
            f"{row['method_a']}↔{row['method_b']}": row["relative_final_action_rms"]
            for row in summary["cross_k_by_pair"]
        },
        "hop_final_vs_mpi1": {
            row["method"]: row["relative_hop_vs_mpi1_final"]
            for row in summary["hop_final_vs_mpi1"]
        },
        "missing": len(missing),
    }
    print(json.dumps(compact, indent=2), flush=True)
    print(
        f"[done] composition={len(composition_rows)} cross_k={len(cross_k_rows)} "
        f"hop_paths={len(hop_path_rows)} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
