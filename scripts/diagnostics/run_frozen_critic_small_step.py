#!/usr/bin/env python3
"""Frozen-critic action-level validation of the local explicit/implicit limit.

This is deliberately not an end-to-end return experiment. For canonical
TD3+BC T=1 critics, it freezes Q, samples interior dataset actions, and compares
one projected explicit action step with the fixed point of the corresponding
proximal first-order condition over decreasing step sizes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4",
)

from _lab_import import REPO_ROOT as ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train_td3bc import DATASET_FILES, TwinCritic, load_checkpoint, load_transition  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results_qnorm"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "audit_report_20260829" / "frozen_critic_small_step"),
    )
    parser.add_argument(
        "--envs",
        default=" ".join(DATASET_FILES),
        help="Space-separated D4RL environment names.",
    )
    parser.add_argument("--seeds", default="0 1")
    parser.add_argument("--n-states", type=int, default=512)
    parser.add_argument("--interior-margin", type=float, default=0.05)
    parser.add_argument(
        "--steps",
        default="0.001 0.003 0.01 0.03 0.1",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    return parser.parse_args()


def tree_params(payload: dict, key: str):
    value = payload[key]
    if isinstance(value, dict) and set(value) == {"params"}:
        return value
    return value


def local_summary(rows: list[dict]) -> dict:
    eligible = [
        row
        for row in rows
        if row["h"] <= 0.01
        and row["converged"]
        and row["projection_fraction_union"] == 0.0
    ]
    positive = [
        row for row in eligible if row["explicit_implicit_rms"] > 0.0
    ]
    slope = None
    if len(positive) >= 3:
        x = np.log([row["h"] for row in positive])
        y = np.log([row["explicit_implicit_rms"] for row in positive])
        slope = float(np.polyfit(x, y, 1)[0])
    return {
        "local_loglog_slope_when_identifiable": slope,
        "local_points_eligible": len(eligible),
        "local_max_relative_difference": (
            max(row["relative_difference"] for row in eligible)
            if eligible
            else None
        ),
        "local_exact_agreement_fraction": (
            float(
                np.mean(
                    [
                        row["explicit_implicit_rms"] == 0.0
                        for row in eligible
                    ]
                )
            )
            if eligible
            else None
        ),
    }

def evaluate_checkpoint(
    checkpoint: Path,
    states: np.ndarray,
    references: np.ndarray,
    steps: list[float],
    max_iterations: int,
    tolerance: float,
) -> list[dict]:
    payload = load_checkpoint(checkpoint)
    critic = TwinCritic()
    critic_params = tree_params(payload, "critic_params")
    max_action = float(payload.get("max_action", 1.0))
    states_jax = jnp.asarray(states)
    references_jax = jnp.asarray(references)

    def q1_values(actions):
        q1, _ = critic.apply(critic_params, states_jax, actions)
        return jnp.squeeze(q1, axis=-1)

    def q_sum(actions):
        return jnp.sum(q1_values(actions))

    q_jit = jax.jit(q1_values)
    grad_jit = jax.jit(jax.grad(q_sum))
    q_ref = np.asarray(q_jit(references_jax), dtype=np.float64)
    q_scale = float(np.mean(np.abs(q_ref)) + 1e-6)
    action_dim = references.shape[-1]
    grad_ref = grad_jit(references_jax)
    rows = []

    for h in steps:
        beta = float(action_dim * h / q_scale)
        unprojected_explicit = references_jax + beta * grad_ref
        explicit = jnp.clip(unprojected_explicit, -max_action, max_action)
        explicit_projected = jnp.any(
            jnp.abs(unprojected_explicit - explicit) > 1e-8, axis=-1
        )

        implicit = references_jax
        converged = False
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            proposal_raw = references_jax + beta * grad_jit(implicit)
            proposal = jnp.clip(proposal_raw, -max_action, max_action)
            delta = float(jnp.max(jnp.abs(proposal - implicit)))
            implicit = proposal
            iterations = iteration
            if delta <= tolerance:
                converged = True
                break

        implicit_raw = references_jax + beta * grad_jit(implicit)
        implicit_projection = jnp.any(
            jnp.abs(implicit_raw - jnp.clip(implicit_raw, -max_action, max_action))
            > 1e-8,
            axis=-1,
        )
        residual = implicit - jnp.clip(
            implicit_raw, -max_action, max_action
        )
        difference = implicit - explicit
        explicit_move = explicit - references_jax
        implicit_move = implicit - references_jax
        diff_rms = float(jnp.sqrt(jnp.mean(jnp.square(difference))))
        explicit_rms = float(jnp.sqrt(jnp.mean(jnp.square(explicit_move))))
        implicit_rms = float(jnp.sqrt(jnp.mean(jnp.square(implicit_move))))
        projection_union = jnp.logical_or(
            explicit_projected, implicit_projection
        )
        rows.append(
            {
                "h": float(h),
                "beta": beta,
                "q_scale": q_scale,
                "converged": bool(converged),
                "iterations": int(iterations),
                "fixed_point_residual_max": float(jnp.max(jnp.abs(residual))),
                "explicit_implicit_rms": diff_rms,
                "explicit_move_rms": explicit_rms,
                "implicit_move_rms": implicit_rms,
                "relative_difference": diff_rms / max(implicit_rms, 1e-12),
                "projection_fraction_explicit": float(
                    jnp.mean(explicit_projected)
                ),
                "projection_fraction_implicit": float(
                    jnp.mean(implicit_projection)
                ),
                "projection_fraction_union": float(jnp.mean(projection_union)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.n_states < 16:
        raise ValueError("--n-states must be at least 16")
    if not 0.0 <= args.interior_margin < 1.0:
        raise ValueError("--interior-margin must lie in [0, 1)")
    steps = sorted(float(value) for value in args.steps.split())
    if not steps or any(value <= 0.0 for value in steps):
        raise ValueError("--steps must contain positive values")
    seeds = [int(value) for value in args.seeds.split()]
    envs = args.envs.split()
    unknown = sorted(set(envs) - set(DATASET_FILES))
    if not envs or unknown:
        raise ValueError(f"invalid --envs; unknown={unknown}")
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "frozen_critic_action_level_small_step",
        "claim_scope": (
            "local explicit/implicit consistency under a fixed learned critic; "
            "not a return or causal actor-critic stability result"
        ),
        "reference_actions": "interior actions sampled from each offline dataset",
        "implicit_solver": (
            "projected fixed-point iteration for a=a0+(d*h/C)*grad_Q(a)"
        ),
        "explicit_step": "clip(a0+(d*h/C)*grad_Q(a0))",
        "critic_scale": "C=mean absolute frozen Q1 over the sampled batch",
        "expected_local_behavior": (
            "relative explicit/implicit discrepancy tends to zero. In smooth "
            "regions the absolute discrepancy is O(h^2); a ReLU critic is "
            "locally affine and can give exact agreement until a kink is crossed."
        ),
        "checkpoint_rule": "canonical TD3+BC T=1 final checkpoint",
        "steps": steps,
        "environments": envs,
        "seeds": seeds,
        "n_states": args.n_states,
        "interior_margin": args.interior_margin,
        "tolerance": args.tolerance,
        "max_iterations": args.max_iterations,
        "cli": vars(args),
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict] = []
    run_summaries = []
    for env_index, env_name in enumerate(envs):
        data, mean, std = load_transition(env_name, data_dir, normalize=True)
        actions = np.asarray(data.actions)
        observations = np.asarray(data.observations)
        interior = np.all(
            np.abs(actions) <= (1.0 - args.interior_margin), axis=-1
        )
        candidate_indices = np.flatnonzero(interior)
        if candidate_indices.size < args.n_states:
            raise RuntimeError(
                f"{env_name} has only {candidate_indices.size} interior actions"
            )
        for seed in seeds:
            checkpoint = (
                results_dir
                / f"{env_name}_tau1_seed{seed}"
                / "params_1000000.pkl"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = load_checkpoint(checkpoint)
            if not np.allclose(np.asarray(payload["mean"]), mean):
                raise ValueError(f"normalization mean mismatch for {checkpoint}")
            if not np.allclose(np.asarray(payload["std"]), std):
                raise ValueError(f"normalization std mismatch for {checkpoint}")
            rng = np.random.default_rng(20_260_829 + 100 * env_index + seed)
            selected = rng.choice(
                candidate_indices, size=args.n_states, replace=False
            )
            run_rows = evaluate_checkpoint(
                checkpoint,
                observations[selected],
                actions[selected],
                steps,
                args.max_iterations,
                args.tolerance,
            )
            diagnostics = local_summary(run_rows)
            for row in run_rows:
                row.update(
                    {
                        "environment": env_name,
                        "seed": seed,
                        "checkpoint": str(checkpoint.resolve()),
                        "n_states": args.n_states,
                        **diagnostics,
                    }
                )
                rows.append(row)
            run_summary = {
                "environment": env_name,
                "seed": seed,
                **diagnostics,
                "all_local_points_converged": all(
                    row["converged"]
                    for row in run_rows
                    if row["h"] <= 0.01
                ),
                "local_projection_free": all(
                    row["projection_fraction_union"] == 0.0
                    for row in run_rows
                    if row["h"] <= 0.01
                ),
            }
            run_summaries.append(run_summary)
            relative = diagnostics["local_max_relative_difference"]
            relative_label = "NA" if relative is None else f"{relative:.4g}"
            print(
                f"[small-step] {env_name} seed={seed} "
                f"max-relative={relative_label}",
                flush=True,
            )
        del data, observations, actions

    csv_path = out_dir / "frozen_critic_small_step.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    finite_slopes = np.asarray(
        [
            row["local_loglog_slope_when_identifiable"]
            for row in run_summaries
            if row["local_loglog_slope_when_identifiable"] is not None
            and np.isfinite(row["local_loglog_slope_when_identifiable"])
        ],
        dtype=np.float64,
    )
    local_maxima = np.asarray(
        [
            row["local_max_relative_difference"]
            for row in run_summaries
            if row["local_max_relative_difference"] is not None
            and np.isfinite(row["local_max_relative_difference"])
        ],
        dtype=np.float64,
    )
    summary = {
        "n_runs": len(run_summaries),
        "n_runs_with_identifiable_slope": int(finite_slopes.size),
        "median_identifiable_local_loglog_slope": (
            float(np.median(finite_slopes)) if finite_slopes.size else None
        ),
        "median_run_max_local_relative_difference": (
            float(np.median(local_maxima)) if local_maxima.size else None
        ),
        "max_run_max_local_relative_difference": (
            float(np.max(local_maxima)) if local_maxima.size else None
        ),
        "local_relative_tolerance": 0.05,
        "all_runs_within_local_relative_tolerance": bool(
            local_maxima.size == len(run_summaries)
            and np.all(local_maxima <= 0.05)
        ),
        "all_runs_local_converged": all(
            row["all_local_points_converged"] for row in run_summaries
        ),
        "all_runs_local_projection_free": all(
            row["local_projection_free"] for row in run_summaries
        ),
        "interpretation": (
            "An unidentifiable O(h^2) slope can be expected when the frozen "
            "ReLU critic stays in one locally affine activation region and "
            "explicit and implicit steps agree to floating-point precision."
        ),
        "runs": run_summaries,
        "csv": str(csv_path.resolve()),
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
