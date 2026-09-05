#!/usr/bin/env python3
"""Summarize P0 all-actor eval + action diagnostics. Does not train."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.diagnostics.p0_intermediate_actors import (
    DEFAULT_OUT_DIR,
    HIGH_T,
    T_VALUES,
    load_json,
    pair_key,
    read_csv,
    write_csv,
    write_json,
)
from scripts.experiments.run_p0_bar_two_actor_p4 import ENVIRONMENTS as TASKS


SHORT_TASK = {
    "hopper-medium-v2": "H-m",
    "hopper-medium-replay-v2": "H-mr",
    "hopper-expert-v2": "H-e",
    "halfcheetah-medium-v2": "HC-m",
    "halfcheetah-medium-replay-v2": "HC-mr",
    "halfcheetah-expert-v2": "HC-e",
    "walker2d-medium-v2": "W-m",
    "walker2d-medium-replay-v2": "W-mr",
    "walker2d-expert-v2": "W-e",
}


def _f(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def policy_means(episode_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in episode_rows:
        key = (
            row["environment"],
            int(row["T"]),
            int(row["seed"]),
            row["condition"],
            row["actor_id"],
            int(row["actor_index"]),
        )
        groups[key].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        env, tau, seed, condition, actor_id, actor_index = key
        scores = [_f(row, "normalized_score") for row in rows]
        rets = [_f(row, "raw_return") for row in rows]
        out.append(
            {
                "environment": env,
                "T": tau,
                "seed": seed,
                "condition": condition,
                "actor_id": actor_id,
                "actor_index": actor_index,
                "n_episodes": len(rows),
                "normalized_score_mean": float(np.mean(scores)),
                "normalized_score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "raw_return_mean": float(np.mean(rets)),
                "checkpoint_sha256": rows[0].get("checkpoint_sha256", ""),
                "pair_key": pair_key(env, tau, seed),
            }
        )
    return out


def mart_profiles(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in policy_rows:
        if row["condition"] != "bar_p4":
            continue
        by_cell[row["pair_key"]][int(row["actor_index"])] = row
    profiles = []
    for pkey, actors in sorted(by_cell.items()):
        if set(actors) != {1, 2, 3, 4}:
            continue
        scores = {k: actors[k]["normalized_score_mean"] for k in (1, 2, 3, 4)}
        hops = {
            "d12": scores[2] - scores[1],
            "d23": scores[3] - scores[2],
            "d34": scores[4] - scores[3],
            "d41": scores[4] - scores[1],
        }
        best_idx = max(scores, key=scores.get)
        profiles.append(
            {
                "pair_key": pkey,
                "environment": actors[1]["environment"],
                "T": actors[1]["T"],
                "seed": actors[1]["seed"],
                "j1": scores[1],
                "j2": scores[2],
                "j3": scores[3],
                "j4": scores[4],
                **hops,
                "oracle_best_index": best_idx,
                "oracle_best_score": scores[best_idx],
                "final_worse_than_mid": scores[4] < max(scores[1], scores[2], scores[3]) - 1e-12,
                "high_T": int(actors[1]["T"]) in HIGH_T,
            }
        )
    return profiles


def paired_contrast(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    want = {
        ("bar_p4", "mart_mu1"): "mart_mu1",
        ("bar_p4", "mart_mu4"): "mart_mu4",
        ("two_actor_p4", "two_actor_target"): "two_target",
        ("two_actor_p4", "two_actor_deploy"): "two_deploy",
    }
    by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in policy_rows:
        label = want.get((row["condition"], row["actor_id"]))
        if label:
            by_cell[row["pair_key"]][label] = row
    out = []
    for pkey, items in sorted(by_cell.items()):
        if set(items) != set(want.values()):
            continue
        env = items["mart_mu4"]["environment"]
        tau = items["mart_mu4"]["T"]
        seed = items["mart_mu4"]["seed"]
        out.append(
            {
                "pair_key": pkey,
                "environment": env,
                "T": tau,
                "seed": seed,
                "mart_mu1": items["mart_mu1"]["normalized_score_mean"],
                "mart_mu4": items["mart_mu4"]["normalized_score_mean"],
                "two_target": items["two_target"]["normalized_score_mean"],
                "two_deploy": items["two_deploy"]["normalized_score_mean"],
                "mart4_minus_two_deploy": (
                    items["mart_mu4"]["normalized_score_mean"]
                    - items["two_deploy"]["normalized_score_mean"]
                ),
                "mart1_minus_two_target": (
                    items["mart_mu1"]["normalized_score_mean"]
                    - items["two_target"]["normalized_score_mean"]
                ),
            }
        )
    return out


def _metric_map(distance_rows: list[dict[str, str]], metric: str, state_set: str) -> dict[str, float]:
    out = {}
    for row in distance_rows:
        if row["metric"] == metric and row["state_set"] == state_set:
            out[pair_key(row["environment"], int(row["T"]), int(row["seed"]))] = float(row["mean"])
    return out


def hypothesis_readout(
    profiles: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    distance_rows: list[dict[str, str]],
) -> dict[str, Any]:
    n = len(profiles)
    if n == 0:
        return {"n_mart_profiles": 0, "note": "no complete MART actor-index profiles"}
    adj = np.array([[p["d12"], p["d23"], p["d34"]] for p in profiles], dtype=np.float64)
    abs_adj = np.abs(adj)
    sat_share = float(np.mean(abs_adj[:, 1:] < 0.5 * (abs_adj[:, [0]] + 1e-9)))
    revert = sum(1 for p in profiles if p["final_worse_than_mid"])
    high = [p for p in profiles if p["high_T"]]
    high_first_neg = []
    for p in high:
        hops = [p["d12"], p["d23"], p["d34"]]
        idx = next((i + 1 for i, v in enumerate(hops) if v < -1.0), None)
        high_first_neg.append(idx)
    ds_rms = _metric_map(distance_rows, "mart_mu4_vs_two_deploy_rms", "dataset")
    paired_with_dist = [p for p in paired if p["pair_key"] in ds_rms]
    if paired_with_dist:
        x = np.array([ds_rms[p["pair_key"]] for p in paired_with_dist])
        y = np.array([p["mart4_minus_two_deploy"] for p in paired_with_dist])
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        far_close_ret = float(np.mean(np.abs(y[x >= np.median(x)]))) if len(x) else float("nan")
        near_ret = float(np.mean(np.abs(y[x < np.median(x)]))) if len(x) else float("nan")
    else:
        corr = far_close_ret = near_ret = float("nan")
    task_means = {}
    for p in paired:
        task_means.setdefault(p["environment"], []).append(p["mart4_minus_two_deploy"])
    task_mean = {k: float(np.mean(v)) for k, v in task_means.items()}
    overall = float(np.mean(list(task_mean.values()))) if task_mean else float("nan")
    first_diff = [p["mart1_minus_two_target"] for p in paired]
    return {
        "n_mart_profiles": n,
        "n_paired_cells": len(paired),
        "mean_hop_deltas": {
            "d12": float(np.mean(adj[:, 0])),
            "d23": float(np.mean(adj[:, 1])),
            "d34": float(np.mean(adj[:, 2])),
            "d41": float(np.mean([p["d41"] for p in profiles])),
        },
        "share_later_hops_smaller_than_half_first": sat_share,
        "n_final_worse_than_mid_oracle": revert,
        "share_final_worse_than_mid_oracle": revert / n,
        "high_T_first_negative_hop": {
            "n": len(high),
            "none": sum(v is None for v in high_first_neg),
            "hop2": sum(v == 1 for v in high_first_neg),
            "hop3": sum(v == 2 for v in high_first_neg),
            "hop4": sum(v == 3 for v in high_first_neg),
        },
        "action_distance_vs_return_corr_dataset": corr,
        "mean_abs_return_gap_high_vs_low_action_distance": [far_close_ret, near_ret],
        "task_mean_mart4_minus_two": task_mean,
        "task_equal_mean_mart4_minus_two": overall,
        "mean_abs_first_actor_gap": float(np.mean(np.abs(first_diff))) if first_diff else float("nan"),
        "seed_limit": "training seeds are {0,1} only; episode repeats are not extra training seeds",
    }


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_profiles(profiles: list[dict[str, Any]], out_dir: Path) -> None:
    tasks = sorted({p["environment"] for p in profiles})
    if not tasks:
        return
    n = len(tasks)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
    x = [1, 2, 3, 4]
    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        for tau in T_VALUES:
            pts = [p for p in profiles if p["environment"] == task and p["T"] == tau]
            if not pts:
                continue
            for p in pts:
                ax.plot(x, [p["j1"], p["j2"], p["j3"], p["j4"]], color="0.75", lw=0.8)
            mean = np.mean([[p["j1"], p["j2"], p["j3"], p["j4"]] for p in pts], axis=0)
            ax.plot(x, mean, lw=2, label=f"T={tau}")
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xlabel("MART actor index")
        ax.set_ylabel("D4RL normalized score")
        ax.legend(fontsize=7, frameon=False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("MART actor-index return profiles (thin: seed; thick: two-seed mean)")
    _save_fig(out_dir / "figures" / "actor_index_profiles.png")


def plot_hop_heatmap(profiles: list[dict[str, Any]], out_dir: Path) -> None:
    tasks = [t for t in TASKS if any(p["environment"] == t for p in profiles)]
    if not tasks:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, hop, title in zip(
        axes, ("d12", "d23", "d34"), ("J(μ2)-J(μ1)", "J(μ3)-J(μ2)", "J(μ4)-J(μ3)")
    ):
        mat = np.full((len(tasks), len(T_VALUES)), np.nan)
        for i, task in enumerate(tasks):
            for j, tau in enumerate(T_VALUES):
                vals = [p[hop] for p in profiles if p["environment"] == task and p["T"] == tau]
                if vals:
                    mat[i, j] = float(np.mean(vals))
        vmax = np.nanmax(np.abs(mat))
        vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else vmax
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(T_VALUES)))
        ax.set_xticklabels(list(T_VALUES))
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([SHORT_TASK.get(t, t) for t in tasks], fontsize=8)
        ax.set_xlabel("T")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Adjacent-hop D4RL score change (two-seed mean). Oracle mid-actor is not a selector.")
    _save_fig(out_dir / "figures" / "adjacent_hop_heatmap.png")


def plot_scatter(
    paired: list[dict[str, Any]], distance_rows: list[dict[str, str]], out_dir: Path
) -> None:
    ds = _metric_map(distance_rows, "mart_mu4_vs_two_deploy_rms", "dataset")
    pts = [p for p in paired if p["pair_key"] in ds]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for tau in T_VALUES:
        sub = [p for p in pts if p["T"] == tau]
        if not sub:
            continue
        ax.scatter(
            [ds[p["pair_key"]] for p in sub],
            [p["mart4_minus_two_deploy"] for p in sub],
            label=f"T={tau}",
            s=28,
        )
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("RMS action distance MART μ4 vs two-actor deploy (dataset states)")
    ax.set_ylabel("Return gap: MART μ4 minus two-actor deploy")
    ax.set_title("Action distance vs return difference (two training seeds)")
    ax.legend(frameon=False, fontsize=8)
    _save_fig(out_dir / "figures" / "distance_vs_return_scatter.png")


def write_report(
    out_dir: Path,
    coverage: Mapping[str, Any],
    hypo: Mapping[str, Any],
    profiles: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    lines = [
        "# P0 intermediate-actor diagnostic",
        "",
        "Scope: frozen P0 K=4 grid (MART4 vs two_actor_p4). This is a checkpoint",
        "evaluation, not a new training run. Historical first/endpoint columns are",
        "not mixed into these scores. Actor 2 of a K=4 run is not a K=2 policy.",
        "All actors come from the same 1M checkpoint (actor index ≠ training step).",
        "",
        "## Coverage (executed)",
        "",
        f"- Planned runs: {coverage.get('planned_runs')}",
        f"- Present 1M checkpoints on this host: {coverage.get('present_runs')} "
        f"({coverage.get('run_coverage'):.1%} of 180)",
        f"- Paired cells with both methods: {coverage.get('paired_cells_both_methods')} / "
        f"{coverage.get('planned_pairs')}",
        f"- Present policies: {coverage.get('present_policies')} / {coverage.get('planned_policies')}",
        f"- Tasks present: {', '.join(coverage.get('tasks_present') or []) or 'none'}",
        f"- Tasks missing (ext_csv head shard, not substituted): "
        f"{', '.join(coverage.get('tasks_missing') or []) or 'none'}",
        "",
        "## What was actually run",
        "",
        f"- Complete MART actor-index profiles: {hypo.get('n_mart_profiles')}",
        f"- Paired MART vs two-actor cells: {hypo.get('n_paired_cells')}",
        "- Episode protocol: 10 deterministic episodes, seeds 1000–1009, shared across",
        "  every compared policy of a task. Episode repeats are not extra training seeds.",
        "- Eval stack: `train_td3bc.EVAL_ENV` Gymnasium v4 + checkpoint mean/std.",
        "",
        "## Actor-index return (MART)",
        "",
        json.dumps(hypo.get("mean_hop_deltas"), indent=2),
        "",
        f"- Cells where final actor is worse than the best earlier actor (oracle only): "
        f"{hypo.get('n_final_worse_than_mid_oracle')} "
        f"({hypo.get('share_final_worse_than_mid_oracle')})",
        f"- Later-hop |ΔJ| < half of first hop, share of hop-pairs: "
        f"{hypo.get('share_later_hops_smaller_than_half_first')}",
        f"- High-T={{10,14,20}} first hop with ΔJ < -1: {hypo.get('high_T_first_negative_hop')}",
        "",
        "## MART vs two-actor",
        "",
        f"- Task-equal mean (MART μ4 − two-actor deploy) on local paired cells: "
        f"{hypo.get('task_equal_mean_mart4_minus_two')}",
        f"- Per-task means: {json.dumps(hypo.get('task_mean_mart4_minus_two'), indent=2, sort_keys=True)}",
        f"- Dataset-state action-distance vs return-gap correlation: "
        f"{hypo.get('action_distance_vs_return_corr_dataset')}",
        f"- Mean |first-actor gap| MART μ1 vs two-actor target: "
        f"{hypo.get('mean_abs_first_actor_gap')}",
        "",
        "## Hypotheses (not pre-declared conclusions)",
        "",
        _hypothesis_bullets(hypo),
        "",
        "## Limits",
        "",
        "- Local coverage is the ext_csh tail90 shard, not the full nine-task grid.",
        "- Two training seeds only.",
        "- Oracle mid-actor return is not an offline selection result.",
        "- Learned Q and local cosine do not prove critic accuracy or global equivalence at large T.",
        "- P0.B shared-driver training was not launched.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd /home/ext_csh/MPI_sweep",
        "export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu",
        "/home/ext_csh/miniconda3/envs/capo_jax/bin/python -u \\",
        "  scripts/diagnostics/run_p0_intermediate_actors.py all --workers 4",
        "/home/ext_csh/miniconda3/envs/capo_jax/bin/python -u \\",
        "  scripts/diagnostics/analyze_p0_intermediate_actors.py",
        "```",
        "",
    ]
    (out_dir / "ANALYSIS_GENERATED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hypothesis_bullets(hypo: Mapping[str, Any]) -> str:
    hops = hypo.get("mean_hop_deltas") or {}
    d12, d23, d34 = hops.get("d12"), hops.get("d23"), hops.get("d34")
    sat = hypo.get("share_later_hops_smaller_than_half_first")
    revert = hypo.get("share_final_worse_than_mid_oracle")
    corr = hypo.get("action_distance_vs_return_corr_dataset")
    first_gap = hypo.get("mean_abs_first_actor_gap")
    task_means = hypo.get("task_mean_mart4_minus_two") or {}
    signs = [np.sign(v) for v in task_means.values() if v == v]
    mixed = len(set(int(s) for s in signs if s != 0)) > 1
    lines = [
        f"- H1 saturation: later |ΔJ| often smaller than the first hop "
        f"(share={sat}; mean hops d12={d12}, d23={d23}, d34={d34}). "
        "This is a magnitude pattern, not proof that later actors are copies.",
        f"- H2 accumulation/reversion: final worse than best earlier actor in {revert} of MART profiles "
        "(oracle diagnostic only).",
        "- H3 direction alignment: see `action_distances.csv` cosine rows; near-zero hops are excluded. "
        "Local cosine is not global equivalence at large T.",
        f"- H4 similar return, different action: dataset RMS(μ4, deploy) vs return gap corr={corr}. "
        "A weak correlation with non-trivial action distance supports this more than H1 alone.",
        f"- H5 task cancellation: task means mixed signs={mixed}; "
        f"task-equal mean={hypo.get('task_equal_mean_mart4_minus_two')}.",
        f"- H6 path difference before extraction: mean |μ1 − two-actor target| = {first_gap}. "
        "Independent first-actor/critic trajectories can contribute; this is not a shared-driver isolation.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    inventory = load_json(out_dir / "INVENTORY.json") if (out_dir / "INVENTORY.json").is_file() else {}
    coverage = inventory.get("coverage") or {}
    episodes = read_csv(out_dir / "episodes.csv")
    distances = read_csv(out_dir / "action_distances.csv")
    policies = policy_means(episodes)
    profiles = mart_profiles(policies)
    paired = paired_contrast(policies)
    write_csv(
        out_dir / "policy_means.csv",
        policies,
        [
            "environment",
            "T",
            "seed",
            "condition",
            "actor_id",
            "actor_index",
            "n_episodes",
            "normalized_score_mean",
            "normalized_score_std",
            "raw_return_mean",
            "pair_key",
            "checkpoint_sha256",
        ],
    )
    write_csv(
        out_dir / "mart_hop_profiles.csv",
        profiles,
        [
            "pair_key",
            "environment",
            "T",
            "seed",
            "j1",
            "j2",
            "j3",
            "j4",
            "d12",
            "d23",
            "d34",
            "d41",
            "oracle_best_index",
            "oracle_best_score",
            "final_worse_than_mid",
            "high_T",
        ],
    )
    write_csv(
        out_dir / "paired_mart_two_actor.csv",
        paired,
        [
            "pair_key",
            "environment",
            "T",
            "seed",
            "mart_mu1",
            "mart_mu4",
            "two_target",
            "two_deploy",
            "mart4_minus_two_deploy",
            "mart1_minus_two_target",
        ],
    )
    hypo = hypothesis_readout(profiles, paired, distances)
    write_json(out_dir / "HYPOTHESES.json", hypo)
    plot_profiles(profiles, out_dir)
    plot_hop_heatmap(profiles, out_dir)
    plot_scatter(paired, distances, out_dir)
    write_report(out_dir, coverage, hypo, profiles, paired)
    print(json.dumps({"n_policies": len(policies), "n_profiles": len(profiles), "n_paired": len(paired)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
