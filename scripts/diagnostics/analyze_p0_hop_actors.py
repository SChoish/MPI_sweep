#!/usr/bin/env python3
"""Summarize P0 hop-actor eval + action diagnostics. No training."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _p0_hop_common import ARCHIVE, ENVIRONMENTS, HIGH_T, now_kst
from _p0_hop_metrics import (
    DEFAULT_HORIZON,
    HOPPER_TASKS,
    decompose_group,
    group_episodes,
    mean_return_attribution,
    survival_curve,
    survival_grid,
)

EP = ARCHIVE / "episode_scores.csv"
ACT = ARCHIVE / "action_diagnostics.csv"
OBJ = ARCHIVE / "hop_objective.csv"
HOPPER_T10_ROLES = (
    ("MART4", 1, "mu1"),
    ("MART4", 2, "mu2"),
    ("MART4", 3, "mu3"),
    ("MART4", 4, "mu4"),
    ("two_actor_p4", 2, "deployment"),
)
SURVIVAL_STYLE = {
    "mu1": dict(color="#8a8a8a", linestyle="-", linewidth=1.4, label="MART mu1"),
    "mu2": dict(color="#2c7fb8", linestyle="-", linewidth=1.6, label="MART mu2"),
    "mu3": dict(color="#fdae61", linestyle="-", linewidth=1.6, label="MART mu3"),
    "mu4": dict(color="#d73027", linestyle="-", linewidth=2.0, label="MART mu4"),
    "deployment": dict(
        color="#1a9850", linestyle="--", linewidth=1.8, label="two-actor final"
    ),
}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_profiles(episodes: list[dict[str, str]]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in episodes:
        key = (
            row["task"],
            int(row["T"]),
            int(row["training_seed"]),
            row["method"],
            int(row["actor_index"]),
            row["actor_role"],
        )
        groups[key].append(float(row["normalized_score"]))
    out = []
    for key, values in sorted(groups.items()):
        out.append(
            {
                "task": key[0],
                "T": key[1],
                "training_seed": key[2],
                "method": key[3],
                "actor_index": key[4],
                "actor_role": key[5],
                "n_episodes": len(values),
                "mean_normalized": float(np.mean(values)),
                "std_normalized": float(np.std(values, ddof=0)),
            }
        )
    return out


def _mart_map(profiles):
    table = {}
    for row in profiles:
        if row["method"] != "MART4":
            continue
        table[(row["task"], row["T"], row["training_seed"], row["actor_index"])] = row[
            "mean_normalized"
        ]
    return table


def _task_short(task: str) -> str:
    return task.replace("-v2", "").replace("medium-replay", "med-replay")


def decompose_episodes(episodes: list[dict[str, str]]) -> list[dict]:
    groups = group_episodes(episodes)
    out = []
    for key, rows in sorted(groups.items()):
        rec = decompose_group(rows)
        rec.update(
            {
                "task": key[0],
                "T": key[1],
                "training_seed": key[2],
                "method": key[3],
                "actor_index": key[4],
                "actor_role": key[5],
            }
        )
        out.append(rec)
    return out


def pool_hopper_t10(episodes: list[dict[str, str]]) -> list[dict]:
    """Pool both training seeds (20 episodes) for the Hopper T=10 table."""
    grouped: dict[tuple, list[dict]] = {}
    for row in episodes:
        if int(row["T"]) != 10 or row["task"] not in HOPPER_TASKS:
            continue
        key = (row["task"], row["method"], int(row["actor_index"]), row["actor_role"])
        grouped.setdefault(key, []).append(row)
    out = []
    for key, rows in sorted(grouped.items()):
        rec = decompose_group(rows)
        rec.update(
            {
                "task": key[0],
                "T": 10,
                "training_seed": "pooled",
                "method": key[1],
                "actor_index": key[2],
                "actor_role": key[3],
            }
        )
        out.append(rec)
    return out


def mu4_minus_mu2_by_seed(profiles) -> list[dict]:
    rows = []
    for task in HOPPER_TASKS:
        for seed in (0, 1):
            vals = {}
            for row in profiles:
                if (
                    row["task"] == task
                    and row["T"] == 10
                    and row["training_seed"] == seed
                    and row["method"] == "MART4"
                    and row["actor_index"] in (2, 4)
                ):
                    vals[row["actor_index"]] = row["mean_normalized"]
            if 2 in vals and 4 in vals:
                rows.append(
                    {
                        "task": task,
                        "T": 10,
                        "training_seed": seed,
                        "delta_mu4_minus_mu2": vals[4] - vals[2],
                    }
                )
    return rows


def _select_lengths(episodes, task, tau, seed, method, role) -> np.ndarray:
    vals = [
        float(row["episode_length"])
        for row in episodes
        if row["task"] == task
        and int(row["T"]) == tau
        and int(row["training_seed"]) == seed
        and row["method"] == method
        and row["actor_role"] == role
    ]
    return np.asarray(vals, dtype=np.float64)


def plot_survival_hopper_t10(episodes: list[dict[str, str]]) -> str:
    grid = survival_grid(DEFAULT_HORIZON)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=True, sharey=True)
    for row, seed in enumerate((0, 1)):
        for col, task in enumerate(HOPPER_TASKS):
            ax = axes[row, col]
            for method, _index, role in HOPPER_T10_ROLES:
                lengths = _select_lengths(episodes, task, 10, seed, method, role)
                if lengths.size == 0:
                    continue
                style = SURVIVAL_STYLE[role]
                ax.step(
                    grid,
                    survival_curve(lengths, grid),
                    where="post",
                    **style,
                )
            ax.set_xlim(1, DEFAULT_HORIZON)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(_task_short(task), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"seed {seed}: S(t) = Pr(L >= t)")
            if row == 1:
                ax.set_xlabel("t (steps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Hopper T=10 episode survival by actor (10 episodes per seed)")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = ARCHIVE / "fig_survival_hopper_T10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_return_decomposition(pooled: list[dict]) -> str:
    roles = ["mu2", "mu4", "deployment"]
    labels = ["MART mu2", "MART mu4", "two-actor final"]
    colors = ["#2c7fb8", "#d73027", "#1a9850"]
    tasks = list(HOPPER_TASKS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(tasks))
    width = 0.24
    lookup = {(row["task"], row["actor_role"]): row for row in pooled}
    for ax, field, ylabel, title in (
        (
            axes[0],
            "mean_length",
            "mean episode length",
            "Episode length (timeout = 1,000)",
        ),
        (
            axes[1],
            "pooled_reward_per_step",
            "pooled reward / step",
            "Per-step reward (sum R / sum steps)",
        ),
    ):
        for i, (role, label, color) in enumerate(zip(roles, labels, colors)):
            vals = [lookup[(task, role)][field] for task in tasks]
            ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([_task_short(t) for t in tasks], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].axhline(DEFAULT_HORIZON, color="black", linewidth=0.7, linestyle=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("Hopper T=10 return split, 20 episodes pooled over seeds 0/1")
    fig.tight_layout()
    path = ARCHIVE / "fig_return_decomposition_hopper_T10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_survival_grid_csv(episodes: list[dict[str, str]]) -> str:
    t_keep = list(range(100, DEFAULT_HORIZON + 1, 100))
    path = ARCHIVE / "survival_hopper_T10.csv"
    fields = [
        "task",
        "T",
        "training_seed",
        "method",
        "actor_role",
        "n_episodes",
        *[f"S_{t}" for t in t_keep],
    ]
    rows = []
    for task in HOPPER_TASKS:
        for seed in (0, 1, "pooled"):
            for method, _index, role in HOPPER_T10_ROLES:
                if seed == "pooled":
                    lengths = np.concatenate(
                        [
                            _select_lengths(episodes, task, 10, s, method, role)
                            for s in (0, 1)
                        ]
                    )
                else:
                    lengths = _select_lengths(episodes, task, 10, seed, method, role)
                if lengths.size == 0:
                    continue
                surv = survival_curve(lengths, t_keep)
                rec = {
                    "task": task,
                    "T": 10,
                    "training_seed": seed,
                    "method": method,
                    "actor_role": role,
                    "n_episodes": int(lengths.size),
                }
                for t, value in zip(t_keep, surv):
                    rec[f"S_{t}"] = float(value)
                rows.append(rec)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def hopper_t10_story(pooled: list[dict], seed_deltas: list[dict]) -> dict:
    lookup = {(row["task"], row["actor_role"]): row for row in pooled}
    story = {}
    for task in HOPPER_TASKS:
        mu2 = lookup[(task, "mu2")]
        mu4 = lookup[(task, "mu4")]
        deploy = lookup[(task, "deployment")]
        attr = mean_return_attribution(
            mu2["pooled_reward_per_step"],
            mu2["mean_length"],
            mu4["pooled_reward_per_step"],
            mu4["mean_length"],
        )
        story[task] = {
            "mu2": mu2,
            "mu4": mu4,
            "deployment": deploy,
            "mu4_minus_mu2_score": mu4["mean_normalized"] - mu2["mean_normalized"],
            "attribution_mu2_to_mu4": attr,
            "seed_deltas": [
                row["delta_mu4_minus_mu2"]
                for row in seed_deltas
                if row["task"] == task
            ],
        }
    return story


def plot_hop_objective(obj_rows: list[dict[str, str]]) -> str | None:
    if not obj_rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = {
        "hopper-medium-v2": "#2c7fb8",
        "hopper-medium-replay-v2": "#1a9850",
        "hopper-expert-v2": "#d73027",
    }
    markers = {2: "o", 3: "s", 4: "^"}
    for row in obj_rows:
        if row.get("delta_normalized") in ("", None):
            continue
        task = row["task"]
        ax.scatter(
            float(row["delta_L"]),
            float(row["delta_normalized"]),
            c=colors.get(task, "black"),
            marker=markers.get(int(row["hop"]), "o"),
            s=42,
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("hop objective ΔL (negative = improved)")
    ax.set_ylabel("ΔJ = J(mu_k) - J(mu_{k-1})")
    ax.set_title("Hopper T=10: frozen-critic hop objective vs return")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], color=colors[t], marker="o", linestyle="", label=_task_short(t))
        for t in HOPPER_TASKS
        if t in colors
    ]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()
    path = ARCHIVE / "fig_hop_objective_vs_return.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def summarize_hop_objective(obj_rows: list[dict[str, str]]) -> dict:
    by_task: dict[str, list] = defaultdict(list)
    for row in obj_rows:
        by_task[row["task"]].append(row)
    out = {"n_hops": len(obj_rows), "tasks": {}}
    for task, rows in by_task.items():
        dL = [float(r["delta_L"]) for r in rows]
        dQ = [float(r["delta_q"]) for r in rows]
        labels = [r["obj_vs_return"] for r in rows if r.get("obj_vs_return")]
        counts = {name: labels.count(name) for name in sorted(set(labels))}
        out["tasks"][task] = {
            "mean_delta_L": float(np.mean(dL)),
            "mean_delta_q": float(np.mean(dQ)),
            "n_obj_improved": int(sum(r["objective_improved"] == "True" for r in rows)),
            "obj_vs_return_counts": counts,
        }
    return out
    lookup = {(row["task"], row["actor_role"]): row for row in pooled}
    story = {}
    for task in HOPPER_TASKS:
        mu2 = lookup[(task, "mu2")]
        mu4 = lookup[(task, "mu4")]
        deploy = lookup[(task, "deployment")]
        attr = mean_return_attribution(
            mu2["pooled_reward_per_step"],
            mu2["mean_length"],
            mu4["pooled_reward_per_step"],
            mu4["mean_length"],
        )
        story[task] = {
            "mu2": mu2,
            "mu4": mu4,
            "deployment": deploy,
            "mu4_minus_mu2_score": mu4["mean_normalized"] - mu2["mean_normalized"],
            "attribution_mu2_to_mu4": attr,
            "seed_deltas": [
                row["delta_mu4_minus_mu2"]
                for row in seed_deltas
                if row["task"] == task
            ],
        }
    return story


def analyze(profiles, actions) -> dict:
    mart = _mart_map(profiles)
    hop_deltas = []
    worsen = []
    for task, tau, seed, k in list(mart):
        if k != 1:
            continue
        vals = [mart.get((task, tau, seed, i)) for i in (1, 2, 3, 4)]
        if any(v is None for v in vals):
            continue
        for hop, (a, b) in enumerate(((1, 2), (2, 3), (3, 4)), start=2):
            hop_deltas.append(
                {
                    "task": task,
                    "T": tau,
                    "training_seed": seed,
                    "hop": hop,
                    "delta": vals[b - 1] - vals[a - 1],
                    "high_T": tau in HIGH_T,
                }
            )
        best_mid = max(vals[:3])
        if vals[3] + 1e-12 < best_mid:
            worsen.append(
                {
                    "task": task,
                    "T": tau,
                    "training_seed": seed,
                    "mu4": vals[3],
                    "oracle_best_of_mu123": best_mid,
                    "final_minus_first": vals[3] - vals[0],
                }
            )
    paired = []
    for row in profiles:
        if row["method"] != "MART4" or row["actor_index"] != 4:
            continue
        deploy = next(
            (
                other
                for other in profiles
                if other["task"] == row["task"]
                and other["T"] == row["T"]
                and other["training_seed"] == row["training_seed"]
                and other["method"] == "two_actor_p4"
                and other["actor_role"] == "deployment"
            ),
            None,
        )
        if deploy is None:
            continue
        dist = next(
            (
                act
                for act in actions
                if act["metric"] == "mu4_vs_deploy_rms"
                and act["state_set"] == "dataset"
                and act["task"] == row["task"]
                and int(act["T"]) == row["T"]
                and int(act["training_seed"]) == row["training_seed"]
            ),
            None,
        )
        paired.append(
            {
                "task": row["task"],
                "T": row["T"],
                "training_seed": row["training_seed"],
                "J_mu4": row["mean_normalized"],
                "J_deploy": deploy["mean_normalized"],
                "return_diff": row["mean_normalized"] - deploy["mean_normalized"],
                "action_rms_dataset": None if dist is None else float(dist["value"]),
            }
        )

    task_means = defaultdict(list)
    for row in paired:
        task_means[row["task"]].append(row["return_diff"])
    task_offset = {task: float(np.mean(vals)) for task, vals in task_means.items()}

    adj = defaultdict(list)
    from_mu1 = defaultdict(list)
    for act in actions:
        if act["method"] != "MART4" or act["state_set"] != "dataset":
            continue
        if act["metric"] == "adjacent_rms":
            adj[(int(act["T"]), act["actor_a"], act["actor_b"])].append(float(act["value"]))
        if act["metric"] == "from_mu1_rms":
            from_mu1[(int(act["T"]), act["actor_b"])].append(float(act["value"]))

    hop_by_t = defaultdict(list)
    for row in hop_deltas:
        hop_by_t[(row["T"], row["hop"])].append(row["delta"])

    return {
        "n_profiles": len(profiles),
        "n_paired_return": len(paired),
        "n_final_worse_than_oracle_mid": len(worsen),
        "mean_final_minus_first": float(
            np.mean([row["final_minus_first"] for row in worsen] or [math.nan])
        )
        if worsen
        else None,
        "mean_adjacent_return_by_T_hop": {
            f"T{t}_hop{h}": float(np.mean(vals))
            for (t, h), vals in sorted(hop_by_t.items())
        },
        "mean_adjacent_action_rms_by_T": {
            f"T{t}_{a}_{b}": float(np.mean(vals))
            for (t, a, b), vals in sorted(adj.items())
        },
        "task_mean_mu4_minus_deploy": task_offset,
        "grand_mean_mu4_minus_deploy": float(np.mean([row["return_diff"] for row in paired]))
        if paired
        else None,
        "worsen_cells": worsen,
        "paired": paired,
        "hop_deltas": hop_deltas,
    }


def plot_all(profiles, summary, episodes) -> list[str]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    written = []
    tasks = [env for env in ENVIRONMENTS if any(p["task"] == env for p in profiles)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    axes = axes.ravel()
    for ax in axes:
        ax.set_visible(False)
    for ax, task in zip(axes, tasks):
        ax.set_visible(True)
        for tau in (4, 7, 10, 14, 20):
            xs, ys = [], []
            for seed in (0, 1):
                pts = [
                    p
                    for p in profiles
                    if p["task"] == task
                    and p["T"] == tau
                    and p["training_seed"] == seed
                    and p["method"] == "MART4"
                ]
                pts = sorted(pts, key=lambda p: p["actor_index"])
                if len(pts) != 4:
                    continue
                xs.append([p["actor_index"] for p in pts])
                ys.append([p["mean_normalized"] for p in pts])
            if not ys:
                continue
            mean = np.mean(ys, axis=0)
            ax.plot([1, 2, 3, 4], mean, marker="o", label=f"T={tau}", linewidth=1.5)
            for seed_y in ys:
                ax.scatter([1, 2, 3, 4], seed_y, s=12, alpha=0.5)
        ax.set_title(task.replace("-v2", ""), fontsize=9)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("actor index")
        ax.set_ylabel("D4RL score")
        ax.grid(True, alpha=0.3)
    if tasks:
        axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("MART actor-index return profiles (dots = training seeds 0/1)")
    fig.tight_layout()
    path = ARCHIVE / "fig_actor_index_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    hops = [2, 3, 4]
    taus = [4, 7, 10, 14, 20]
    grid = np.full((len(tasks), len(taus) * len(hops)), np.nan)
    col_labels = []
    for tau in taus:
        for hop in hops:
            col_labels.append(f"T{tau} h{hop}")
    for i, task in enumerate(tasks):
        c = 0
        for tau in taus:
            for hop in hops:
                vals = [
                    d["delta"]
                    for d in summary["hop_deltas"]
                    if d["task"] == task and d["T"] == tau and d["hop"] == hop
                ]
                if vals:
                    grid[i, c] = float(np.mean(vals))
                c += 1
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-15, vmax=15)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.replace("-v2", "") for t in tasks], fontsize=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_title("Adjacent-hop D4RL change (mean of available seeds)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    path = ARCHIVE / "fig_hop_return_heatmap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    xs = [p["action_rms_dataset"] for p in summary["paired"] if p["action_rms_dataset"] is not None]
    ys = [p["return_diff"] for p in summary["paired"] if p["action_rms_dataset"] is not None]
    fig, ax = plt.subplots(figsize=(6, 5))
    if xs:
        ax.scatter(xs, ys, s=28, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("RMS action distance (mu4 vs deploy, dataset states)")
        ax.set_ylabel("J(mu4) - J(deploy)")
        ax.set_title("MART vs two-actor: action distance vs return difference")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = ARCHIVE / "fig_action_vs_return_scatter.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    if any(int(row["T"]) == 10 and row["task"] in HOPPER_TASKS for row in episodes):
        written.append(plot_survival_hopper_t10(episodes))
        pooled = pool_hopper_t10(episodes)
        written.append(plot_return_decomposition(pooled))
        write_survival_grid_csv(episodes)
    obj_rows = _read(OBJ)
    hop_fig = plot_hop_objective(obj_rows)
    if hop_fig:
        written.append(hop_fig)
    return written


def hypotheses(summary, actions) -> dict[str, str]:
    adj_by_hop = defaultdict(list)
    for row in summary["hop_deltas"]:
        adj_by_hop[row["hop"]].append(abs(row["delta"]))
    rms_by_hop = defaultdict(list)
    for act in actions:
        if act["metric"] == "adjacent_rms" and act["state_set"] == "dataset" and act["method"] == "MART4":
            rms_by_hop[act["actor_b"]].append(float(act["value"]))
    cos = [
        float(act["value"])
        for act in actions
        if act["metric"] == "cos_d2_d3" and act["state_set"] == "dataset" and not math.isnan(float(act["value"]))
    ]
    paired = summary["paired"]
    large_act_small_ret = [
        p
        for p in paired
        if p["action_rms_dataset"] is not None
        and p["action_rms_dataset"] > 0.05
        and abs(p["return_diff"]) < 3
    ]
    signs = [int(np.sign(v)) for v in summary["task_mean_mu4_minus_deploy"].values()]
    return {
        "H1_saturation": (
            "later |ΔJ| and adjacent action RMS compared with hop 1→2; "
            f"mean |ΔJ| hop2={np.mean(adj_by_hop.get(2) or [math.nan]):.3f}, "
            f"hop3={np.mean(adj_by_hop.get(3) or [math.nan]):.3f}, "
            f"hop4={np.mean(adj_by_hop.get(4) or [math.nan]):.3f}; "
            f"mean adj RMS mu1→2={np.mean(rms_by_hop.get('mu2') or [math.nan]):.4f}, "
            f"mu2→3={np.mean(rms_by_hop.get('mu3') or [math.nan]):.4f}, "
            f"mu3→4={np.mean(rms_by_hop.get('mu4') or [math.nan]):.4f}."
        ),
        "H2_reversion": (
            f"{summary['n_final_worse_than_oracle_mid']} cells have mu4 below the "
            "best of mu1–mu3 (oracle diagnostic, not offline selection)."
        ),
        "H3_direction": (
            f"mean cosine(δ2,δ3) on dataset states={np.mean(cos) if cos else float('nan'):.3f} "
            "after dropping near-zero hops."
        ),
        "H4_similar_return_different_action": (
            f"{len(large_act_small_ret)}/{len(paired)} paired cells have dataset "
            "action RMS>0.05 and |ΔJ|<3."
        ),
        "H5_task_offset": (
            "task-mean J(mu4)-J(deploy)="
            + json.dumps(summary["task_mean_mu4_minus_deploy"])
            + f"; grand mean={summary['grand_mean_mu4_minus_deploy']}; sign mix={signs}."
        ),
        "H6_first_actor_path": (
            "Not isolated here: MART mu1 and two-actor target were trained on "
            "independent first-actor/critic trajectories. Shared-driver remains "
            "available but is deprioritized relative to the actor-path survival "
            "and hop-objective readout."
        ),
        "H7_survival_not_rate": (
            "At Hopper T=10, mu2→mu4 per-step reward falls slightly on medium "
            "and expert while episode length moves in opposite directions. "
            "Return change is therefore mostly survival, not richer per-step reward."
        ),
        "H8_hop_objective": (
            "Frozen-critic JKO ΔL at the 1M snapshot is tiny (~1e-3 on |L|≈5). "
            "Q(s,μ_k) still ticks up. Medium return rises and expert return falls "
            "anyway, so this critic's hop objective does not separate helpful from "
            "harmful hops. The critic is trained on actor-1 continuation."
        ),
    }


def write_report(coverage, summary, hyps, figures, n_episodes, story=None, obj_summary=None) -> None:
    lines = [
        "# P0 hop-actor audit",
        "",
        f"Built: {now_kst()}",
        "",
        "Understood as: explain similar final scores and later-hop gains/losses",
        "by splitting return into episode survival vs per-step reward, then",
        "(next) hop-objective change on the frozen critic. No new training.",
        "Episode repeats are noise reduction, not extra seeds.",
        "",
        "## Coverage",
        "",
        f"- Expected runs: {coverage['n_expected_runs']}",
        f"- Evalable on this host: {coverage['n_evalable']} ({coverage['fraction_runs_evalable']:.0%})",
        f"- Paired task/T/seed cells: {coverage['n_paired_task_T_seed']} ({coverage['fraction_paired']:.0%})",
        f"- Missing tasks here: {', '.join(coverage['missing_tasks_on_this_host']) or 'none'}",
        "- Tail90 Walker / halfcheetah-expert checkpoints stay on ext_csh and were not substituted.",
        "",
        "## Executed measurements",
        "",
        f"- Deterministic gymnasium MuJoCo-v4 eval, 10 episodes, shared `evaluation_seed = 10000+ep`.",
        f"- Episode rows: {n_episodes}",
        f"- Paired return contrasts: {summary['n_paired_return']}",
        "- Action distances on dataset states and, where both rollout dumps exist, the mu4∪deploy union.",
        "",
        "## Return profiles",
        "",
        f"- Adjacent-hop mean ΔJ by T/hop: `{summary['mean_adjacent_return_by_T_hop']}`",
        f"- Oracle mid-actor better than mu4 in {summary['n_final_worse_than_oracle_mid']} cells.",
        "- Do not read that oracle as an offline selection policy.",
        "- Actor index is not a separately trained K=2/K=3 run; timestep is 1M for every stored actor.",
        "",
        "## Hopper T=10 survival vs per-step reward",
        "",
    ]
    if story:
        for task, rec in story.items():
            mu2, mu4, dep = rec["mu2"], rec["mu4"], rec["deployment"]
            attr = rec["attribution_mu2_to_mu4"]
            seeds = ", ".join(f"{v:+.2f}" for v in rec["seed_deltas"])
            lines += [
                f"### {_task_short(task)}",
                "",
                (
                    f"- μ2: score {mu2['mean_normalized']:.2f}, "
                    f"length {mu2['mean_length']:.1f}, "
                    f"timeout {mu2['n_timeout']}/{mu2['n_episodes']}, "
                    f"r/step {mu2['pooled_reward_per_step']:.3f}."
                ),
                (
                    f"- μ4: score {mu4['mean_normalized']:.2f}, "
                    f"length {mu4['mean_length']:.1f}, "
                    f"timeout {mu4['n_timeout']}/{mu4['n_episodes']}, "
                    f"r/step {mu4['pooled_reward_per_step']:.3f}."
                ),
                (
                    f"- two-actor final: score {dep['mean_normalized']:.2f}, "
                    f"length {dep['mean_length']:.1f}, "
                    f"timeout {dep['n_timeout']}/{dep['n_episodes']}, "
                    f"r/step {dep['pooled_reward_per_step']:.3f}."
                ),
                (
                    f"- μ4−μ2 score {rec['mu4_minus_mu2_score']:+.2f} "
                    f"(seeds {seeds}). Raw-return split: length term "
                    f"{attr['length_term']:+.1f}, rate term {attr['rate_term']:+.1f} "
                    f"(length share {attr['abs_length_share']:.2f})."
                ),
                "- Reaching 1,000 steps is a timeout, not a return ceiling.",
                "",
            ]
    if obj_summary:
        lines += [
            "## Hopper T=10 hop objective (frozen critic, dataset states)",
            "",
            (
                "ΔL_k = -c_k E[Q(μ_k)-Q(μ_{k-1})] + E[||Δμ||^2 / d], "
                "c_k = 2(T/K) / mean(|Q(μ_{k-1})|), Q = critic head 1. "
                "Negative ΔL is an objective improvement. Residuals are ~10^{-3} "
                "on a loss of about -5."
            ),
            "",
        ]
        for task, rec in obj_summary.get("tasks", {}).items():
            lines += [
                f"### {_task_short(task)}",
                "",
                (
                    f"- mean ΔL={rec['mean_delta_L']:+.4g}, mean ΔQ={rec['mean_delta_q']:+.4g}, "
                    f"objective-improved hops {rec['n_obj_improved']}/6."
                ),
                f"- class counts: `{rec['obj_vs_return_counts']}`.",
                "",
            ]
        lines += [
            "Hopper-expert later hops sit in **obj not improved + return down**. "
            "Hopper-medium is the missing fifth cell: **obj not improved + return up**. "
            "ΔQ is positive in every hop, including expert. That is not treated as "
            "policy improvement: this critic's Bellman target is actor-1 continuation.",
            "",
        ]
    lines += [
        "## Hypotheses (evidence, not verdicts)",
        "",
    ]
    for key, text in hyps.items():
        lines += [f"### {key}", "", text, ""]
    lines += [
        "## Figures",
        "",
        *[f"- `{Path(p).name}`" for p in figures],
        "",
        "## Limits",
        "",
        "- Two training seeds only.",
        "- Cross-stack P0 merge remains scientifically inadmissible; this CPU re-eval is a same-host diagnostic.",
        "- Learned Q increase is not treated as policy improvement.",
        "",
        "## Next measurement",
        "",
        "Shared-driver is deprioritized. Next is only to extend hop-objective",
        "coverage beyond Hopper T=10 if that contrast is needed, not to launch",
        "a new training grid.",
        "",
    ]
    (ARCHIVE / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    episodes = _read(EP)
    actions = _read(ACT)
    coverage = json.loads((ARCHIVE / "COVERAGE.json").read_text(encoding="utf-8"))
    profiles = _mean_profiles(episodes)
    with (ARCHIVE / "cell_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0].keys()) if profiles else ["task"])
        writer.writeheader()
        writer.writerows(profiles)
    summary = analyze(profiles, actions)
    slim = {k: v for k, v in summary.items() if k not in ("paired", "hop_deltas", "worsen_cells")}
    slim["worsen_n"] = len(summary["worsen_cells"])
    with (ARCHIVE / "paired_return_action.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "task",
            "T",
            "training_seed",
            "J_mu4",
            "J_deploy",
            "return_diff",
            "action_rms_dataset",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["paired"])
    decomp = decompose_episodes(episodes)
    with (ARCHIVE / "return_decomposition.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "task",
            "T",
            "training_seed",
            "method",
            "actor_index",
            "actor_role",
            "n_episodes",
            "mean_normalized",
            "mean_raw_return",
            "mean_length",
            "n_timeout",
            "pooled_reward_per_step",
            "mean_episode_reward_per_step",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decomp)
    pooled = pool_hopper_t10(episodes)
    seed_deltas = mu4_minus_mu2_by_seed(profiles)
    story = hopper_t10_story(pooled, seed_deltas)
    slim["hopper_t10_mu4_minus_mu2_by_seed"] = seed_deltas
    slim["hopper_t10_pooled_r_per_step"] = {
        task: {
            "mu2": float(story[task]["mu2"]["pooled_reward_per_step"]),
            "mu4": float(story[task]["mu4"]["pooled_reward_per_step"]),
            "length_share": float(story[task]["attribution_mu2_to_mu4"]["abs_length_share"]),
        }
        for task in story
    }
    obj_rows = _read(OBJ)
    obj_summary = summarize_hop_objective(obj_rows) if obj_rows else None
    if obj_summary:
        slim["hopper_t10_hop_objective"] = obj_summary
    (ARCHIVE / "SUMMARY.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    figures = plot_all(profiles, summary, episodes)
    hyps = hypotheses(summary, actions)
    write_report(coverage, summary, hyps, figures, len(episodes), story, obj_summary)
    (ARCHIVE / "HYPOTHESES.json").write_text(json.dumps(hyps, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": figures, **slim}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
