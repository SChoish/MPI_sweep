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

EP = ARCHIVE / "episode_scores.csv"
ACT = ARCHIVE / "action_diagnostics.csv"


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


def plot_all(profiles, summary) -> list[str]:
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
            "independent first-actor/critic trajectories. Shared-driver P0.B remains "
            "the experiment that holds that path fixed."
        ),
    }


def write_report(coverage, summary, hyps, figures, n_episodes) -> None:
    lines = [
        "# P0 hop-actor audit",
        "",
        f"Built: {now_kst()}",
        "",
        "Understood as: explain similar MART vs two-actor *final* scores by measuring",
        "each hop's return and action change on existing P0 K=4 checkpoints.",
        "No new training. Episode repeats are noise reduction, not extra seeds.",
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
        "## Next experiment",
        "",
        "P0.B shared-driver (`recenter` minus `data_anchor`) remains the first training experiment,",
        "because H6 is unresolved and hop diagnostics cannot hold the first actor/critic path fixed.",
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
    (ARCHIVE / "SUMMARY.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
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
    figures = plot_all(profiles, summary)
    hyps = hypotheses(summary, actions)
    write_report(coverage, summary, hyps, figures, len(episodes))
    (ARCHIVE / "HYPOTHESES.json").write_text(json.dumps(hyps, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": figures, **slim}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
