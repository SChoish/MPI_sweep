#!/usr/bin/env python3
"""Episode survival curves and return = length × reward-rate.

Uses existing episode CSVs. Does not train. First90 Hopper rows come from
commit 0c8c50cd (ext_csv); tail90 rows are local. Protocols are not mixed
inside one curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.diagnostics.p0_survival import (  # noqa: E402
    MAX_EPISODE_STEPS,
    SURVIVAL_TIMES,
    classify_hop,
    decompose_episodes,
    survival_curve,
)
from scripts.diagnostics.p0_intermediate_actors import write_csv, write_json  # noqa: E402

FIRST90 = ROOT / "sweep_results/diagnostics/p0_hop_actor_audit/episode_scores.csv"
TAIL90 = ROOT / "sweep_results/diagnostics/p0_intermediate_actors/episodes.csv"
OUT = ROOT / "sweep_results/diagnostics/p0_episode_survival"
OBJECTIVES = ROOT / "sweep_results/diagnostics/p0_intermediate_actors/hop_objectives.csv"

HOPPER_TASKS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
)
ROLE_MAP_FIRST90 = {
    "mu1": "mart_mu1",
    "mu2": "mart_mu2",
    "mu3": "mart_mu3",
    "mu4": "mart_mu4",
    "target_value": "two_actor_target",
    "deployment": "two_actor_deploy",
}
CURVE_SPECS = (
    ("mart_mu1", "MART μ1", "0.55"),
    ("mart_mu2", "MART μ2", "C0"),
    ("mart_mu3", "MART μ3", "C1"),
    ("mart_mu4", "MART μ4", "C3"),
    ("two_actor_deploy", "two-actor deploy", "C2"),
)


def _load_first90(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "source": "first90_ext_csv_0c8c50cd",
                    "environment": raw["task"],
                    "T": int(raw["T"]),
                    "seed": int(raw["training_seed"]),
                    "condition": "bar_p4" if raw["method"] == "MART4" else "two_actor_p4",
                    "actor_id": ROLE_MAP_FIRST90[raw["actor_role"]],
                    "eval_seed": int(raw["evaluation_seed"]),
                    "raw_return": float(raw["raw_return"]),
                    "normalized_score": float(raw["normalized_score"]),
                    "episode_length": int(float(raw["episode_length"])),
                    "n_eval_protocol": 10,
                }
            )
    return rows


def _load_tail90(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "source": "tail90_ext_csh",
                    "environment": raw["environment"],
                    "T": int(raw["T"]),
                    "seed": int(raw["seed"]),
                    "condition": raw["condition"],
                    "actor_id": raw["actor_id"],
                    "eval_seed": int(raw["eval_seed"]),
                    "raw_return": float(raw["raw_return"]),
                    "normalized_score": float(raw["normalized_score"]),
                    "episode_length": int(float(raw["episode_length"])),
                    "n_eval_protocol": 10,
                }
            )
    return rows


def policy_groups(rows: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["source"],
            row["environment"],
            row["T"],
            row["seed"],
            row["actor_id"],
        )
        groups[key].append(row)
    return groups


def policy_table(groups: Mapping[tuple, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for key, items in sorted(groups.items()):
        source, env, tau, seed, actor_id = key
        stats = decompose_episodes(items)
        out.append(
            {
                "source": source,
                "environment": env,
                "T": tau,
                "seed": seed,
                "actor_id": actor_id,
                **stats,
            }
        )
    return out


def plot_hopper_t10(rows: list[dict[str, Any]], out_dir: Path) -> None:
    hopper = [r for r in rows if r["source"].startswith("first90") and r["T"] == 10]
    groups = policy_groups(hopper)
    times = np.asarray(SURVIVAL_TIMES)
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.2), sharex=True, sharey=True)
    for i, task in enumerate(HOPPER_TASKS):
        for j, seed in enumerate((0, 1)):
            ax = axes[i][j]
            for actor_id, label, color in CURVE_SPECS:
                items = groups.get(("first90_ext_csv_0c8c50cd", task, 10, seed, actor_id), [])
                if not items:
                    continue
                lengths = [r["episode_length"] for r in items]
                ax.step(times, survival_curve(lengths, times), where="post", label=label, color=color, lw=1.6)
            ax.set_title(f"{task}  seed {seed}")
            ax.set_xlim(0, MAX_EPISODE_STEPS)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.25)
            if i == 2:
                ax.set_xlabel("t (env steps)")
            if j == 0:
                ax.set_ylabel(r"$S_k(t)=\Pr(L_k \geq t)$")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Hopper T=10 episode survival by actor (10 episodes/seed; seeds not pooled)")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "hopper_t10_survival.png", dpi=140)
    plt.close(fig)


def plot_timeout_bars(table: list[dict[str, Any]], out_dir: Path) -> None:
    want_actors = [spec[0] for spec in CURVE_SPECS]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    x = np.arange(len(want_actors))
    for ax, task in zip(axes, HOPPER_TASKS):
        w = 0.35
        for offset, seed, hatch in ((-w / 2, 0, None), (w / 2, 1, "//")):
            rates = []
            for actor_id in want_actors:
                rows = [
                    r
                    for r in table
                    if r["environment"] == task
                    and r["T"] == 10
                    and r["seed"] == seed
                    and r["actor_id"] == actor_id
                    and str(r["source"]).startswith("first90")
                ]
                rates.append(rows[0]["timeout_rate"] if rows else np.nan)
            ax.bar(x + offset, rates, width=w, label=f"seed {seed}", hatch=hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(["μ1", "μ2", "μ3", "μ4", "2A"], fontsize=8)
        ax.set_title(task.replace("-v2", ""))
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="0.8", lw=0.6)
    axes[0].set_ylabel("Pr(L = 1000)")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Hopper T=10: fraction of episodes that reach the 1000-step time limit")
    fig.tight_layout()
    fig.savefig(out_dir / "hopper_t10_timeout_rate.png", dpi=140)
    plt.close(fig)


def join_objectives(
    policy_rows: list[dict[str, Any]], objective_path: Path
) -> list[dict[str, Any]]:
    if not objective_path.is_file():
        return []
    with objective_path.open(newline="", encoding="utf-8") as handle:
        obj_rows = list(csv.DictReader(handle))
    score = {}
    for row in policy_rows:
        if row["source"] != "tail90_ext_csh":
            continue
        score[(row["environment"], row["T"], row["seed"], row["actor_id"])] = row["normalized_score_mean"]
    out = []
    for row in obj_rows:
        if row["condition"] != "bar_p4" or int(row["hop_index"]) < 2:
            continue
        if row["state_set"] != "dataset":
            continue
        env, tau, seed = row["environment"], int(row["T"]), int(row["seed"])
        new_id, ref_id = row["new_actor"], row["ref_actor"]
        j_new = score.get((env, tau, seed, new_id))
        j_ref = score.get((env, tau, seed, ref_id))
        if j_new is None or j_ref is None:
            continue
        dL = float(row["delta_L"])
        dJ = float(j_new) - float(j_ref)
        out.append(
            {
                "environment": env,
                "T": tau,
                "seed": seed,
                "hop_index": int(row["hop_index"]),
                "ref_actor": ref_id,
                "new_actor": new_id,
                "delta_L": dL,
                "q_term": float(row["q_term"]),
                "transport_mean_sq": float(row["transport_mean_sq"]),
                "mean_q_gain": float(row["mean_q_gain"]),
                "return_delta": dJ,
                "j_ref": j_ref,
                "j_new": j_new,
                "label": classify_hop(dL, dJ),
                "optimizer_diverged": row["optimizer_diverged"],
                "c_k": float(row["c_k"]),
            }
        )
    return out


def write_report(
    out_dir: Path,
    hopper_table: list[dict[str, Any]],
    joined: list[dict[str, Any]],
) -> None:
    def pick(task: str, actor: str) -> list[dict[str, Any]]:
        return [
            r
            for r in hopper_table
            if r["environment"] == task and r["T"] == 10 and r["actor_id"] == actor
        ]

    lines = [
        "# Episode survival and hop objectives",
        "",
        "Return is split into how long the episode lasts and how much reward",
        "arrives per step. Survival curves use stored `episode_length` only.",
        "Hopper T=10 uses the ext_csv first90 checkpoints (training seeds 0",
        "and 1 of those Hopper cells). Tail90 hop objectives use a different",
        "checkpoint shard on this host; the same integer seed label is not the",
        "same run. Do not pool episode scores across shards. Tail90 hop",
        "objectives use local frozen critics; Hopper checkpoints are not here.",
        "",
        "## Hopper T=10 pooled two-seed table (20 episodes = 2×10)",
        "",
        "| Task | Actor | Score | Mean L | Timeouts | Reward / step |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task in HOPPER_TASKS:
        for actor, name in (
            ("mart_mu2", "MART μ2"),
            ("mart_mu4", "MART μ4"),
            ("two_actor_deploy", "two-actor deploy"),
        ):
            rows = pick(task, actor)
            if not rows:
                continue
            n = sum(r["n_episodes"] for r in rows)
            score = np.average([r["normalized_score_mean"] for r in rows], weights=[r["n_episodes"] for r in rows])
            length = np.average([r["episode_length_mean"] for r in rows], weights=[r["n_episodes"] for r in rows])
            timeouts = sum(r["n_timeout"] for r in rows)
            # pooled rps across seeds: re-weight by n * length
            rps = np.average(
                [r["reward_per_step_pooled"] for r in rows],
                weights=[r["n_episodes"] * r["episode_length_mean"] for r in rows],
            )
            lines.append(
                f"| {task} | {name} | {score:.2f} | {length:.0f} | {timeouts}/{n} | {rps:.3f} |"
            )
    lines += [
        "",
        "Hopper-medium μ2→μ4: reward/step 3.305→3.258 but L 653→1000.",
        "Hopper-expert μ2→μ4: reward/step 3.675→3.629 but L 875→457.",
        "The score change is an episode-survival change, not a per-step bonus.",
        "Reaching 1000 steps is not a claim that return is at its task maximum.",
        "",
        "Per-seed MART μ4−μ2 at T=10: Hopper-medium +36.95 / +30.64;",
        "Hopper-expert −31.29 / −64.48. Not a one-seed artifact.",
        "",
        "The survival curves (seeds not pooled) show where the lifetime change",
        "happens, and that it is cohort-wide rather than a few failed episodes:",
        "",
        "- Hopper-medium: μ3 converts most episodes to timeouts (seed 0: 1/10→6/10;",
        "  seed 1: 0/10→9/10); μ4 finishes 10/10. Two-actor deploy matches μ4.",
        "- Hopper-medium-replay: μ1 still fails early; μ2 already 10/10 at 1000.",
        "  Later hops cannot add lifetime.",
        "- Hopper-expert seed 0: timeouts 7/10→7/10→4/10→0/10; μ4 lengths 472–815.",
        "  Seed 1: μ3 already 9/10 below 400 steps; μ4 mean L=294. Two-actor",
        "  deploy stays with μ1/μ2.",
        "",
        "## Hop objective ΔL on local tail90 (dataset states)",
        "",
    ]
    if not joined:
        lines.append("`hop_objectives.csv` not present yet. Run `run_p0_intermediate_actors.py objectives`.")
    else:
        counts: dict[str, int] = defaultdict(int)
        for row in joined:
            counts[row["label"]] += 1
        n = len(joined)
        n_q_up = sum(1 for r in joined if r["mean_q_gain"] > 0)
        n_obj_up = sum(1 for r in joined if r["delta_L"] < 0)
        n_div = sum(1 for r in joined if str(r["optimizer_diverged"]).lower() in ("true", "1"))
        lines += [
            "MART hops k=2,3,4 vs predecessor, local 5-task tail90 shard, dataset",
            f"states, frozen final critic. {n} hops; {n_div} hops",
            "carry a nonfinite critic optimizer and are kept but not averaged.",
            "",
            f"Q(μ_k)−Q(μ_{{k-1}}) is positive on {n_q_up}/{n} hops, but the hop",
            f"loss ΔL is negative on only {n_obj_up}/{n}. Transport usually",
            "exceeds the scaled Q term, especially after hop 2:",
            "",
        ]
        for hop in (2, 3, 4):
            sub = [r for r in joined if r["hop_index"] == hop]
            n_neg = sum(1 for r in sub if r["delta_L"] < 0)
            lines.append(f"- hop {hop}: ΔL<0 on {n_neg}/{len(sub)}")
        lines += [
            "",
            "Labels (loss decrease = objective improvement; |ΔJ|>1 = return change):",
            "",
        ]
        for key, n_lab in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{key}`: {n_lab}")
        special = [
            r
            for r in joined
            if r["environment"] == "walker2d-medium-v2"
            and r["T"] == 14
            and r["seed"] == 0
            and r["hop_index"] == 4
        ]
        wm = [
            r
            for r in hopper_table
            if r["source"] == "tail90_ext_csh"
            and r["environment"] == "walker2d-medium-v2"
            and r["T"] == 14
            and r["seed"] == 0
        ]
        wm_by = {r["actor_id"]: r for r in wm}
        if special:
            r = special[0]
            mu3 = wm_by.get("mart_mu3", {})
            mu4 = wm_by.get("mart_mu4", {})
            lines += [
                "",
                "Walker-medium T=14 seed 0 is the local survival-collapse analog of",
                "Hopper-expert. Hop 2 is `objective_up_return_up`. Hop 4 is",
                f"`{r['label']}`: ΔL={r['delta_L']:.4g}, mean ΔQ={r['mean_q_gain']:.3f},",
                f"ΔJ={r['return_delta']:.1f}. Episode length "
                f"{mu3.get('episode_length_mean', float('nan')):.0f}→"
                f"{mu4.get('episode_length_mean', float('nan')):.0f} "
                f"({mu3.get('n_timeout', '?')}/10→{mu4.get('n_timeout', '?')}/10 "
                "timeouts). Q still rose; the hop loss did not improve because",
                "transport exceeded c_k ΔQ. This is objective-not-up / return-down,",
                "not automatic critic error.",
            ]
        hc = [
            r
            for r in hopper_table
            if r["source"] == "tail90_ext_csh"
            and r["environment"] == "halfcheetah-expert-v2"
            and r["T"] == 20
            and r["seed"] == 0
            and r["actor_id"] in ("mart_mu1", "mart_mu4")
        ]
        if len(hc) == 2:
            by = {r["actor_id"]: r for r in hc}
            lines += [
                "",
                "HalfCheetah-expert is a different failure mode. At T=20 seed 0,",
                f"μ1 and μ4 both timeout {by['mart_mu1']['n_timeout']}/10 and "
                f"{by['mart_mu4']['n_timeout']}/10; reward/step "
                f"{by['mart_mu1']['reward_per_step_pooled']:.3f}→"
                f"{by['mart_mu4']['reward_per_step_pooled']:.3f}. Return drops",
                "without early termination. Do not treat every harmful hop as a",
                "Hopper-style survival failure.",
            ]
        lines += [
            "",
            "Hopper-expert ΔL is **not** computed here (checkpoints stay on ext_csv).",
            "Its return drop is already a survival drop. Seed 0 μ4 lengths are all",
            "in 472–815 (0/10 timeouts); seed 1 μ3 already puts 9/10 episodes",
            "below 400 steps. Q-up at a frozen first-actor critic would still not",
            "by itself prove critic error, because backups follow Polyak μ1",
            "continuation rather than μ4 rollouts.",
        ]
    lines += [
        "",
        "## Next",
        "",
        "Shared-driver is deprioritized. The manuscript question is how extra hops",
        "change episode survival, not whether two methods match under a shared critic.",
        "First no-train follow-up remains hop-objective coverage on Hopper first90",
        "checkpoints when that host is available. MC continuation only where ΔL",
        "improved and lifetime collapsed (local: walker-medium T=20 seed 0 hops",
        "2–3). Walker-medium T=14 seed 0 hop 4 is the wrong cell for that test:",
        "its hop loss did not improve.",
        "",
    ]
    (out_dir / "SURVIVAL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first90-episodes", type=Path, default=FIRST90)
    parser.add_argument("--tail90-episodes", type=Path, default=TAIL90)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--objectives", type=Path, default=OBJECTIVES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    if args.first90_episodes.is_file():
        rows.extend(_load_first90(args.first90_episodes))
    if args.tail90_episodes.is_file():
        rows.extend(_load_tail90(args.tail90_episodes))
    groups = policy_groups(rows)
    table = policy_table(groups)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "policy_length_reward.csv",
        table,
        [
            "source",
            "environment",
            "T",
            "seed",
            "actor_id",
            "n_episodes",
            "normalized_score_mean",
            "raw_return_mean",
            "episode_length_mean",
            "n_timeout",
            "timeout_rate",
            "reward_per_step_pooled",
            "reward_per_step_episode_mean",
        ],
    )
    plot_hopper_t10(rows, args.out_dir)
    plot_timeout_bars(table, args.out_dir)
    joined = join_objectives(table, args.objectives)
    if joined:
        write_csv(
            args.out_dir / "hop_objective_vs_return.csv",
            joined,
            [
                "environment",
                "T",
                "seed",
                "hop_index",
                "ref_actor",
                "new_actor",
                "delta_L",
                "q_term",
                "transport_mean_sq",
                "mean_q_gain",
                "return_delta",
                "j_ref",
                "j_new",
                "label",
                "optimizer_diverged",
                "c_k",
            ],
        )
        counts: dict[str, int] = defaultdict(int)
        for row in joined:
            counts[row["label"]] += 1
        write_json(args.out_dir / "OBJECTIVE_LABELS.json", dict(counts))
    write_report(args.out_dir, table, joined)
    print(json.dumps({"n_policy_rows": len(table), "n_objective_joins": len(joined)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
