#!/usr/bin/env python3
"""CPU 2×2: first action × Polyak continuation on Hopper T=10 P0 checkpoints.

Understood as: on existing first90 1M snapshots, swap only the first action
then follow MART Polyak μ1 or two-actor Polyak target. Fills the missing
two-actor cells of the hop-2/3 continuation table. No training. No GPU.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import multiprocessing as mp
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
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import REPO_ROOT as ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _ckpt_compat import normalize_checkpoint  # noqa: E402
from _p0_first_action_continuation import (  # noqa: E402
    BASELINE_FIRST,
    CONT_MART_POLYAK,
    CONT_TWO_POLYAK,
    DISCOUNT,
    EVAL_SEED_BASE,
    FIRST_MART_HOP,
    FIRST_TWO_DEPLOY,
    HOPPER_T10_TASKS,
    HOPS,
    HORIZON,
    baseline_selector,
    discounted_return,
    eval_kinds_for_cell,
    hybrid_cells,
    mean_score,
    paired_delta,
    ranking_agreement,
    row_matches,
    table_row_selector,
)
from _p0_hop_common import ARCHIVE as HOP_ARCHIVE, now_kst  # noqa: E402
from train_td3bc import (  # noqa: E402
    EVAL_ENV,
    Actor,
    TwinCritic,
    _domain,
    d4rl_normalized_score,
    load_checkpoint,
)

OUT = ROOT / "sweep_results/diagnostics/p0_first_action_continuation"
INVENTORY = HOP_ARCHIVE / "INVENTORY.json"
FULL_EPISODES = HOP_ARCHIVE / "episode_scores.csv"
FIELDS = (
    "task",
    "T",
    "training_seed",
    "kind",
    "hop",
    "first_source",
    "continuation_source",
    "evaluation_seed",
    "raw_return",
    "discounted_return",
    "normalized_score",
    "episode_length",
    "terminated",
    "truncated",
    "q_mart_first",
    "q_mart_cont_action",
    "delta_q_mart",
    "q_two_first",
    "q_two_cont_action",
    "delta_q_two",
    "first_action_rms",
    "checkpoint_hash_mart",
    "checkpoint_hash_two",
)


def _infer_action_dim(params: Any, mean: np.ndarray) -> int:
    kernel = np.asarray(params["params"]["Dense_0"]["kernel"])
    output_kernel = np.asarray(params["params"]["Dense_2"]["kernel"])
    if kernel.shape[0] != mean.shape[0]:
        raise ValueError("actor input dim does not match checkpoint mean")
    return int(output_kernel.shape[-1])


def _actor_params(payload: dict[str, Any], actor_index: int):
    actors = payload.get("actors_params")
    if actors is not None:
        return actors[actor_index - 1]
    normalized = normalize_checkpoint(payload)
    key = "actor_params" if actor_index == 1 else f"actor{actor_index}_params"
    return normalized[key]


def _policy(apply, params, mean: np.ndarray, std: np.ndarray):
    def policy(raw: np.ndarray) -> np.ndarray:
        x = (np.asarray(raw, dtype=np.float32) - mean) / std
        if x.ndim == 1:
            x = x[None, :]
            return np.asarray(apply(params, jnp.asarray(x))[0], dtype=np.float32)
        return np.asarray(apply(params, jnp.asarray(x)), dtype=np.float32)

    return policy


def _q1(q_apply, critic_params, raw: np.ndarray, mean: np.ndarray, std: np.ndarray, action: np.ndarray) -> float:
    q1, _ = q_apply(
        critic_params,
        jnp.asarray(((raw - mean) / std)[None, :], dtype=jnp.float32),
        jnp.asarray(action[None, :], dtype=jnp.float32),
    )
    return float(np.asarray(q1)[0, 0])


def paired_inventory(tasks: tuple[str, ...], tau: int, seeds: list[int]) -> list[dict[str, Any]]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_key = {}
    for row in inventory["rows"]:
        if (
            row.get("status") == "evalable"
            and row.get("environment") in tasks
            and int(row["T"]) == int(tau)
            and int(row["seed"]) in set(seeds)
        ):
            by_key[(row["environment"], int(row["seed"]), row["condition"])] = row
    pairs = []
    for task in tasks:
        for seed in seeds:
            mart = by_key.get((task, seed, "bar_p4"))
            two = by_key.get((task, seed, "two_actor_p4"))
            if mart is None or two is None:
                continue
            if not Path(mart["checkpoint_path"]).is_file() or not Path(two["checkpoint_path"]).is_file():
                continue
            if not mart.get("checkpoint", {}).get("has_target_actor"):
                continue
            if not two.get("checkpoint", {}).get("has_target_actor"):
                continue
            pairs.append({"task": task, "T": tau, "seed": seed, "mart": mart, "two": two})
    return pairs


def load_bundle(record: dict[str, Any]) -> dict[str, Any]:
    payload = load_checkpoint(Path(record["checkpoint_path"]))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    first = _actor_params(payload, 1)
    actor = Actor(action_dim=_infer_action_dim(first, mean), max_action=max_action)
    apply = jax.jit(actor.apply)
    critic = TwinCritic()
    q_apply = jax.jit(critic.apply)
    return {
        "payload": payload,
        "mean": mean,
        "std": std,
        "apply": apply,
        "q_apply": q_apply,
        "hash": record["checkpoint_hash"],
    }


def resolve_policy(kind: dict[str, Any], mart: dict[str, Any], two: dict[str, Any]):
    first_src = kind["first_source"]
    cont_src = kind["continuation_source"]
    hop = kind.get("hop")
    if first_src == FIRST_MART_HOP:
        first = _policy(
            mart["apply"],
            _actor_params(mart["payload"], int(hop)),
            mart["mean"],
            mart["std"],
        )
    elif first_src == FIRST_TWO_DEPLOY:
        first = _policy(
            two["apply"],
            _actor_params(two["payload"], 2),
            two["mean"],
            two["std"],
        )
    elif first_src == CONT_MART_POLYAK:
        first = _policy(
            mart["apply"],
            mart["payload"]["target_actor_params"],
            mart["mean"],
            mart["std"],
        )
    elif first_src == CONT_TWO_POLYAK:
        first = _policy(
            two["apply"],
            two["payload"]["target_actor_params"],
            two["mean"],
            two["std"],
        )
    else:
        raise ValueError(first_src)

    if cont_src == CONT_MART_POLYAK:
        cont = _policy(
            mart["apply"],
            mart["payload"]["target_actor_params"],
            mart["mean"],
            mart["std"],
        )
    elif cont_src == CONT_TWO_POLYAK:
        cont = _policy(
            two["apply"],
            two["payload"]["target_actor_params"],
            two["mean"],
            two["std"],
        )
    else:
        raise ValueError(cont_src)
    return first, cont


def evaluate_kind(
    env_name: str,
    kind: dict[str, Any],
    mart: dict[str, Any],
    two: dict[str, Any],
    episodes: int,
) -> list[dict[str, Any]]:
    import gymnasium as gym

    first_fn, cont_fn = resolve_policy(kind, mart, two)
    env = gym.make(EVAL_ENV[_domain(env_name)])
    rows = []
    hop_label = "" if kind.get("hop") is None else str(kind["hop"])
    for ep in range(episodes):
        eval_seed = EVAL_SEED_BASE + ep
        obs, _ = env.reset(seed=eval_seed)
        raw0 = np.asarray(obs, dtype=np.float32)
        a_first = first_fn(raw0)
        a_cont = cont_fn(raw0)
        q_mart_first = _q1(
            mart["q_apply"], mart["payload"]["critic_params"], raw0, mart["mean"], mart["std"], a_first
        )
        q_mart_cont = _q1(
            mart["q_apply"], mart["payload"]["critic_params"], raw0, mart["mean"], mart["std"], a_cont
        )
        q_two_first = _q1(
            two["q_apply"], two["payload"]["critic_params"], raw0, two["mean"], two["std"], a_first
        )
        q_two_cont = _q1(
            two["q_apply"], two["payload"]["critic_params"], raw0, two["mean"], two["std"], a_cont
        )
        rewards = []
        obs, reward, terminated, truncated, _ = env.step(a_first)
        rewards.append(float(reward))
        length = 1
        done = bool(terminated or truncated)
        while not done:
            raw = np.asarray(obs, dtype=np.float32)
            action = cont_fn(raw)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            length += 1
            done = bool(terminated or truncated)
            if length >= HORIZON:
                break
        raw_return = float(sum(rewards))
        rows.append(
            {
                "kind": kind["kind"],
                "hop": hop_label,
                "first_source": kind["first_source"],
                "continuation_source": kind["continuation_source"],
                "evaluation_seed": eval_seed,
                "raw_return": raw_return,
                "discounted_return": discounted_return(rewards, DISCOUNT),
                "normalized_score": float(d4rl_normalized_score(env_name, raw_return)),
                "episode_length": length,
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
                "q_mart_first": q_mart_first,
                "q_mart_cont_action": q_mart_cont,
                "delta_q_mart": q_mart_first - q_mart_cont,
                "q_two_first": q_two_first,
                "q_two_cont_action": q_two_cont,
                "delta_q_two": q_two_first - q_two_cont,
                "first_action_rms": float(np.sqrt(np.mean(np.square(a_first - a_cont)))),
            }
        )
    env.close()
    return rows


def _done_keys(path: Path) -> set[tuple]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (
                row["task"],
                row["T"],
                row["training_seed"],
                row["kind"],
                row["hop"],
                row["first_source"],
                row["continuation_source"],
                row["evaluation_seed"],
            )
            for row in csv.DictReader(handle)
        }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()


def process_pair(pair: dict[str, Any], episodes: int, output_csv: Path) -> None:
    mart = load_bundle(pair["mart"])
    two = load_bundle(pair["two"])
    done = _done_keys(output_csv)
    for kind in eval_kinds_for_cell():
        hop_label = "" if kind.get("hop") is None else str(kind["hop"])
        needed = [
            (
                pair["task"],
                str(pair["T"]),
                str(pair["seed"]),
                kind["kind"],
                hop_label,
                kind["first_source"],
                kind["continuation_source"],
                str(EVAL_SEED_BASE + ep),
            )
            for ep in range(episodes)
        ]
        if all(key in done for key in needed):
            continue
        ep_rows = evaluate_kind(pair["task"], kind, mart, two, episodes)
        out = []
        for row in ep_rows:
            out.append(
                {
                    "task": pair["task"],
                    "T": str(pair["T"]),
                    "training_seed": str(pair["seed"]),
                    "checkpoint_hash_mart": mart["hash"],
                    "checkpoint_hash_two": two["hash"],
                    **{k: row[k] if not isinstance(row[k], float) else f"{row[k]}" for k in row},
                }
            )
        _append_rows(output_csv, out)
        mean_norm = np.mean([float(r["normalized_score"]) for r in ep_rows])
        print(
            f"{pair['task']} seed={pair['seed']} hop={hop_label or '-'} "
            f"{kind['first_source']} -> {kind['continuation_source']} "
            f"mean_norm={mean_norm:.2f}",
            flush=True,
        )


def _worker(worker_id: int, pairs: list[dict[str, Any]], output_csv: str, episodes: int, cpu0: int) -> None:
    try:
        os.sched_setaffinity(0, {cpu0 + worker_id})
    except OSError:
        pass
    out = Path(output_csv)
    for pair in pairs:
        process_pair(pair, episodes, out)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: list[dict[str, str]], task: str, seed: int | None, selector: dict[str, Any]) -> list[dict[str, str]]:
    picked = []
    for row in rows:
        if row["task"] != task:
            continue
        if seed is not None and int(row["training_seed"]) != int(seed):
            continue
        if row_matches(row, selector):
            picked.append(row)
    return picked


def full_policy_means() -> dict[tuple, dict[str, float]]:
    if not FULL_EPISODES.is_file():
        return {}
    grouped: dict[tuple, list[dict[str, str]]] = {}
    with FULL_EPISODES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["T"]) != 10:
                continue
            key = (row["task"], int(row["training_seed"]), row["method"], row["actor_role"])
            grouped.setdefault(key, []).append(row)
    out = {}
    for key, group in grouped.items():
        out[key] = {
            "mean_normalized": float(np.mean([float(r["normalized_score"]) for r in group])),
            "mean_length": float(np.mean([float(r["episode_length"]) for r in group])),
            "n_timeout": float(np.sum([float(r["episode_length"]) >= HORIZON for r in group])),
            "n": float(len(group)),
        }
    return out


def fmt(value: float, digits: int = 2) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def write_report(rows: list[dict[str, str]], pairs: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    full = full_policy_means()
    lines = [
        "# First-action × Polyak continuation (Hopper T=10)",
        "",
        f"Built: {now_kst()}",
        "",
        "Understood as: same Gymnasium reset seeds as the hop-actor audit",
        f"({EVAL_SEED_BASE}–{EVAL_SEED_BASE + extra['episodes'] - 1}), first action from",
        "MART μ2/μ3 or two-actor deployment, then continuation from MART Polyak μ1",
        "or two-actor Polyak target. CPU only. First90 checkpoints on ext_csv.",
        "Walker / HalfCheetah-expert tail90 checkpoints are not on this host.",
        "",
        "## Checkpoints",
        "",
        f"- Paired cells run: {len(pairs)} (3 Hopper tasks × seeds {{0,1}}).",
        "- MART and two-actor 1M files exist locally; `target_actor_params` present.",
        "",
        "## Protocol",
        "",
        "- Continuation is the Polyak target actor, not the online first actor.",
        "- Each policy uses its own checkpoint mean/std.",
        f"- Discount for ΔG is {DISCOUNT}, matching `train_td3bc --discount`.",
        "- Two-actor first-action rows are shared across hop-2 and hop-3 tables.",
        "- Full-episode μk / deployment numbers come from `p0_hop_actor_audit`.",
        "",
    ]
    for task in HOPPER_T10_TASKS:
        lines += [f"## {task.replace('-v2', '')}", ""]
        for hop in HOPS:
            lines += [
                f"### Hop {hop}",
                "",
                "| First action | Continue MART Polyak μ1 | Continue two-actor Polyak target |",
                "| --- | ---: | ---: |",
            ]
            for first, first_label in (
                (FIRST_MART_HOP, f"MART μ{hop}"),
                (FIRST_TWO_DEPLOY, "two-actor deploy"),
            ):
                vals = []
                for cont in (CONT_MART_POLYAK, CONT_TWO_POLYAK):
                    sub = select_rows(rows, task, None, table_row_selector(hop, first, cont))
                    vals.append(mean_score(sub))
                lines.append(
                    f"| {first_label} | {fmt(vals[0])} | {fmt(vals[1])} |"
                )
            lines.append("")
            # first-action effect holding MART continuation
            treat = select_rows(rows, task, None, table_row_selector(hop, FIRST_MART_HOP, CONT_MART_POLYAK))
            base = select_rows(rows, task, None, baseline_selector(CONT_MART_POLYAK))
            if treat and base:
                by_seed_eval_t = {
                    (int(r["training_seed"]), int(r["evaluation_seed"])): float(r["normalized_score"])
                    for r in treat
                }
                by_seed_eval_b = {
                    (int(r["training_seed"]), int(r["evaluation_seed"])): float(r["normalized_score"])
                    for r in base
                }
                keys = sorted(set(by_seed_eval_t) & set(by_seed_eval_b))
                if keys:
                    delta = paired_delta(
                        [by_seed_eval_t[k] for k in keys],
                        [by_seed_eval_b[k] for k in keys],
                    )
                    dlen = paired_delta(
                        [
                            float(r["episode_length"])
                            for r in treat
                            if (int(r["training_seed"]), int(r["evaluation_seed"])) in set(keys)
                        ],
                        [
                            float(r["episode_length"])
                            for r in base
                            if (int(r["training_seed"]), int(r["evaluation_seed"])) in set(keys)
                        ],
                    )
                    n_timeout_h = sum(int(float(r["episode_length"]) >= HORIZON) for r in treat)
                    lines += [
                        f"- First-action swap μ{hop} vs Polyak μ1, same MART continuation: "
                        f"mean Δ score {delta['mean']:+.2f} (std {delta['std']:.2f}, "
                        f"{int(delta['n_negative'])}/{int(delta['n'])} negative).",
                        f"- Hybrid mean length change {dlen['mean']:+.1f}; hybrid timeouts "
                        f"{n_timeout_h}/{len(treat)}.",
                        "",
                    ]
            mu_full = full.get((task, 0, "MART4", f"mu{hop}")) or {}
            mu_full1 = full.get((task, 1, "MART4", f"mu{hop}")) or {}
            if mu_full or mu_full1:
                scores = [mu_full.get("mean_normalized", float("nan")), mu_full1.get("mean_normalized", float("nan"))]
                lengths = [mu_full.get("mean_length", float("nan")), mu_full1.get("mean_length", float("nan"))]
                lines.append(
                    f"- Full-episode MART μ{hop} (existing hop audit): "
                    f"scores {fmt(scores[0])}/{fmt(scores[1])}, "
                    f"lengths {fmt(lengths[0], 0)}/{fmt(lengths[1], 0)}."
                )
                lines.append("")
            if hop == 3:
                treat_g = select_rows(rows, task, None, table_row_selector(3, FIRST_MART_HOP, CONT_MART_POLYAK))
                base_g = select_rows(rows, task, None, baseline_selector(CONT_MART_POLYAK))
                if treat_g and base_g:
                    g_t = {
                        (int(r["training_seed"]), int(r["evaluation_seed"])): r
                        for r in treat_g
                    }
                    g_b = {
                        (int(r["training_seed"]), int(r["evaluation_seed"])): r
                        for r in base_g
                    }
                    keys = sorted(set(g_t) & set(g_b))
                    if keys:
                        dq = [float(g_t[k]["delta_q_mart"]) for k in keys]
                        dg = [
                            float(g_t[k]["discounted_return"]) - float(g_b[k]["discounted_return"])
                            for k in keys
                        ]
                        rank = ranking_agreement(dq, dg)
                        lines += [
                            "Hop-3 ranking on the same discount 0.99:",
                            f"- mean ΔQ(s0, μ3−Polyak μ1) {rank['mean_delta_q']:+.3f}.",
                            f"- mean discounted ΔG {rank['mean_delta_g']:+.3f} "
                            f"(std {rank['std_delta_g']:.3f}).",
                            f"- sign agree {int(rank['n_sign_agree'])}/{int(rank['n'])}; "
                            f"Q↑ G↓ {int(rank['n_q_pos_g_neg'])}; "
                            f"Q↓ G↑ {int(rank['n_q_neg_g_pos'])}.",
                            "",
                        ]
        # continuation contrast holding first=μ2
        lines += [
            "First-action contrast (MART hop minus two-actor deploy, same continuation) "
            "and continuation contrast (MART Polyak μ1 minus two-actor Polyak, same first "
            "action) are in `contrasts.csv`.",
            "",
        ]
    lines += [
        "## Read",
        "",
        "A small first-action gap under MART Polyak continuation, together with a",
        "much larger full-episode μk gap, supports a continuation difference on the",
        "evaluated initial states. The 2×2 separates that from a two-actor first-action",
        "difference and from a two-actor continuation difference. It does not decompose",
        "the full-episode drop into a percentage, because later states diverge.",
        "",
        "Walker-medium is not in this dump: those 1M checkpoints are not on ext_csv.",
        "",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contrasts(rows: list[dict[str, str]]) -> None:
    out_rows = []
    for task in HOPPER_T10_TASKS:
        for hop in HOPS:
            for cont in (CONT_MART_POLYAK, CONT_TWO_POLYAK):
                a = select_rows(rows, task, None, table_row_selector(hop, FIRST_MART_HOP, cont))
                b = select_rows(rows, task, None, table_row_selector(hop, FIRST_TWO_DEPLOY, cont))
                if a and b:
                    ka = {(int(r["training_seed"]), int(r["evaluation_seed"])): float(r["normalized_score"]) for r in a}
                    kb = {(int(r["training_seed"]), int(r["evaluation_seed"])): float(r["normalized_score"]) for r in b}
                    keys = sorted(set(ka) & set(kb))
                    if keys:
                        delta = paired_delta([ka[k] for k in keys], [kb[k] for k in keys])
                        out_rows.append(
                            {
                                "task": task,
                                "hop": hop,
                                "contrast": "first_mart_hop_minus_two_deploy",
                                "held_fixed": cont,
                                **{k: delta[k] for k in delta},
                            }
                        )
            for first in (FIRST_MART_HOP, FIRST_TWO_DEPLOY):
                a = select_rows(rows, task, None, table_row_selector(hop, first, CONT_MART_POLYAK))
                b = select_rows(rows, task, None, table_row_selector(hop, first, CONT_TWO_POLYAK))
                if a and b:
                    ka = {(int(r["training_seed"]), int(r["evaluation_seed"])): float(r["normalized_score"]) for r in a}
                    kb = {(int(r["training_seed"]), int(r["evaluation_seed"])): float(r["normalized_score"]) for r in b}
                    keys = sorted(set(ka) & set(kb))
                    if keys:
                        delta = paired_delta([ka[k] for k in keys], [kb[k] for k in keys])
                        out_rows.append(
                            {
                                "task": task,
                                "hop": hop,
                                "contrast": "cont_mart_polyak_minus_two_polyak",
                                "held_fixed": first if first != FIRST_MART_HOP else f"mart_mu{hop}",
                                **{k: delta[k] for k in delta},
                            }
                        )
    path = OUT / "contrasts.csv"
    if not out_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def plot_tables(rows: list[dict[str, str]]) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    labels = [
        "μk → MART μ1",
        "μk → two tgt",
        "deploy → MART μ1",
        "deploy → two tgt",
    ]
    colors = ["#2c7fb8", "#41b6c4", "#fdae61", "#d73027"]
    for ax, task in zip(axes, HOPPER_T10_TASKS):
        x = np.arange(len(labels))
        width = 0.35
        for offset, hop, hatch in ((-width / 2, 2, None), (width / 2, 3, "//")):
            vals = []
            for first, cont in hybrid_cells():
                sub = select_rows(rows, task, None, table_row_selector(hop, first, cont))
                vals.append(mean_score(sub))
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                label=f"hop {hop}",
            )
        ax.set_title(task.replace("-v2", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("mean D4RL score")
    axes[0].legend(fontsize=7)
    fig.suptitle("Hopper T=10: first action × Polyak continuation")
    fig.tight_layout()
    path = OUT / "fig_first_action_continuation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=list(HOPPER_T10_TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu0", type=int, default=200)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"refusing non-CPU JAX backend {jax.default_backend()!r}")
    pairs = paired_inventory(tuple(args.tasks), args.T, list(args.seeds))
    OUT.mkdir(parents=True, exist_ok=True)
    output_csv = OUT / "episodes.csv"
    print(
        json.dumps(
            {
                "understood_as": (
                    "Fill the 2x2 first-action x Polyak-continuation table for "
                    "MART mu2/mu3 vs two-actor deploy on Hopper T=10 first90 checkpoints."
                ),
                "backend": jax.default_backend(),
                "n_pairs": len(pairs),
                "jobs_per_pair": len(eval_kinds_for_cell()),
                "episodes": args.episodes,
                "missing": [
                    t
                    for t in args.tasks
                    if not any(p["task"] == t for p in pairs)
                ],
                "started_at": now_kst(),
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.report_only:
        leftover = pairs
        workers = max(1, min(int(args.workers), len(leftover) or 1))
        if leftover:
            if workers == 1:
                for pair in leftover:
                    process_pair(pair, args.episodes, output_csv)
            else:
                shards: list[list] = [[] for _ in range(workers)]
                for index, pair in enumerate(leftover):
                    shards[index % workers].append(pair)
                ctx = mp.get_context("spawn")
                procs = []
                for worker_id, shard in enumerate(shards):
                    if not shard:
                        continue
                    proc = ctx.Process(
                        target=_worker,
                        args=(worker_id, shard, str(output_csv), args.episodes, args.cpu0),
                    )
                    proc.start()
                    procs.append(proc)
                failures = 0
                for proc in procs:
                    proc.join()
                    if proc.exitcode != 0:
                        failures += 1
                if failures:
                    raise RuntimeError(f"{failures} CPU workers failed")
    rows = load_rows(output_csv)
    write_contrasts(rows)
    figure = plot_tables(rows) if rows else None
    write_report(
        rows,
        pairs,
        {"episodes": args.episodes, "figure": figure},
    )
    (OUT / "RECEIPT.json").write_text(
        json.dumps(
            {
                "built_at": now_kst(),
                "n_episode_rows": len(rows),
                "n_pairs": len(pairs),
                "backend": jax.default_backend(),
                "discount": DISCOUNT,
                "eval_seed_base": EVAL_SEED_BASE,
                "continuation": "Polyak target_actor_params, not online actor 1",
                "figure": figure,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_csv} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
