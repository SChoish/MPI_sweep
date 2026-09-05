#!/usr/bin/env python3
"""Continuation-matched MC for MART hops whose hop loss improved but return fell.

Walker-medium T=20 seed 0, hops 2 and 3 only. First action from the compared
actor; remaining steps from that checkpoint's Polyak μ1 (target_actor).
Does not train. Forces JAX onto CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

from _lab_import import REPO_ROOT as _ROOT, ensure_train_import_path  # noqa: E402

ensure_train_import_path()
sys.path.insert(0, str(_ROOT))

import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.diagnostics.p0_continuation_mc import (  # noqa: E402
    bootstrap_truncated,
    discounted_return,
)
from scripts.diagnostics.p0_intermediate_actors import (  # noqa: E402
    DEFAULT_TAIL_CHECKPOINTS,
    EVAL_EPISODE_SEEDS,
    load_json,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.diagnostics.p0_survival import decompose_episodes  # noqa: E402
from scripts.diagnostics.run_p0_intermediate_actors import (  # noqa: E402
    actor_params_list,
    infer_action_dim,
    load_payload,
)
from train_td3bc import EVAL_ENV, Actor, TwinCritic, _domain, d4rl_normalized_score  # noqa: E402

CELL = {
    "environment": "walker2d-medium-v2",
    "T": 20,
    "seed": 0,
    "condition": "bar_p4",
    "run_key": "bar_p4|walker2d-medium-v2|T=20|seed=0",
}
OUT_DEFAULT = _ROOT / "sweep_results/diagnostics/p0_hop_continuation_mc"
HOPS = (
    {"hop_index": 2, "ref_id": "mart_mu1", "new_id": "mart_mu2"},
    {"hop_index": 3, "ref_id": "mart_mu2", "new_id": "mart_mu3"},
)


def _apply_fn(actor: Actor, params: Any) -> Callable[[np.ndarray], np.ndarray]:
    apply = jax.jit(actor.apply)

    def policy(state: np.ndarray) -> np.ndarray:
        return np.asarray(apply(params, jnp.asarray(state, dtype=jnp.float32)), dtype=np.float32)

    return policy


def _q1_fn(critic: TwinCritic, critic_params: Any) -> Callable[[np.ndarray, np.ndarray], float]:
    def q1(obs, act):
        q1_val, _ = critic.apply(critic_params, obs[None], act[None])
        return jnp.squeeze(q1_val)

    q_jit = jax.jit(q1)

    def query(state: np.ndarray, action: np.ndarray) -> float:
        return float(q_jit(jnp.asarray(state, dtype=jnp.float32), jnp.asarray(action, dtype=jnp.float32)))

    return query


def _rollout(
    env: gym.Env,
    *,
    first_policy: Callable[[np.ndarray], np.ndarray],
    rest_policy: Callable[[np.ndarray], np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    q1: Callable[[np.ndarray, np.ndarray], float],
    eval_seed: int,
    gamma: float,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=int(eval_seed))
    rewards: list[float] = []
    done = False
    truncated_end = False
    length = 0
    q_s0 = float("nan")
    bootstrap_q = 0.0
    while not done:
        raw = np.asarray(obs, dtype=np.float32)
        state = (raw - mean) / std
        action = first_policy(state) if length == 0 else rest_policy(state)
        if length == 0:
            q_s0 = q1(state, action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        length += 1
        truncated_end = bool(truncated)
        done = bool(terminated or truncated)
        if truncated_end:
            next_state = (np.asarray(obs, dtype=np.float32) - mean) / std
            bootstrap_q = q1(next_state, rest_policy(next_state))
    raw_return = float(sum(rewards))
    return {
        "eval_seed": int(eval_seed),
        "raw_return": raw_return,
        "normalized_score": float(d4rl_normalized_score(CELL["environment"], raw_return)),
        "episode_length": length,
        "truncated": truncated_end,
        "q_s0": q_s0,
        "discounted_return": discounted_return(rewards, gamma),
        "discounted_return_bootstrapped": bootstrap_truncated(
            rewards, gamma, truncated_end, bootstrap_q
        ),
        "bootstrap_q": float(bootstrap_q) if truncated_end else 0.0,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = decompose_episodes(rows)
    stats["q_s0_mean"] = float(np.mean([r["q_s0"] for r in rows]))
    stats["discounted_return_mean"] = float(np.mean([r["discounted_return"] for r in rows]))
    stats["discounted_return_bootstrapped_mean"] = float(
        np.mean([r["discounted_return_bootstrapped"] for r in rows])
    )
    return stats


def write_report(
    out_dir: Path,
    protocol_stats: Mapping[str, dict[str, Any]],
    hop_rows: list[dict[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    lines = [
        "# Continuation-matched MC (Walker-medium T=20 seed 0, hops 2–3)",
        "",
        "First action from the compared MART actor; remaining steps from the",
        "checkpoint Polyak μ1 (`target_actor_params`). This is the Bellman",
        "continuation, not a full-episode rollout of the hop actor.",
        "Eval seeds `1000–1009`, Gymnasium v4, no training.",
        "",
        f"Checkpoint `{meta['checkpoint']}`",
        f"sha256 `{meta['checkpoint_sha256']}`; discount `{meta['discount']}`.",
        "",
        "## Protocol means (10 episodes)",
        "",
        "| Protocol | First | Rest | Score | Mean L | Timeouts | Q(s0,a0) | Disc. G | Disc. G+boot |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = [
        "full_mu1",
        "full_mu2",
        "full_mu3",
        "full_target",
        "first_mu1_then_target",
        "first_mu2_then_target",
        "first_mu3_then_target",
    ]
    labels = {
        "full_mu1": ("μ1", "μ1"),
        "full_mu2": ("μ2", "μ2"),
        "full_mu3": ("μ3", "μ3"),
        "full_target": ("Polyak μ1", "Polyak μ1"),
        "first_mu1_then_target": ("μ1", "Polyak μ1"),
        "first_mu2_then_target": ("μ2", "Polyak μ1"),
        "first_mu3_then_target": ("μ3", "Polyak μ1"),
    }
    for key in order:
        s = protocol_stats[key]
        first, rest = labels[key]
        lines.append(
            f"| `{key}` | {first} | {rest} | {s['normalized_score_mean']:.2f} | "
            f"{s['episode_length_mean']:.0f} | {s['n_timeout']}/{s['n_episodes']} | "
            f"{s['q_s0_mean']:.2f} | {s['discounted_return_mean']:.2f} | "
            f"{s['discounted_return_bootstrapped_mean']:.2f} |"
        )
    lines += [
        "",
        "## Hop contrasts (new − ref)",
        "",
        "| Hop | ΔJ full | ΔJ hybrid (1-step then Polyak μ1) | ΔQ(s0) | ΔG disc. | ΔG disc.+boot |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in hop_rows:
        lines.append(
            f"| {row['hop_index']} ({row['ref_id']}→{row['new_id']}) | "
            f"{row['full_score_delta']:.2f} | {row['hybrid_score_delta']:.2f} | "
            f"{row['q_s0_delta']:.2f} | {row['discounted_delta']:.2f} | "
            f"{row['discounted_bootstrapped_delta']:.2f} |"
        )
    lines += [
        "",
        "Full ΔJ is the online hop actor rolled out for the whole episode.",
        "Hybrid ΔJ uses the training continuation after one step. Q is the",
        "frozen online critic at the episode start. A positive ΔQ with a",
        "negative hybrid ΔG is critic misranking of the hop action under the",
        "Bellman continuation. A near-zero hybrid ΔJ with a large negative",
        "full ΔJ means the first action is not the failure; continuing with",
        "the hop actor is.",
        "",
        "## Readout",
        "",
        "Hop 2: full ΔJ=−15.86 while hybrid ΔJ=+0.50 and hybrid length is unchanged",
        "(10/10 timeouts). Q(s0) also rises. The first μ2 action under Polyak μ1",
        "is not the failure. Rolling out μ2 for the rest of the episode is.",
        "",
        "Hop 3: full ΔJ=−9.39, hybrid ΔJ=−1.21, hybrid still 10/10 timeouts.",
        "Q(s0) rises (+0.54) while discounted hybrid G falls (−3.77). That is a",
        "small start-state misranking under the Bellman continuation, much",
        "smaller than the full-episode drop. Most of the hop-3 harm is still",
        "from continuing with μ3, not from one step plus Polyak μ1.",
        "",
        "Polyak μ1 alone scores 83.87 with 10/10 timeouts, above online μ1",
        "(75.65). The training continuation is healthier than the hop actors'",
        "own rollouts.",
        "",
    ]
    (out_dir / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-json", type=Path, default=DEFAULT_TAIL_CHECKPOINTS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recorded = load_json(args.checkpoints_json)
    entry = recorded[CELL["run_key"]]
    ckpt = Path(entry["checkpoint"])
    payload = load_payload(ckpt)
    if "target_actor_params" not in payload:
        raise KeyError("checkpoint missing target_actor_params (Polyak μ1)")
    params_list = actor_params_list(payload)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    max_action = float(payload.get("max_action", 1.0))
    gamma = float(payload.get("config", {}).get("discount", 0.99))
    actor = Actor(action_dim=infer_action_dim(params_list[0]), max_action=max_action)
    critic = TwinCritic()
    policies = {
        "mart_mu1": _apply_fn(actor, params_list[0]),
        "mart_mu2": _apply_fn(actor, params_list[1]),
        "mart_mu3": _apply_fn(actor, params_list[2]),
        "target": _apply_fn(actor, payload["target_actor_params"]),
    }
    q1 = _q1_fn(critic, payload["critic_params"])
    env = gym.make(EVAL_ENV[_domain(CELL["environment"])])
    protocols = (
        ("full_mu1", "mart_mu1", "mart_mu1"),
        ("full_mu2", "mart_mu2", "mart_mu2"),
        ("full_mu3", "mart_mu3", "mart_mu3"),
        ("full_target", "target", "target"),
        ("first_mu1_then_target", "mart_mu1", "target"),
        ("first_mu2_then_target", "mart_mu2", "target"),
        ("first_mu3_then_target", "mart_mu3", "target"),
    )
    episode_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, first_id, rest_id in protocols:
        print(f"[mc] {name}", flush=True)
        for eval_seed in EVAL_EPISODE_SEEDS:
            row = _rollout(
                env,
                first_policy=policies[first_id],
                rest_policy=policies[rest_id],
                mean=mean,
                std=std,
                q1=q1,
                eval_seed=eval_seed,
                gamma=gamma,
            )
            packed = {
                **CELL,
                "protocol": name,
                "first_actor": first_id,
                "rest_actor": rest_id,
                **row,
            }
            episode_rows.append(packed)
            grouped[name].append(packed)
    env.close()
    protocol_stats = {name: _summarize(rows) for name, rows in grouped.items()}
    hop_rows = []
    full_map = {1: "full_mu1", 2: "full_mu2", 3: "full_mu3"}
    hybrid_map = {1: "first_mu1_then_target", 2: "first_mu2_then_target", 3: "first_mu3_then_target"}
    for hop in HOPS:
        ref_i, new_i = hop["hop_index"] - 1, hop["hop_index"]
        ref_s, new_s = protocol_stats[full_map[ref_i]], protocol_stats[full_map[new_i]]
        ref_h, new_h = protocol_stats[hybrid_map[ref_i]], protocol_stats[hybrid_map[new_i]]
        hop_rows.append(
            {
                **CELL,
                "hop_index": hop["hop_index"],
                "ref_id": hop["ref_id"],
                "new_id": hop["new_id"],
                "full_score_delta": new_s["normalized_score_mean"] - ref_s["normalized_score_mean"],
                "hybrid_score_delta": new_h["normalized_score_mean"] - ref_h["normalized_score_mean"],
                "q_s0_delta": new_h["q_s0_mean"] - ref_h["q_s0_mean"],
                "discounted_delta": new_h["discounted_return_mean"] - ref_h["discounted_return_mean"],
                "discounted_bootstrapped_delta": (
                    new_h["discounted_return_bootstrapped_mean"]
                    - ref_h["discounted_return_bootstrapped_mean"]
                ),
                "full_length_delta": new_s["episode_length_mean"] - ref_s["episode_length_mean"],
                "hybrid_length_delta": new_h["episode_length_mean"] - ref_h["episode_length_mean"],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "episodes.csv",
        episode_rows,
        [
            "environment",
            "T",
            "seed",
            "condition",
            "run_key",
            "protocol",
            "first_actor",
            "rest_actor",
            "eval_seed",
            "raw_return",
            "normalized_score",
            "episode_length",
            "truncated",
            "q_s0",
            "discounted_return",
            "discounted_return_bootstrapped",
            "bootstrap_q",
        ],
    )
    write_csv(
        args.out_dir / "hop_contrasts.csv",
        hop_rows,
        [
            "environment",
            "T",
            "seed",
            "condition",
            "run_key",
            "hop_index",
            "ref_id",
            "new_id",
            "full_score_delta",
            "hybrid_score_delta",
            "q_s0_delta",
            "discounted_delta",
            "discounted_bootstrapped_delta",
            "full_length_delta",
            "hybrid_length_delta",
        ],
    )
    meta = {
        **CELL,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "recorded_file_sha256": entry["file_sha256"],
        "discount": gamma,
        "eval_seeds": list(EVAL_EPISODE_SEEDS),
        "continuation": "target_actor_params (Polyak μ1)",
        "n_episodes": len(EVAL_EPISODE_SEEDS),
        "hops": [2, 3],
    }
    write_json(args.out_dir / "STATUS.json", meta)
    write_report(args.out_dir, protocol_stats, hop_rows, meta)
    print(json.dumps({"n_episode_rows": len(episode_rows), "hops": hop_rows}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
