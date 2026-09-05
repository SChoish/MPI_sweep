#!/usr/bin/env python3
"""Cross-method continuation MC: MART hops vs two-actor deployment.

Same Walker-medium T=20 seed 0 start states (eval seeds 1000–1009). For MART
μ2 and μ3, fill the 2×2 of first action × continuation:

  first: MART hop actor vs two-actor deployment
  rest:  MART Polyak μ1 vs two-actor Polyak target

Each actor uses its own checkpoint mean/std. Q(s0,a0) uses the
continuation checkpoint's critic. Does not train. Forces JAX onto CPU.
Does not mix first90/tail90 shards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    paired_deltas,
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
    "pair_key": "walker2d-medium-v2|T=20|seed=0",
}
MART_KEY = "bar_p4|walker2d-medium-v2|T=20|seed=0"
TA_KEY = "two_actor_p4|walker2d-medium-v2|T=20|seed=0"
OUT_DEFAULT = _ROOT / "sweep_results/diagnostics/p0_cross_method_continuation_mc"
ENV_NAME = CELL["environment"]


@dataclass
class Bundle:
    name: str
    act: Callable[[np.ndarray], np.ndarray]
    q1: Callable[[np.ndarray, np.ndarray], float]
    mean: np.ndarray
    std: np.ndarray
    checkpoint: str
    sha256: str

    def action(self, raw: np.ndarray) -> np.ndarray:
        state = (raw - self.mean) / self.std
        return self.act(state)

    def q(self, raw: np.ndarray, action: np.ndarray) -> float:
        state = (raw - self.mean) / self.std
        return self.q1(state, action)


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
        return float(
            q_jit(
                jnp.asarray(state, dtype=jnp.float32),
                jnp.asarray(action, dtype=jnp.float32),
            )
        )

    return query


def _load_bundle(
    *,
    name: str,
    checkpoint: Path,
    expected_sha: str,
    actor_params: Any,
    critic_params: Any,
    mean: np.ndarray,
    std: np.ndarray,
    max_action: float,
) -> Bundle:
    digest = sha256_file(checkpoint)
    if digest != expected_sha:
        raise ValueError(f"hash mismatch for {checkpoint}: got {digest}, expected {expected_sha}")
    actor = Actor(action_dim=infer_action_dim(actor_params), max_action=max_action)
    critic = TwinCritic()
    return Bundle(
        name=name,
        act=_apply_fn(actor, actor_params),
        q1=_q1_fn(critic, critic_params),
        mean=mean,
        std=std,
        checkpoint=str(checkpoint),
        sha256=digest,
    )


def _rollout(
    env: gym.Env,
    *,
    first: Bundle,
    rest: Bundle,
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
        action = first.action(raw) if length == 0 else rest.action(raw)
        if length == 0:
            q_s0 = rest.q(raw, action)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        length += 1
        truncated_end = bool(truncated)
        done = bool(terminated or truncated)
        if truncated_end:
            next_raw = np.asarray(obs, dtype=np.float32)
            bootstrap_q = rest.q(next_raw, rest.action(next_raw))
    raw_return = float(sum(rewards))
    return {
        "eval_seed": int(eval_seed),
        "raw_return": raw_return,
        "normalized_score": float(d4rl_normalized_score(ENV_NAME, raw_return)),
        "episode_length": length,
        "truncated": truncated_end,
        "q_s0": q_s0,
        "discounted_return": discounted_return(rewards, gamma),
        "discounted_return_bootstrapped": bootstrap_truncated(
            rewards, gamma, truncated_end, bootstrap_q
        ),
        "bootstrap_q": float(bootstrap_q) if truncated_end else 0.0,
        "q_source": rest.name,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = decompose_episodes(rows)
    stats["q_s0_mean"] = float(np.mean([r["q_s0"] for r in rows]))
    stats["discounted_return_mean"] = float(np.mean([r["discounted_return"] for r in rows]))
    stats["discounted_return_bootstrapped_mean"] = float(
        np.mean([r["discounted_return_bootstrapped"] for r in rows])
    )
    return stats


def _by_seed(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in sorted(rows, key=lambda r: int(r["eval_seed"]))]


def _contrast_row(
    *,
    hop: int,
    label: str,
    new_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    same_critic: bool,
) -> dict[str, Any]:
    score = paired_deltas(
        _by_seed(new_rows, "normalized_score"),
        _by_seed(ref_rows, "normalized_score"),
    )
    disc = paired_deltas(
        _by_seed(new_rows, "discounted_return"),
        _by_seed(ref_rows, "discounted_return"),
    )
    length = paired_deltas(
        _by_seed(new_rows, "episode_length"),
        _by_seed(ref_rows, "episode_length"),
    )
    q_stats = (
        paired_deltas(_by_seed(new_rows, "q_s0"), _by_seed(ref_rows, "q_s0"))
        if same_critic
        else None
    )
    row: dict[str, Any] = {
        **CELL,
        "hop_index": hop,
        "contrast": label,
        "score_mean": score["mean"],
        "score_se": score["se"],
        "score_mean_over_se": score["mean_over_se"],
        "score_n_pos": score["n_pos"],
        "score_n_neg": score["n_neg"],
        "discounted_mean": disc["mean"],
        "discounted_se": disc["se"],
        "discounted_mean_over_se": disc["mean_over_se"],
        "discounted_n_pos": disc["n_pos"],
        "discounted_n_neg": disc["n_neg"],
        "length_mean": length["mean"],
        "same_critic": same_critic,
    }
    if q_stats is None:
        row.update(
            {
                "q_s0_mean": float("nan"),
                "q_s0_se": float("nan"),
                "q_s0_mean_over_se": float("nan"),
                "q_s0_n_pos": "",
                "q_s0_n_neg": "",
            }
        )
    else:
        row.update(
            {
                "q_s0_mean": q_stats["mean"],
                "q_s0_se": q_stats["se"],
                "q_s0_mean_over_se": q_stats["mean_over_se"],
                "q_s0_n_pos": q_stats["n_pos"],
                "q_s0_n_neg": q_stats["n_neg"],
            }
        )
    return row


def write_report(
    out_dir: Path,
    protocol_stats: Mapping[str, dict[str, Any]],
    contrasts: list[dict[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    def fmt_proto(key: str) -> str:
        s = protocol_stats[key]
        return (
            f"{s['normalized_score_mean']:.2f} | {s['episode_length_mean']:.0f} | "
            f"{s['n_timeout']}/{s['n_episodes']} | {s['q_s0_mean']:.2f} | "
            f"{s['discounted_return_mean']:.2f}"
        )

    lines = [
        "# Cross-method continuation MC (Walker-medium T=20 seed 0)",
        "",
        "Tail90 only. Same eval seeds `1000–1009`. First action and continuation",
        "may come from different checkpoints; each actor uses its own mean/std.",
        "Q(s0,a0) is the continuation critic. Start-state only: later visited",
        "states are not restored here. Hybrid vs full rollout is not a percent",
        "decomposition of return, because later actions and visited states both",
        "change.",
        "",
        f"MART `{meta['mart_checkpoint']}`",
        f"sha256 `{meta['mart_sha256']}`",
        f"two-actor `{meta['two_actor_checkpoint']}`",
        f"sha256 `{meta['two_actor_sha256']}`; discount `{meta['discount']}`.",
        "",
        "## Full rollouts",
        "",
        "| Protocol | Score | Mean L | Timeouts | Q(s0,a0) | Disc. G |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| MART μ2 | {fmt_proto('full_mart_mu2')} |",
        f"| MART μ3 | {fmt_proto('full_mart_mu3')} |",
        f"| two-actor deployment | {fmt_proto('full_ta_deploy')} |",
        f"| MART Polyak μ1 | {fmt_proto('full_mart_polyak')} |",
        f"| two-actor Polyak target | {fmt_proto('full_ta_polyak')} |",
        "",
        "## 2×2 hybrid (first action × continuation)",
        "",
    ]
    for hop, mart_first, title in (
        (2, "first_mart_mu2", "Hop 2: MART μ2 vs two-actor deployment"),
        (3, "first_mart_mu3", "Hop 3: MART μ3 vs two-actor deployment"),
    ):
        lines += [
            f"### {title}",
            "",
            "| First \\\\ rest | MART Polyak μ1 | two-actor Polyak target |",
            "| --- | ---: | ---: |",
        ]
        cells = {
            ("mart", "mart"): protocol_stats[f"{mart_first}_rest_mart_polyak"],
            ("mart", "ta"): protocol_stats[f"{mart_first}_rest_ta_polyak"],
            ("ta", "mart"): protocol_stats["first_ta_deploy_rest_mart_polyak"],
            ("ta", "ta"): protocol_stats["first_ta_deploy_rest_ta_polyak"],
        }

        def cell(first: str, rest: str) -> str:
            s = cells[(first, rest)]
            return (
                f"J={s['normalized_score_mean']:.2f}, L={s['episode_length_mean']:.0f} "
                f"({s['n_timeout']}/10), G={s['discounted_return_mean']:.2f}"
            )

        lines += [
            f"| MART hop | {cell('mart', 'mart')} | {cell('mart', 'ta')} |",
            f"| two-actor deploy | {cell('ta', 'mart')} | {cell('ta', 'ta')} |",
            "",
        ]

    lines += [
        "## Paired contrasts (new − ref, 10 episodes)",
        "",
        "Q deltas are reported only when both arms use the same continuation",
        "critic. Continuation swaps change the critic, so those rows leave ΔQ blank.",
        "",
        "| Hop | Contrast | ΔJ mean (se, mean/se) | ΔG disc. mean (se, mean/se) | ΔQ(s0) |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in contrasts:
        q = (
            "n/a"
            if not row["same_critic"]
            else (
                f"{row['q_s0_mean']:.2f} (se {row['q_s0_se']:.2f}, "
                f"{row['q_s0_mean_over_se']:.2f}; {row['q_s0_n_pos']}+/{row['q_s0_n_neg']}-)"
            )
        )
        lines.append(
            f"| {row['hop_index']} | {row['contrast']} | "
            f"{row['score_mean']:.2f} (se {row['score_se']:.2f}, "
            f"{row['score_mean_over_se']:.2f}; {row['score_n_pos']}+/{row['score_n_neg']}-) | "
            f"{row['discounted_mean']:.2f} (se {row['discounted_se']:.2f}, "
            f"{row['discounted_mean_over_se']:.2f}; {row['discounted_n_pos']}+/{row['discounted_n_neg']}-) | "
            f"{q} |"
        )
    hop3_qg = next(
        r
        for r in contrasts
        if r["hop_index"] == 3 and r["contrast"] == "MART μ3−μ2 | MART Polyak μ1"
    )
    lines += [
        "",
        "## Hop 3 start-state ΔQ vs discounted ΔG",
        "",
        "Same discount 0.99. MART μ3 vs μ2, both followed by MART Polyak μ1.",
        f"ΔQ(s0) mean {hop3_qg['q_s0_mean']:.3f} (se {hop3_qg['q_s0_se']:.3f}, "
        f"mean/se {hop3_qg['q_s0_mean_over_se']:.2f}, {hop3_qg['q_s0_n_pos']}/10 positive).",
        f"Discounted ΔG mean {hop3_qg['discounted_mean']:.3f} (se {hop3_qg['discounted_se']:.3f}, "
        f"mean/se {hop3_qg['discounted_mean_over_se']:.2f}, "
        f"{hop3_qg['discounted_n_pos']}+/{hop3_qg['discounted_n_neg']}-).",
        "A consistent Q rise with a smaller, noisier G drop is start-state",
        "misranking under this continuation. It is not a claim about later states.",
        "",
        "## Readout constraints",
        "",
        "- First-action findings apply to these start states only.",
        "- Hybrid minus full is not a share of return explained by continuation.",
        "- Extra hop-actor optimization is not measured here.",
        "- This cell is tail90 Walker; do not pool with first90 Hopper scores.",
        "",
        "Start-state action swaps under a conservative target continuation did",
        "not reproduce the early terminations seen when rolling out the later",
        "actor for the whole episode. First-action value at s0 does not explain",
        "deployment return of the later actor.",
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
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"refusing non-CPU backend {jax.default_backend()!r}")
    recorded = load_json(args.checkpoints_json)
    mart_entry = recorded[MART_KEY]
    ta_entry = recorded[TA_KEY]
    mart_ckpt = Path(mart_entry["checkpoint"])
    ta_ckpt = Path(ta_entry["checkpoint"])
    mart_payload = load_payload(mart_ckpt)
    ta_payload = load_payload(ta_ckpt)
    for payload, label in ((mart_payload, "MART"), (ta_payload, "two-actor")):
        if "target_actor_params" not in payload:
            raise KeyError(f"{label} checkpoint missing target_actor_params")
    mart_actors = actor_params_list(mart_payload)
    ta_actors = actor_params_list(ta_payload)
    if len(mart_actors) < 3:
        raise ValueError("MART checkpoint needs μ1–μ3")
    if len(ta_actors) < 2:
        raise ValueError("two-actor checkpoint needs target and deployment")

    mart_mean = np.asarray(mart_payload["mean"], dtype=np.float32)
    mart_std = np.asarray(mart_payload["std"], dtype=np.float32)
    ta_mean = np.asarray(ta_payload["mean"], dtype=np.float32)
    ta_std = np.asarray(ta_payload["std"], dtype=np.float32)
    mart_max = float(mart_payload.get("max_action", 1.0))
    ta_max = float(ta_payload.get("max_action", 1.0))
    gamma = float(mart_payload.get("config", {}).get("discount", 0.99))
    ta_gamma = float(ta_payload.get("config", {}).get("discount", 0.99))
    if gamma != ta_gamma:
        raise ValueError(f"discount mismatch MART {gamma} vs two-actor {ta_gamma}")

    common = dict(checkpoint="", sha256="")
    bundles = {
        "mart_mu2": _load_bundle(
            name="mart_mu2",
            checkpoint=mart_ckpt,
            expected_sha=mart_entry["file_sha256"],
            actor_params=mart_actors[1],
            critic_params=mart_payload["critic_params"],
            mean=mart_mean,
            std=mart_std,
            max_action=mart_max,
        ),
        "mart_mu3": _load_bundle(
            name="mart_mu3",
            checkpoint=mart_ckpt,
            expected_sha=mart_entry["file_sha256"],
            actor_params=mart_actors[2],
            critic_params=mart_payload["critic_params"],
            mean=mart_mean,
            std=mart_std,
            max_action=mart_max,
        ),
        "mart_polyak": _load_bundle(
            name="mart_polyak",
            checkpoint=mart_ckpt,
            expected_sha=mart_entry["file_sha256"],
            actor_params=mart_payload["target_actor_params"],
            critic_params=mart_payload["critic_params"],
            mean=mart_mean,
            std=mart_std,
            max_action=mart_max,
        ),
        "ta_deploy": _load_bundle(
            name="ta_deploy",
            checkpoint=ta_ckpt,
            expected_sha=ta_entry["file_sha256"],
            actor_params=ta_actors[1],
            critic_params=ta_payload["critic_params"],
            mean=ta_mean,
            std=ta_std,
            max_action=ta_max,
        ),
        "ta_polyak": _load_bundle(
            name="ta_polyak",
            checkpoint=ta_ckpt,
            expected_sha=ta_entry["file_sha256"],
            actor_params=ta_payload["target_actor_params"],
            critic_params=ta_payload["critic_params"],
            mean=ta_mean,
            std=ta_std,
            max_action=ta_max,
        ),
    }
    del common

    protocols = (
        ("full_mart_mu2", "mart_mu2", "mart_mu2"),
        ("full_mart_mu3", "mart_mu3", "mart_mu3"),
        ("full_ta_deploy", "ta_deploy", "ta_deploy"),
        ("full_mart_polyak", "mart_polyak", "mart_polyak"),
        ("full_ta_polyak", "ta_polyak", "ta_polyak"),
        ("first_mart_mu2_rest_mart_polyak", "mart_mu2", "mart_polyak"),
        ("first_mart_mu2_rest_ta_polyak", "mart_mu2", "ta_polyak"),
        ("first_mart_mu3_rest_mart_polyak", "mart_mu3", "mart_polyak"),
        ("first_mart_mu3_rest_ta_polyak", "mart_mu3", "ta_polyak"),
        ("first_ta_deploy_rest_mart_polyak", "ta_deploy", "mart_polyak"),
        ("first_ta_deploy_rest_ta_polyak", "ta_deploy", "ta_polyak"),
    )
    env = gym.make(EVAL_ENV[_domain(ENV_NAME)])
    episode_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, first_id, rest_id in protocols:
        print(f"[mc] {name}", flush=True)
        for eval_seed in EVAL_EPISODE_SEEDS:
            row = _rollout(
                env,
                first=bundles[first_id],
                rest=bundles[rest_id],
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
    contrasts = [
        _contrast_row(
            hop=3,
            label="MART μ3−μ2 | MART Polyak μ1",
            new_rows=grouped["first_mart_mu3_rest_mart_polyak"],
            ref_rows=grouped["first_mart_mu2_rest_mart_polyak"],
            same_critic=True,
        ),
        _contrast_row(
            hop=3,
            label="MART μ3−μ2 | two-actor Polyak target",
            new_rows=grouped["first_mart_mu3_rest_ta_polyak"],
            ref_rows=grouped["first_mart_mu2_rest_ta_polyak"],
            same_critic=True,
        ),
        _contrast_row(
            hop=2,
            label="MART μ2 − two-actor deploy | MART Polyak μ1",
            new_rows=grouped["first_mart_mu2_rest_mart_polyak"],
            ref_rows=grouped["first_ta_deploy_rest_mart_polyak"],
            same_critic=True,
        ),
        _contrast_row(
            hop=2,
            label="MART μ2 − two-actor deploy | two-actor Polyak target",
            new_rows=grouped["first_mart_mu2_rest_ta_polyak"],
            ref_rows=grouped["first_ta_deploy_rest_ta_polyak"],
            same_critic=True,
        ),
        _contrast_row(
            hop=3,
            label="MART μ3 − two-actor deploy | MART Polyak μ1",
            new_rows=grouped["first_mart_mu3_rest_mart_polyak"],
            ref_rows=grouped["first_ta_deploy_rest_mart_polyak"],
            same_critic=True,
        ),
        _contrast_row(
            hop=3,
            label="MART μ3 − two-actor deploy | two-actor Polyak target",
            new_rows=grouped["first_mart_mu3_rest_ta_polyak"],
            ref_rows=grouped["first_ta_deploy_rest_ta_polyak"],
            same_critic=True,
        ),
        _contrast_row(
            hop=2,
            label="MART Polyak μ1 − two-actor Polyak | first MART μ2",
            new_rows=grouped["first_mart_mu2_rest_mart_polyak"],
            ref_rows=grouped["first_mart_mu2_rest_ta_polyak"],
            same_critic=False,
        ),
        _contrast_row(
            hop=3,
            label="MART Polyak μ1 − two-actor Polyak | first MART μ3",
            new_rows=grouped["first_mart_mu3_rest_mart_polyak"],
            ref_rows=grouped["first_mart_mu3_rest_ta_polyak"],
            same_critic=False,
        ),
        _contrast_row(
            hop=0,
            label="MART Polyak μ1 − two-actor Polyak | first two-actor deploy",
            new_rows=grouped["first_ta_deploy_rest_mart_polyak"],
            ref_rows=grouped["first_ta_deploy_rest_ta_polyak"],
            same_critic=False,
        ),
        _contrast_row(
            hop=2,
            label="full MART μ2 − full two-actor deploy",
            new_rows=grouped["full_mart_mu2"],
            ref_rows=grouped["full_ta_deploy"],
            same_critic=False,
        ),
        _contrast_row(
            hop=3,
            label="full MART μ3 − full two-actor deploy",
            new_rows=grouped["full_mart_mu3"],
            ref_rows=grouped["full_ta_deploy"],
            same_critic=False,
        ),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "episodes.csv",
        episode_rows,
        [
            "environment",
            "T",
            "seed",
            "pair_key",
            "protocol",
            "first_actor",
            "rest_actor",
            "q_source",
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
        args.out_dir / "contrasts.csv",
        contrasts,
        [
            "environment",
            "T",
            "seed",
            "pair_key",
            "hop_index",
            "contrast",
            "same_critic",
            "score_mean",
            "score_se",
            "score_mean_over_se",
            "score_n_pos",
            "score_n_neg",
            "discounted_mean",
            "discounted_se",
            "discounted_mean_over_se",
            "discounted_n_pos",
            "discounted_n_neg",
            "q_s0_mean",
            "q_s0_se",
            "q_s0_mean_over_se",
            "q_s0_n_pos",
            "q_s0_n_neg",
            "length_mean",
        ],
    )
    meta = {
        **CELL,
        "mart_run_key": MART_KEY,
        "two_actor_run_key": TA_KEY,
        "mart_checkpoint": str(mart_ckpt),
        "two_actor_checkpoint": str(ta_ckpt),
        "mart_sha256": bundles["mart_mu2"].sha256,
        "two_actor_sha256": bundles["ta_deploy"].sha256,
        "discount": gamma,
        "eval_seeds": list(EVAL_EPISODE_SEEDS),
        "n_episodes": len(EVAL_EPISODE_SEEDS),
        "n_protocols": len(protocols),
        "shard": "ext_csh_tail90",
        "q_convention": "continuation critic at s0",
        "jax_backend": jax.default_backend(),
    }
    write_json(args.out_dir / "STATUS.json", meta)
    write_report(args.out_dir, protocol_stats, contrasts, meta)
    print(
        json.dumps(
            {"n_episode_rows": len(episode_rows), "backend": jax.default_backend()},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
