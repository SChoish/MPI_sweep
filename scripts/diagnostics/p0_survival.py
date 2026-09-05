"""Episode survival and return decomposition helpers. No JAX."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MAX_EPISODE_STEPS = 1000
SURVIVAL_TIMES = tuple(range(0, MAX_EPISODE_STEPS + 1))


def survival_curve(lengths: Sequence[int], times: Sequence[int] = SURVIVAL_TIMES) -> np.ndarray:
    lengths = np.asarray(list(lengths), dtype=np.int32)
    if lengths.size == 0:
        return np.full(len(times), np.nan)
    return np.array([float(np.mean(lengths >= t)) for t in times], dtype=np.float64)


def pooled_reward_per_step(returns: Sequence[float], lengths: Sequence[int]) -> float:
    total_r = float(np.sum(np.asarray(returns, dtype=np.float64)))
    total_t = float(np.sum(np.asarray(lengths, dtype=np.float64)))
    if total_t <= 0:
        return float("nan")
    return total_r / total_t


def timeout_count(lengths: Sequence[int], horizon: int = MAX_EPISODE_STEPS) -> int:
    return int(np.sum(np.asarray(lengths, dtype=np.int32) >= horizon))


def decompose_episodes(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(r["normalized_score"]) for r in rows]
    returns = [float(r["raw_return"]) for r in rows]
    lengths = [int(float(r["episode_length"])) for r in rows]
    n = len(lengths)
    return {
        "n_episodes": n,
        "normalized_score_mean": float(np.mean(scores)) if n else float("nan"),
        "raw_return_mean": float(np.mean(returns)) if n else float("nan"),
        "episode_length_mean": float(np.mean(lengths)) if n else float("nan"),
        "n_timeout": timeout_count(lengths),
        "timeout_rate": timeout_count(lengths) / n if n else float("nan"),
        "reward_per_step_pooled": pooled_reward_per_step(returns, lengths),
        "reward_per_step_episode_mean": (
            float(np.mean([r / t for r, t in zip(returns, lengths) if t > 0])) if n else float("nan")
        ),
    }


def delta_loss_terms(
    q_new: np.ndarray,
    q_ref: np.ndarray,
    actions_new: np.ndarray,
    actions_ref: np.ndarray,
    c_k: float,
) -> dict[str, float]:
    """ΔL = -c E[Q_new - Q_ref] + E[||a_new - a_ref||^2 / d]. Smaller is better."""
    q_gain = np.asarray(q_new, dtype=np.float64) - np.asarray(q_ref, dtype=np.float64)
    delta = np.asarray(actions_new, dtype=np.float64) - np.asarray(actions_ref, dtype=np.float64)
    transport = float(np.mean(delta * delta))
    q_term = float(-float(c_k) * np.mean(q_gain))
    return {
        "c_k": float(c_k),
        "mean_q_new": float(np.mean(q_new)),
        "mean_q_ref": float(np.mean(q_ref)),
        "mean_q_gain": float(np.mean(q_gain)),
        "transport_mean_sq": transport,
        "delta_L": q_term + transport,
        "q_term": q_term,
    }


def classify_hop(delta_L: float, return_delta: float, return_eps: float = 1.0) -> str:
    obj_better = delta_L < 0.0
    if return_delta > return_eps:
        ret = "return_up"
    elif return_delta < -return_eps:
        ret = "return_down"
    else:
        ret = "return_flat"
    if obj_better and ret == "return_up":
        return "objective_up_return_up"
    if obj_better and ret == "return_down":
        return "objective_up_return_down"
    if (not obj_better) and ret == "return_down":
        return "objective_not_up_return_down"
    if obj_better and ret == "return_flat":
        return "objective_up_return_flat"
    return f"other_{'obj_up' if obj_better else 'obj_not_up'}_{ret}"


def group_key(row: Mapping[str, Any]) -> tuple:
    return (
        row["source"],
        row["environment"],
        int(row["T"]),
        int(row["seed"]),
        row["condition"],
        row["actor_id"],
    )
