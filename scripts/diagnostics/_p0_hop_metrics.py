"""Pure-numpy hop-audit metrics. No JAX, no checkpoints."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HOPPER_TASKS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
)
DEFAULT_HORIZON = 1000
Q_WEIGHT_EPS = 1e-6


def survival_curve(lengths: Sequence[float], t_grid: Sequence[int]) -> np.ndarray:
    """S(t) = Pr(L >= t) on a fixed time grid."""
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.size == 0:
        return np.full(len(t_grid), np.nan, dtype=np.float64)
    grid = np.asarray(t_grid, dtype=np.float64)
    return np.mean(lengths[None, :] >= grid[:, None], axis=1)


def survival_grid(horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    return np.arange(1, int(horizon) + 1, dtype=np.int32)


def pooled_reward_per_step(raw_returns: Sequence[float], lengths: Sequence[float]) -> float:
    """sum reward / sum steps. Empty or zero-length total is NaN."""
    ret = np.asarray(raw_returns, dtype=np.float64)
    length = np.asarray(lengths, dtype=np.float64)
    total_steps = float(np.sum(length))
    if ret.size == 0 or total_steps <= 0:
        return float("nan")
    return float(np.sum(ret) / total_steps)


def timeout_count(lengths: Sequence[float], horizon: int = DEFAULT_HORIZON) -> int:
    return int(np.sum(np.asarray(lengths, dtype=np.float64) >= horizon))


def mean_return_attribution(
    r_before: float,
    length_before: float,
    r_after: float,
    length_after: float,
) -> dict[str, float]:
    """Split Δ mean raw return into length vs per-step terms.

    mean_return = r * mean_length with r = sum(R)/sum(L).
    Δ = r_before * ΔL + L_after * Δr.
    """
    delta_length = float(length_after - length_before)
    delta_r = float(r_after - r_before)
    length_term = float(r_before * delta_length)
    rate_term = float(length_after * delta_r)
    return {
        "delta_mean_raw": length_term + rate_term,
        "length_term": length_term,
        "rate_term": rate_term,
        "abs_length_share": float(
            abs(length_term) / (abs(length_term) + abs(rate_term) + 1e-12)
        ),
    }


def hop_q_weight(q_ref: Sequence[float], tau_step: float, scale_norm: bool = True) -> float:
    """Match train_td3bc.normalized_q_weight on a numpy vector of Q(s, μ_{k-1})."""
    q_ref = np.asarray(q_ref, dtype=np.float64)
    alpha = 2.0 * float(tau_step)
    if not scale_norm:
        return alpha
    return float(alpha / (np.mean(np.abs(q_ref)) + Q_WEIGHT_EPS))


def hop_objective_delta(
    q_k: Sequence[float],
    q_prev: Sequence[float],
    actions_k: np.ndarray,
    actions_prev: np.ndarray,
    tau_step: float,
    scale_norm: bool = True,
) -> dict[str, float]:
    """JKO hop loss of μ_k minus the loss of keeping μ_{k-1}.

    Implementation match: ``-λ mean(Q) + mean((a-a_ref)^2)`` with
    ``λ = 2 τ_step / mean(|Q_ref|)``. ``mean(square)`` already averages over
    action coordinates, i.e. E[||Δμ||^2 / d].
    Negative ``delta_L`` is an objective improvement.
    """
    q_k = np.asarray(q_k, dtype=np.float64)
    q_prev = np.asarray(q_prev, dtype=np.float64)
    actions_k = np.asarray(actions_k, dtype=np.float64)
    actions_prev = np.asarray(actions_prev, dtype=np.float64)
    lam = hop_q_weight(q_prev, tau_step, scale_norm=scale_norm)
    mean_q_k = float(np.mean(q_k))
    mean_q_prev = float(np.mean(q_prev))
    w2 = float(np.mean(np.square(actions_k - actions_prev)))
    delta_q = mean_q_k - mean_q_prev
    loss_k = -lam * mean_q_k + w2
    loss_stay = -lam * mean_q_prev
    return {
        "q_weight": lam,
        "mean_q_k": mean_q_k,
        "mean_q_prev": mean_q_prev,
        "delta_q": float(delta_q),
        "w2": w2,
        "loss_k": float(loss_k),
        "loss_stay": float(loss_stay),
        "delta_L": float(loss_k - loss_stay),
        "q_term": float(-lam * delta_q),
        "q_gain_move_ratio": q_gain_move_ratio(lam, delta_q, w2),
        "objective_improved": bool(loss_k < loss_stay),
    }


def q_gain_move_ratio(c_k: float, delta_q: float, w2: float) -> float:
    """(c_k ΔQ) / E[||Δμ||^2 / d]. Ratio > 1 iff ΔL < 0."""
    move = float(w2)
    gain = float(c_k) * float(delta_q)
    if move <= 0.0:
        return float("inf") if gain > 0.0 else float("nan")
    return float(gain / move)


def classify_obj_vs_return(
    delta_L: float,
    delta_return: float,
    return_flat: float = 3.0,
) -> str:
    """Map (ΔL, ΔJ) onto the four readout rows. Lower L is better."""
    obj_up = delta_L < 0.0
    ret_up = delta_return > return_flat
    ret_down = delta_return < -return_flat
    if obj_up and ret_up:
        return "objective_improved_return_up"
    if obj_up and ret_down:
        return "objective_improved_return_down"
    if (not obj_up) and ret_down:
        return "objective_not_improved_return_down"
    if obj_up and not ret_up and not ret_down:
        return "objective_improved_return_flat"
    if (not obj_up) and ret_up:
        return "objective_not_improved_return_up"
    return "objective_not_improved_return_flat"


def group_episodes(episodes: Iterable[Mapping[str, Any]]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in episodes:
        key = (
            row["task"],
            int(row["T"]),
            int(row["training_seed"]),
            row["method"],
            int(row["actor_index"]),
            row["actor_role"],
        )
        groups[key].append(dict(row))
    return groups


def decompose_group(
    rows: Sequence[Mapping[str, Any]],
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    raw = [float(r["raw_return"]) for r in rows]
    scores = [float(r["normalized_score"]) for r in rows]
    lengths = [float(r["episode_length"]) for r in rows]
    return {
        "n_episodes": len(rows),
        "mean_normalized": float(np.mean(scores)) if scores else float("nan"),
        "mean_raw_return": float(np.mean(raw)) if raw else float("nan"),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "n_timeout": timeout_count(lengths, horizon),
        "pooled_reward_per_step": pooled_reward_per_step(raw, lengths),
        "mean_episode_reward_per_step": float(
            np.mean([r / L for r, L in zip(raw, lengths) if L > 0])
        )
        if rows
        else float("nan"),
    }
