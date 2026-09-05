"""Pure helpers for hop-actor then Polyak-μ1 continuation MC. No JAX."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    total = 0.0
    for reward in reversed(list(rewards)):
        total = float(reward) + float(gamma) * total
    return total


def bootstrap_truncated(
    rewards: Sequence[float],
    gamma: float,
    truncated: bool,
    bootstrap_q: float,
) -> float:
    """Discounted return; if truncated, add γ^T Q(s_T, a_cont)."""
    base = discounted_return(rewards, gamma)
    if not truncated:
        return base
    horizon = len(rewards)
    return base + (float(gamma) ** horizon) * float(bootstrap_q)


def paired_deltas(new_vals: Sequence[float], ref_vals: Sequence[float]) -> dict[str, float]:
    """Episode-paired new − ref. ``mean_over_se`` is a scale, not a p-value."""
    new = np.asarray(list(new_vals), dtype=np.float64)
    ref = np.asarray(list(ref_vals), dtype=np.float64)
    if new.shape != ref.shape:
        raise ValueError("paired_deltas requires aligned sequences")
    delta = new - ref
    n = int(delta.size)
    mean = float(np.mean(delta)) if n else float("nan")
    std = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    se = std / float(np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "mean_over_se": (mean / se) if se and se > 0 else float("nan"),
        "n_pos": int(np.sum(delta > 0.0)),
        "n_neg": int(np.sum(delta < 0.0)),
    }
