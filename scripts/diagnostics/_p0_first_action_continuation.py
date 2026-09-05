"""First-action × continuation grid for MART hops vs two-actor.

JAX-free. The runner evaluates the 2×2 on shared reset seeds. Continuation
uses the Polyak target actor from each method's checkpoint, not the online
first actor.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HOPPER_T10_TASKS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-expert-v2",
)
HOPS = (2, 3)
FIRST_MART_HOP = "mart_hop"
FIRST_TWO_DEPLOY = "two_actor_deploy"
CONT_MART_POLYAK = "mart_polyak_mu1"
CONT_TWO_POLYAK = "two_actor_polyak_target"
HYBRID_FIRST = (FIRST_MART_HOP, FIRST_TWO_DEPLOY)
HYBRID_CONT = (CONT_MART_POLYAK, CONT_TWO_POLYAK)
BASELINE_FIRST = {
    CONT_MART_POLYAK: CONT_MART_POLYAK,
    CONT_TWO_POLYAK: CONT_TWO_POLYAK,
}
DISCOUNT = 0.99
HORIZON = 1000
EVAL_SEED_BASE = 10_000


def hybrid_cells() -> tuple[tuple[str, str], ...]:
    return tuple((first, cont) for first in HYBRID_FIRST for cont in HYBRID_CONT)


def eval_kinds_for_cell() -> list[dict[str, Any]]:
    """Unique simulator jobs per (task, seed). Deploy-first is shared across hops."""
    jobs = []
    for hop in HOPS:
        for cont in HYBRID_CONT:
            jobs.append(
                {
                    "kind": "hybrid",
                    "hop": hop,
                    "first_source": FIRST_MART_HOP,
                    "continuation_source": cont,
                }
            )
    for cont in HYBRID_CONT:
        jobs.append(
            {
                "kind": "hybrid",
                "hop": None,
                "first_source": FIRST_TWO_DEPLOY,
                "continuation_source": cont,
            }
        )
    for cont in HYBRID_CONT:
        jobs.append(
            {
                "kind": "baseline",
                "hop": None,
                "first_source": BASELINE_FIRST[cont],
                "continuation_source": cont,
            }
        )
    return jobs


def job_key(job: Mapping[str, Any]) -> tuple:
    hop = job.get("hop")
    return (
        job["first_source"],
        job["continuation_source"],
        None if hop is None else int(hop),
    )


def table_row_selector(hop: int, first: str, cont: str) -> dict[str, Any]:
    """Which stored jobs fill one 2×2 cell for a MART hop."""
    if first == FIRST_MART_HOP:
        return {"first_source": FIRST_MART_HOP, "continuation_source": cont, "hop": hop}
    return {"first_source": FIRST_TWO_DEPLOY, "continuation_source": cont, "hop": None}


def baseline_selector(cont: str) -> dict[str, Any]:
    return {
        "first_source": BASELINE_FIRST[cont],
        "continuation_source": cont,
        "hop": None,
    }


def row_matches(row: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    if row["first_source"] != selector["first_source"]:
        return False
    if row["continuation_source"] != selector["continuation_source"]:
        return False
    hop = selector.get("hop")
    stored = row.get("hop")
    if hop is None:
        return stored in (None, "", "None")
    return int(stored) == int(hop)


def discounted_return(rewards: Sequence[float], discount: float = DISCOUNT) -> float:
    total = 0.0
    scale = 1.0
    for reward in rewards:
        total += scale * float(reward)
        scale *= float(discount)
    return float(total)


def paired_delta(
    treatment: Sequence[float],
    baseline: Sequence[float],
) -> dict[str, float]:
    t = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if t.shape != b.shape or t.size == 0:
        raise ValueError("paired_delta needs nonempty aligned arrays")
    delta = t - b
    return {
        "n": float(delta.size),
        "mean": float(np.mean(delta)),
        "std": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
        "n_negative": float(np.sum(delta < 0)),
        "n_positive": float(np.sum(delta > 0)),
        "n_zero": float(np.sum(delta == 0)),
    }


def ranking_agreement(delta_q: Sequence[float], delta_g: Sequence[float]) -> dict[str, float]:
    q = np.asarray(delta_q, dtype=np.float64)
    g = np.asarray(delta_g, dtype=np.float64)
    if q.shape != g.shape or q.size == 0:
        raise ValueError("ranking_agreement needs nonempty aligned arrays")
    q_sign = np.sign(q)
    g_sign = np.sign(g)
    both_nonzero = (q_sign != 0) & (g_sign != 0)
    agree = q_sign == g_sign
    return {
        "n": float(q.size),
        "n_sign_agree": float(np.sum(agree)),
        "n_sign_disagree": float(np.sum((q_sign != g_sign) & both_nonzero)),
        "n_q_pos_g_neg": float(np.sum((q > 0) & (g < 0))),
        "n_q_neg_g_pos": float(np.sum((q < 0) & (g > 0))),
        "mean_delta_q": float(np.mean(q)),
        "mean_delta_g": float(np.mean(g)),
        "std_delta_g": float(np.std(g, ddof=1)) if g.size > 1 else 0.0,
    }


def group_episodes(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(dict(row))
    return dict(grouped)


def mean_score(rows: Sequence[Mapping[str, Any]], field: str = "normalized_score") -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(r[field]) for r in rows]))
