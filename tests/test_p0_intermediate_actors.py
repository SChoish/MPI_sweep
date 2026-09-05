from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.diagnostics.p0_intermediate_actors import (
    ACTORS_BY_CONDITION,
    ENVIRONMENTS,
    bound_fraction,
    cosine_with_near_zero_mask,
    coverage_stats,
    hop_tau,
    planned_runs,
    rms_action_difference,
    run_key,
)


def test_planned_grid_is_180_runs_with_six_policies_per_cell():
    rows = planned_runs()
    assert len(rows) == 180
    assert len(ENVIRONMENTS) == 9
    mart = [row for row in rows if row["condition"] == "bar_p4"]
    two = [row for row in rows if row["condition"] == "two_actor_p4"]
    assert len(mart) == 90 and len(two) == 90
    assert [a.actor_id for a in ACTORS_BY_CONDITION["bar_p4"]] == [
        "mart_mu1",
        "mart_mu2",
        "mart_mu3",
        "mart_mu4",
    ]
    assert [a.actor_id for a in ACTORS_BY_CONDITION["two_actor_p4"]] == [
        "two_actor_target",
        "two_actor_deploy",
    ]
    assert ACTORS_BY_CONDITION["bar_p4"][0].enters_bellman
    assert not ACTORS_BY_CONDITION["bar_p4"][3].enters_bellman
    assert hop_tau(20.0, "T/K") == 5.0
    assert hop_tau(20.0, "T") == 20.0


def test_rms_is_mean_over_action_dim():
    a = np.zeros((2, 4))
    b = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rms = rms_action_difference(a, b)
    assert rms[0] == np.sqrt((4.0) / 4.0)
    assert rms[1] == 0.0


def test_near_zero_cosine_is_excluded_not_filled():
    u = np.array([[1.0, 0.0], [0.0, 0.0]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    cosine, valid = cosine_with_near_zero_mask(u, v, min_l2=1e-6)
    assert valid[0] and not valid[1]
    assert cosine[0] == 1.0
    assert np.isnan(cosine[1])


def test_coverage_does_not_count_missing_as_paired():
    rows = []
    for env in ENVIRONMENTS[:2]:
        rows.append(
            {
                "condition": "bar_p4",
                "environment": env,
                "T": 4,
                "seed": 0,
                "eval_eligible": True,
                "actors": ACTORS_BY_CONDITION["bar_p4"],
            }
        )
    rows.append(
        {
            "condition": "two_actor_p4",
            "environment": ENVIRONMENTS[0],
            "T": 4,
            "seed": 0,
            "eval_eligible": True,
            "actors": ACTORS_BY_CONDITION["two_actor_p4"],
        }
    )
    stats = coverage_stats(rows)
    assert stats["present_runs"] == 3
    assert stats["paired_cells_both_methods"] == 1
    assert stats["planned_runs"] == 180
    assert run_key("bar_p4", "hopper-medium-v2", 4, 0) == "bar_p4|hopper-medium-v2|T=4|seed=0"


def test_bound_fraction_uses_abs_threshold():
    actions = np.array([[1.0, 0.0], [0.96, -0.96]])
    assert bound_fraction(actions, max_action=1.0, frac=0.95) == 0.75
