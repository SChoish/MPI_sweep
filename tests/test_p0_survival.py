from __future__ import annotations

import numpy as np

from scripts.diagnostics.p0_survival import (
    classify_hop,
    delta_loss_terms,
    pooled_reward_per_step,
    survival_curve,
    timeout_count,
)


def test_survival_uses_length_only():
    curve = survival_curve([400, 1000, 1000], times=[0, 400, 401, 1000, 1001])
    assert curve[0] == 1.0
    assert curve[1] == 1.0
    assert curve[2] == 2 / 3
    assert curve[3] == 2 / 3
    assert curve[4] == 0.0


def test_pooled_reward_rate_is_total_over_total():
    assert pooled_reward_per_step([10.0, 30.0], [5, 15]) == 2.0
    assert timeout_count([999, 1000, 1000]) == 2


def test_delta_L_matches_mean_sq_transport_and_q_term():
    q_new = np.array([3.0, 5.0])
    q_ref = np.array([1.0, 1.0])
    a_new = np.array([[1.0, 0.0], [0.0, 1.0]])
    a_ref = np.zeros_like(a_new)
    c = 0.5
    terms = delta_loss_terms(q_new, q_ref, a_new, a_ref, c)
    assert terms["mean_q_gain"] == 3.0
    assert terms["transport_mean_sq"] == 0.5  # two 1s and two 0s
    assert terms["delta_L"] == -0.5 * 3.0 + 0.5


def test_hopper_style_labels():
    assert classify_hop(-0.01, 30.0) == "objective_up_return_up"
    assert classify_hop(-0.01, -30.0) == "objective_up_return_down"
    assert classify_hop(0.2, -30.0) == "objective_not_up_return_down"
    assert classify_hop(-0.01, 0.2) == "objective_up_return_flat"
