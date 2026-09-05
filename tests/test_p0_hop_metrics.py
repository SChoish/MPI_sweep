"""Unit tests for hop-audit survival, return split, and JKO ΔL."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from _p0_hop_metrics import (  # noqa: E402
    classify_obj_vs_return,
    hop_objective_delta,
    hop_q_weight,
    mean_return_attribution,
    pooled_reward_per_step,
    survival_curve,
    timeout_count,
)


def test_survival_is_empirical_tail():
    lengths = [400, 700, 1000, 1000]
    grid = [1, 400, 701, 1000, 1001]
    surv = survival_curve(lengths, grid)
    np.testing.assert_allclose(surv, [1.0, 1.0, 0.5, 0.5, 0.0])


def test_pooled_reward_is_total_over_total_steps():
    # Two episodes: (10 over 2 steps) and (20 over 8 steps) → 30/10 = 3.0
    assert pooled_reward_per_step([10.0, 20.0], [2, 8]) == 3.0
    assert np.isnan(pooled_reward_per_step([], []))


def test_timeout_counts_horizon_inclusive():
    assert timeout_count([999, 1000, 1000], 1000) == 2


def test_length_term_dominates_when_rate_barely_moves():
    # Matches the hopper-medium / expert story: r falls slightly, L moves a lot.
    split = mean_return_attribution(
        r_before=3.3, length_before=650.0, r_after=3.25, length_after=1000.0
    )
    assert split["length_term"] > 0
    assert split["rate_term"] < 0
    assert split["abs_length_share"] > 0.9


def test_jko_delta_matches_closed_form():
    q_prev = np.array([10.0, 12.0, 8.0])
    q_k = q_prev + 1.0
    a_prev = np.zeros((3, 2))
    a_k = np.ones((3, 2)) * 0.5
    tau = 2.5
    out = hop_objective_delta(q_k, q_prev, a_k, a_prev, tau_step=tau)
    lam = hop_q_weight(q_prev, tau)
    w2 = float(np.mean(np.square(a_k - a_prev)))
    np.testing.assert_allclose(out["q_weight"], lam)
    np.testing.assert_allclose(out["delta_L"], -lam * 1.0 + w2)
    assert out["objective_improved"] is True


def test_w2_is_mean_over_action_coordinates():
    q = np.ones(2)
    a_prev = np.zeros((2, 3))
    a_k = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = hop_objective_delta(q, q, a_k, a_prev, tau_step=1.0, scale_norm=False)
    # mean(square) = 1 / (2*3) = 1/6, which is E[||Δ||^2 / d]
    np.testing.assert_allclose(out["w2"], 1.0 / 6.0)


def test_classify_obj_up_return_down():
    assert classify_obj_vs_return(delta_L=-0.01, delta_return=-31.0) == "objective_improved_return_down"
    assert classify_obj_vs_return(delta_L=-0.01, delta_return=36.0) == "objective_improved_return_up"
    assert classify_obj_vs_return(delta_L=0.02, delta_return=-31.0) == "objective_not_improved_return_down"
    assert classify_obj_vs_return(delta_L=-0.01, delta_return=0.5) == "objective_improved_return_flat"


def test_q_gain_move_ratio_exceeds_one_iff_delta_L_negative():
    from _p0_hop_metrics import q_gain_move_ratio

    c_k, delta_q, w2 = 0.02, 0.16, 0.004
    ratio = q_gain_move_ratio(c_k, delta_q, w2)
    np.testing.assert_allclose(ratio, 0.8)
    assert q_gain_move_ratio(0.02, 0.16, 0.002) > 1.0
