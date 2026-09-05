"""Unit tests for the first-action × continuation 2×2 grid."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from _p0_first_action_continuation import (  # noqa: E402
    CONT_MART_POLYAK,
    CONT_TWO_POLYAK,
    FIRST_MART_HOP,
    FIRST_TWO_DEPLOY,
    baseline_selector,
    discounted_return,
    eval_kinds_for_cell,
    hybrid_cells,
    paired_delta,
    ranking_agreement,
    row_matches,
    table_row_selector,
)


def test_four_hybrid_cells_and_shared_deploy_jobs():
    cells = hybrid_cells()
    assert cells == (
        (FIRST_MART_HOP, CONT_MART_POLYAK),
        (FIRST_MART_HOP, CONT_TWO_POLYAK),
        (FIRST_TWO_DEPLOY, CONT_MART_POLYAK),
        (FIRST_TWO_DEPLOY, CONT_TWO_POLYAK),
    )
    jobs = eval_kinds_for_cell()
    mart = [j for j in jobs if j["first_source"] == FIRST_MART_HOP]
    deploy = [j for j in jobs if j["first_source"] == FIRST_TWO_DEPLOY]
    baselines = [j for j in jobs if j["kind"] == "baseline"]
    assert len(mart) == 4  # hops 2,3 × 2 continuations
    assert len(deploy) == 2  # shared across hops
    assert len(baselines) == 2
    assert {j["hop"] for j in mart} == {2, 3}
    assert all(j["hop"] is None for j in deploy)


def test_table_selector_holds_one_factor_fixed():
    hop2_mart_mu1 = table_row_selector(2, FIRST_MART_HOP, CONT_MART_POLYAK)
    hop2_deploy_two = table_row_selector(2, FIRST_TWO_DEPLOY, CONT_TWO_POLYAK)
    assert hop2_mart_mu1["hop"] == 2
    assert hop2_deploy_two["hop"] is None
    mart_row = {
        "first_source": FIRST_MART_HOP,
        "continuation_source": CONT_MART_POLYAK,
        "hop": 2,
    }
    deploy_row = {
        "first_source": FIRST_TWO_DEPLOY,
        "continuation_source": CONT_TWO_POLYAK,
        "hop": "",
    }
    assert row_matches(mart_row, hop2_mart_mu1)
    assert row_matches(deploy_row, hop2_deploy_two)
    assert not row_matches(mart_row, table_row_selector(3, FIRST_MART_HOP, CONT_MART_POLYAK))
    assert row_matches(
        {"first_source": CONT_MART_POLYAK, "continuation_source": CONT_MART_POLYAK, "hop": None},
        baseline_selector(CONT_MART_POLYAK),
    )


def test_discounted_return_and_ranking():
    rewards = [1.0, 1.0, 1.0]
    assert discounted_return(rewards, 1.0) == 3.0
    np.testing.assert_allclose(discounted_return(rewards, 0.5), 1.0 + 0.5 + 0.25)
    delta = paired_delta([10.0, 8.0, 6.0], [9.0, 9.0, 6.0])
    assert delta["n"] == 3
    np.testing.assert_allclose(delta["mean"], 0.0)
    assert delta["n_positive"] == 1
    assert delta["n_negative"] == 1
    rank = ranking_agreement([1.0, 2.0, -1.0], [-3.0, 4.0, -2.0])
    assert rank["n_q_pos_g_neg"] == 1
    assert rank["n_sign_agree"] == 2
    assert rank["n_sign_disagree"] == 1
