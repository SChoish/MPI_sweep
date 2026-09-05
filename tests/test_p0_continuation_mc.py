from scripts.diagnostics.p0_continuation_mc import (
    bootstrap_truncated,
    discounted_return,
    paired_deltas,
)


def test_discounted_return_is_backward_sum():
    assert discounted_return([1.0, 1.0], 0.5) == 1.0 + 0.5 * 1.0


def test_bootstrap_only_when_truncated():
    rewards = [2.0]
    assert bootstrap_truncated(rewards, 0.5, truncated=False, bootstrap_q=10.0) == 2.0
    assert bootstrap_truncated(rewards, 0.5, truncated=True, bootstrap_q=10.0) == 2.0 + 0.5 * 10.0


def test_paired_deltas_are_new_minus_ref():
    out = paired_deltas([3.0, 1.0], [1.0, 1.0])
    assert out["n"] == 2
    assert out["mean"] == 1.0
    assert out["n_pos"] == 1
    assert out["n_neg"] == 0
