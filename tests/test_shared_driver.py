import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train_td3bc import (
    DEPLOYMENT_BRANCHES,
    Transition,
    apply_deployment_branch,
    create_mcep_train_state,
    create_train_state,
    evaluation_contract,
    update_n_times,
)


def _toy_data():
    observations = jnp.arange(48, dtype=jnp.float32).reshape(16, 3) / 10
    actions = jnp.zeros((16, 2), dtype=jnp.float32)
    data = Transition(
        observations=observations,
        actions=actions,
        rewards=jnp.ones((16, 1), dtype=jnp.float32),
        next_observations=observations,
        not_dones=jnp.ones((16, 1), dtype=jnp.float32),
    )
    return observations, actions, data


def _leaves_equal(actual, expected):
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def _step(state, data, *, method, reference_depth, deployment_branch=None):
    return update_n_times(
        state,
        data,
        jax.random.PRNGKey(1),
        start_it=jnp.asarray(0),
        n_updates=1,
        batch_size=4,
        discount=0.99,
        tau=12.0,
        polyak=0.005,
        policy_freq=1,
        scale_norm=True,
        method=method,
        reference_depth=reference_depth,
        deployment_branch=deployment_branch,
    )


def test_shared_eval_columns_follow_existing_contracts():
    recenter = evaluation_contract("shared", 4, "recenter")
    fixed_ref = evaluation_contract("shared", 4, "fixed_ref")
    anchor = evaluation_contract("shared", 4, "data_anchor")
    assert recenter.columns == evaluation_contract("bar", 4).columns
    assert fixed_ref.columns == evaluation_contract("bar", 4).columns
    assert anchor.columns == evaluation_contract("mcep", 4).columns
    assert set(DEPLOYMENT_BRANCHES) == {
        "recenter",
        "data_anchor",
        "fixed_ref",
        "data_anchor_matched",
    }


def test_shared_branches_keep_first_actor_critic_and_targets_identical():
    observations, actions, data = _toy_data()
    bar_state = create_train_state(
        jax.random.PRNGKey(0),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=4,
    )
    two_actor_state = create_mcep_train_state(
        jax.random.PRNGKey(0),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        reference_depth=4,
    )

    recenter, _ = _step(
        bar_state, data, method="shared", reference_depth=4, deployment_branch="recenter"
    )
    fixed_ref, _ = _step(
        bar_state, data, method="shared", reference_depth=4, deployment_branch="fixed_ref"
    )
    data_anchor, _ = _step(
        two_actor_state,
        data,
        method="shared",
        reference_depth=4,
        deployment_branch="data_anchor",
    )
    matched, _ = _step(
        two_actor_state,
        data,
        method="shared",
        reference_depth=4,
        deployment_branch="data_anchor_matched",
    )
    bar, _ = _step(bar_state, data, method="bar", reference_depth=None)
    control, _ = _step(
        two_actor_state, data, method="mcep", reference_depth=4
    )

    shared = (
        (recenter.actors[0].params, fixed_ref.actors[0].params),
        (recenter.actors[0].params, data_anchor.actors[0].params),
        (recenter.actors[0].params, matched.actors[0].params),
        (recenter.actors[0].params, bar.actors[0].params),
        (recenter.actors[0].params, control.actors[0].params),
        (recenter.critic.params, fixed_ref.critic.params),
        (recenter.critic.params, data_anchor.critic.params),
        (recenter.critic.params, bar.critic.params),
        (recenter.critic.params, control.critic.params),
        (recenter.target_actor.params, data_anchor.target_actor.params),
        (recenter.target_critic.params, data_anchor.target_critic.params),
    )
    for actual, expected in shared:
        _leaves_equal(actual, expected)

    _leaves_equal(tuple(a.params for a in recenter.actors), tuple(a.params for a in bar.actors))
    _leaves_equal(
        tuple(a.params for a in data_anchor.actors),
        tuple(a.params for a in control.actors),
    )
    with pytest.raises(AssertionError):
        _leaves_equal(recenter.actors[-1].params, fixed_ref.actors[-1].params)
    with pytest.raises(AssertionError):
        _leaves_equal(data_anchor.actors[1].params, matched.actors[1].params)


def test_deployment_branch_does_not_touch_shared_modules():
    observations, actions, data = _toy_data()
    state = create_train_state(
        jax.random.PRNGKey(3),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=3,
    )
    batch = jax.tree_util.tree_map(lambda x: x[:4], data)
    after, _ = apply_deployment_branch(
        state, batch, "recenter", 4.0, 12.0, True, hop_count=3
    )
    _leaves_equal(after.actors[0].params, state.actors[0].params)
    _leaves_equal(after.critic.params, state.critic.params)
    _leaves_equal(after.target_actor.params, state.target_actor.params)
    _leaves_equal(after.target_critic.params, state.target_critic.params)
    with pytest.raises(AssertionError):
        _leaves_equal(after.actors[-1].params, state.actors[-1].params)
