from argparse import Namespace
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tau_grids import MPI_TAU_GRID, mpi_tau_grid
from train_td3bc import (
    Transition,
    create_train_state,
    create_mcep_train_state,
    evaluation_contract,
    latest_checkpoint,
    load_checkpoint,
    normalized_q_weight,
    mcep_actor_taus,
    projected_explicit_target,
    qlearning_from_hdf5,
    restore_train_state,
    save_checkpoint,
    update_in_blocks,
    update_n_times,
)


def test_normalized_q_objective_is_scale_invariant():
    q = jnp.asarray([1.0, 3.0])
    scaled_q = 100.0 * q

    def objective(values):
        return -normalized_q_weight(values, tau=1.25) * jnp.mean(values)

    assert float(objective(q)) == pytest.approx(float(objective(scaled_q)))


def test_unnormalized_weight_is_two_tau():
    q = jnp.asarray([1.0, -3.0])
    assert float(normalized_q_weight(q, tau=1.25, scale_norm=False)) == pytest.approx(2.5)


def test_two_actor_budget_mapping_uses_reference_hop_count():
    target_tau, evaluation_tau = mcep_actor_taus(12.0, reference_depth=3)
    assert target_tau == pytest.approx(4.0)
    assert evaluation_tau == pytest.approx(12.0)
    p4_target_tau, p4_evaluation_tau = mcep_actor_taus(20.0, reference_depth=4)
    assert p4_target_tau == pytest.approx(5.0)
    assert p4_evaluation_tau == pytest.approx(20.0)


def test_bar_and_two_actor_eval_columns_keep_target_and_deployment_distinct():
    bar = evaluation_contract("bar", mpi_steps=4)
    control = evaluation_contract("mcep", mpi_steps=4)

    assert bar.target_actor_index == control.target_actor_index == 0
    assert bar.deployment_actor_index == control.deployment_actor_index == -1
    assert bar.columns == (
        "step", "return", "d4rl_score", "critic_loss", "actor_loss",
        "return_pi4", "d4rl_pi4", "final_actor_loss",
    )
    assert control.columns == (
        "step", "return", "d4rl_score", "critic_loss", "actor_loss",
        "return_eval", "d4rl_eval", "final_actor_loss",
    )

def test_mcep_initialization_matches_bar_target_final_and_critic_keys():
    observations = jnp.zeros((1, 3), dtype=jnp.float32)
    actions = jnp.zeros((1, 2), dtype=jnp.float32)
    kwargs = dict(
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
    )
    rng = jax.random.PRNGKey(7)
    bar = create_train_state(rng, observations, actions, mpi_steps=3, **kwargs)
    mcep = create_mcep_train_state(
        rng, observations, actions, reference_depth=3, **kwargs
    )

    matched = (
        (mcep.actors[0].params, bar.actors[0].params),
        (mcep.actors[1].params, bar.actors[-1].params),
        (mcep.critic.params, bar.critic.params),
    )
    for actual_tree, expected_tree in matched:
        for actual, expected in zip(
            jax.tree_util.tree_leaves(actual_tree),
            jax.tree_util.tree_leaves(expected_tree),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

def test_projected_explicit_target_matches_mean_action_metric_scale():
    observations = jnp.zeros((2, 1), dtype=jnp.float32)
    references = jnp.zeros((2, 3), dtype=jnp.float32)

    def linear_critic(_params, _observations, actions):
        q1 = 2.0 + jnp.sum(actions, axis=-1, keepdims=True)
        return q1, q1

    target = projected_explicit_target(
        linear_critic,
        None,
        observations,
        references,
        tau_step=0.5,
        scale_norm=True,
        max_action=10.0,
    )

    # d * h / q_scale = 3 * 0.5 / 2 = 0.75 for every coordinate.
    np.testing.assert_allclose(np.asarray(target), 0.75, rtol=1e-5, atol=1e-5)


def test_tau_grid_is_stable_and_validated():
    assert mpi_tau_grid(3) == ["0.05", "0.1", "0.2"]
    assert len(mpi_tau_grid(100)) == len(MPI_TAU_GRID)
    with pytest.raises(ValueError):
        mpi_tau_grid(0)


def test_latest_checkpoint_ignores_unrelated_pickles(tmp_path: Path):
    (tmp_path / "params_8.pkl").touch()
    (tmp_path / "params_100.pkl").touch()
    (tmp_path / "params_latest.pkl").touch()
    (tmp_path / "other_200.pkl").touch()
    assert latest_checkpoint(tmp_path) == tmp_path / "params_100.pkl"


def test_hdf5_conversion_drops_timeout_boundaries(tmp_path: Path):
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.arange(10, dtype=np.float32).reshape(5, 2)
        handle["actions"] = np.arange(5, dtype=np.float32)[:, None]
        handle["rewards"] = np.arange(5, dtype=np.float32)
        handle["terminals"] = np.asarray([0, 1, 0, 0, 0], dtype=np.float32)
        handle["timeouts"] = np.asarray([0, 0, 1, 0, 0], dtype=np.float32)

    data = qlearning_from_hdf5(path)

    assert data["observations"].shape == (3, 2)
    assert data["actions"][:, 0].tolist() == [0.0, 1.0, 3.0]
    assert data["not_dones"][:, 0].tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("integrator", ["implicit", "explicit"])
def test_arbitrary_four_hop_update_and_checkpoint_smoke(
    tmp_path: Path, integrator: str
):
    observations = jnp.zeros((16, 3), dtype=jnp.float32)
    actions = jnp.zeros((16, 2), dtype=jnp.float32)
    data = Transition(
        observations=observations,
        actions=actions,
        rewards=jnp.zeros((16, 1), dtype=jnp.float32),
        next_observations=observations,
        not_dones=jnp.ones((16, 1), dtype=jnp.float32),
    )
    state = create_train_state(
        jax.random.PRNGKey(0),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=4,
    )

    updated, metrics = update_n_times(
        state,
        data,
        jax.random.PRNGKey(1),
        start_it=jnp.asarray(0),
        n_updates=1,
        batch_size=4,
        discount=0.99,
        tau=1.5,
        polyak=0.005,
        policy_freq=1,
        scale_norm=True,
        integrator=integrator,
    )

    assert len(updated.actors) == 4
    assert all(actor.step == 1 for actor in updated.actors)
    assert set(metrics) == {
        "critic_loss",
        "actor_loss",
        "final_actor_loss",
    }

    checkpoint = tmp_path / "params_1.pkl"
    save_checkpoint(
        checkpoint,
        updated,
        step=1,
        rng=jax.random.PRNGKey(2),
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
        args=Namespace(mpi_steps=4),
    )
    restored = restore_train_state(state, load_checkpoint(checkpoint))
    assert len(restored.actors) == 4
    assert all(actor.step == 1 for actor in restored.actors)


def test_mcep_two_actor_update_smoke():
    observations = jnp.zeros((16, 3), dtype=jnp.float32)
    actions = jnp.zeros((16, 2), dtype=jnp.float32)
    data = Transition(
        observations=observations,
        actions=actions,
        rewards=jnp.zeros((16, 1), dtype=jnp.float32),
        next_observations=observations,
        not_dones=jnp.ones((16, 1), dtype=jnp.float32),
    )
    state = create_mcep_train_state(
        jax.random.PRNGKey(0),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        reference_depth=3,
    )
    bar_state = create_train_state(
        jax.random.PRNGKey(0),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=3,
    )
    updated, metrics = update_n_times(
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
        method="mcep",
        reference_depth=3,
    )

    bar_updated, _ = update_n_times(
        bar_state,
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
    )
    assert len(updated.actors) == 2
    assert all(actor.step == 1 for actor in updated.actors)
    assert set(metrics) == {"critic_loss", "actor_loss", "final_actor_loss"}

    shared_trajectory = (
        (updated.actors[0].params, bar_updated.actors[0].params),
        (updated.critic.params, bar_updated.critic.params),
        (updated.target_actor.params, bar_updated.target_actor.params),
        (updated.target_critic.params, bar_updated.target_critic.params),
    )
    for actual_tree, expected_tree in shared_trajectory:
        for actual, expected in zip(
            jax.tree_util.tree_leaves(actual_tree),
            jax.tree_util.tree_leaves(expected_tree),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


def test_fused_dispatch_preserves_block_rng_and_updates():
    observations = jnp.arange(48, dtype=jnp.float32).reshape(16, 3) / 10
    actions = jnp.zeros((16, 2), dtype=jnp.float32)
    data = Transition(
        observations=observations,
        actions=actions,
        rewards=jnp.ones((16, 1), dtype=jnp.float32),
        next_observations=observations,
        not_dones=jnp.ones((16, 1), dtype=jnp.float32),
    )
    state = create_train_state(
        jax.random.PRNGKey(10),
        observations[:1],
        actions[:1],
        max_action=1.0,
        lr=3e-4,
        policy_noise=0.2,
        noise_clip=0.5,
        mpi_steps=2,
    )

    baseline = state
    baseline_rng = jax.random.PRNGKey(11)
    for block in range(2):
        baseline_rng, update_rng = jax.random.split(baseline_rng)
        baseline, _ = update_n_times(
            baseline,
            data,
            update_rng,
            start_it=jnp.asarray(block),
            n_updates=1,
            batch_size=4,
            discount=0.99,
            tau=0.01,
            polyak=0.005,
            policy_freq=1,
            scale_norm=False,
        )

    fused, fused_rng, _ = update_in_blocks(
        state,
        data,
        jax.random.PRNGKey(11),
        start_it=jnp.asarray(0),
        tau=jnp.asarray(0.01),
        n_blocks=2,
        updates_per_block=1,
        batch_size=4,
        discount=0.99,
        polyak=0.005,
        policy_freq=1,
        scale_norm=False,
    )

    np.testing.assert_array_equal(fused_rng, baseline_rng)
    baseline_params = (
        tuple(actor.params for actor in baseline.actors),
        baseline.critic.params,
        baseline.target_actor.params,
        baseline.target_critic.params,
    )
    fused_params = (
        tuple(actor.params for actor in fused.actors),
        fused.critic.params,
        fused.target_actor.params,
        fused.target_critic.params,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(fused_params),
        jax.tree_util.tree_leaves(baseline_params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
