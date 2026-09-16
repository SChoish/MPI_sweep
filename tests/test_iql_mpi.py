"""Behavioral gates for IQL independence, geometry, variance, and resumability."""
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

import iql_mpi as core
import train_iql_mpi as trainer
from iql_mpi_config import IQLConfig, VARIANTS, config_from_args, run_name
from train_td3bc import Transition


def batch():
    obs = jnp.arange(24, dtype=jnp.float32).reshape(8, 3) / 30
    actions = jnp.stack((obs[:, 0], -obs[:, 1]), axis=-1)
    return Transition(obs, actions, jnp.ones((8, 1)), obs * 0.9,
                      jnp.asarray([[1.], [0.]] * 4))


def assert_tree_equal(a, b):
    leaves_a, leaves_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_w2_and_dirac_limit():
    m, s = jnp.array([[1., 2.]]), jnp.array([[2., 3.]])
    rm, rs = jnp.array([[-1., 1.]]), jnp.array([[1., 1.]])
    np.testing.assert_allclose(core.gaussian_w2_squared(m, s, rm, rs), [10.])
    np.testing.assert_allclose(core.gaussian_w2_squared(m, s, rm, rs, "mean"), [5.])
    np.testing.assert_allclose(core.gaussian_w2_squared(m, s*0, rm, rs*0), [5.])


def test_local_kl_direction_and_fisher_factor():
    m, s = jnp.array([[0.4]]), jnp.array([[0.7]])
    rm, rs = jnp.array([[0.]]), jnp.array([[0.2]])
    expected = np.log(.2/.7) + (.7**2 + .4**2)/(2*.2**2) - .5
    np.testing.assert_allclose(core.gaussian_kl(m, s, rm, rs), [expected], rtol=1e-6)
    assert not np.isclose(float(core.gaussian_kl(m, s, rm, rs)[0]),
                          float(core.gaussian_kl(rm, rs, m, s)[0]))
    # Coordinates (mean, log sigma), Fisher tensor diag(1/sigma^2, 2).
    center = jnp.array([0.2, jnp.log(0.7)])
    h = .3
    def penalty(x):
        return jnp.sum(core.gaussian_kl(x[:1], jnp.exp(x[1:]), center[:1], jnp.exp(center[1:]))) / h
    np.testing.assert_allclose(jax.hessian(penalty)(center), np.diag([1/.7**2, 2])/h, rtol=2e-6, atol=1e-6)


def quadratic_critic():
    def apply(_, states, actions):
        q = -0.5 * jnp.sum(actions**2, axis=-1, keepdims=True)
        return q, q
    return SimpleNamespace(params={}, apply_fn=apply)


def test_expected_q_has_variance_gradient_and_matches_quadratic_integral():
    critic = quadratic_critic()
    obs, mean = jnp.zeros((1, 1)), jnp.array([[.3]])
    std, eps = jnp.array([[.4]]), jnp.array([[[-1.]], [[1.]]])
    def q_of_std(s):
        return core.sampled_q(critic, obs, mean, s, eps).mean()
    np.testing.assert_allclose(q_of_std(std), -.5 * (.3**2 + .4**2), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(q_of_std)(std), -std, rtol=1e-6)
    # mean-only Q has exactly zero variance gradient.
    np.testing.assert_array_equal(jax.grad(lambda s: core.min_q(critic, obs, mean).mean())(std), std*0)


def test_qbc_gaussian_bc_is_expected_square():
    # With a zero critic and lambda=1, Gaussian BC also reduces sigma.
    b = batch()
    cfg = IQLConfig(variants=(VARIANTS[2],), mpi_steps=1, hidden_dims=(8,))
    ts = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    critic = SimpleNamespace(params={}, apply_fn=lambda p, s, a: (jnp.zeros((*a.shape[:-1], 1)),)*2)
    actor = ts.actors[0][0]
    updated, _ = core.update_base(actor, critic, ts.value, b, VARIANTS[2], jax.random.PRNGKey(2), cfg, 1.)
    _, before = core.policy_stats(actor, actor.params, b.observations, True)
    _, after = core.policy_stats(updated, updated.params, b.observations, True)
    assert float(after.mean()) < float(before.mean())


def test_actual_gaussian_refinement_moves_std_and_stops_reference_gradient():
    b = batch()
    cfg = IQLConfig(variants=(VARIANTS[2],), mpi_steps=1, tau=.5,
                    hidden_dims=(8,), mc_samples=8, inner_updates=2)
    ts = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    actor = ts.actors[0][0]
    mean, std = core.policy_stats(actor, actor.params, b.observations, True)
    def run(reference):
        return core.refine_actor(actor, reference, quadratic_critic(), b,
                                  VARIANTS[2], jax.random.PRNGKey(1), cfg)
    updated, info = jax.jit(run)((mean, std))
    _, new_std = core.policy_stats(updated, updated.params, b.observations, True)
    assert float(new_std.mean()) < float(std.mean())
    assert float(info["std_shift"]) > 0
    grads = jax.grad(lambda ref: run(ref)[1]["loss"])((mean, std))
    for grad in grads:
        np.testing.assert_array_equal(grad, jnp.zeros_like(grad))


def test_flat_q_preserves_gaussian_reference():
    b = batch()
    cfg = IQLConfig(variants=(VARIANTS[2],), mpi_steps=1, hidden_dims=(8,))
    ts = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    actor = ts.actors[0][0]
    reference = core.policy_stats(actor, actor.params, b.observations, True)
    critic = SimpleNamespace(params={}, apply_fn=lambda p, s, a: (jnp.zeros((*a.shape[:-1], 1)),)*2)
    new_actor, _ = core.refine_actor(actor, reference, critic, b, VARIANTS[2], jax.random.PRNGKey(1), cfg)
    assert_tree_equal(new_actor.params, actor.params)


def test_iql_value_and_q_use_dataset_and_value_targets():
    # Closed scalar SGD updates distinguish expectile weighting, twin Q target,
    # terminal masking, and whether actor actions leak into Bellman backups.
    def value_apply(p, obs):
        return jnp.ones((len(obs), 1)) * p["value"]
    def q_apply(p, obs, actions):
        return jnp.ones((len(obs), 1))*p["q"][0], jnp.ones((len(obs), 1))*p["q"][1]
    v = TrainState.create(apply_fn=value_apply, params={"value": jnp.array(0.)}, tx=optax.sgd(.1))
    q = TrainState.create(apply_fn=q_apply, params={"q": jnp.array([1., 2.])}, tx=optax.sgd(.1))
    state = core.IQLState((), q, q, v)
    b = batch()
    cfg = IQLConfig(expectile=.7, discount=.9, polyak=.5)
    state, loss = core.update_value(state, b, cfg)
    np.testing.assert_allclose(loss, .7, rtol=1e-6)
    np.testing.assert_allclose(state.value.params["value"], .14, rtol=1e-6)
    state, _ = core.update_q(state, b, cfg)
    # Half nonterminal: mean target = 1 + .5*.9*.14 = 1.063.
    np.testing.assert_allclose(state.critic.params["q"], [1.0126, 1.8126], rtol=1e-6)
    np.testing.assert_allclose(state.target_critic.params["q"], [1.0063, 1.9063], rtol=1e-6)


def test_shared_critic_and_rng_independent_of_variants_depth_and_dispatch():
    b = batch()
    cfg = IQLConfig(mpi_steps=2, hidden_dims=(8,), mc_samples=2)
    single_cfg = replace(cfg, variants=(VARIANTS[1],), mpi_steps=0)
    key, rng = jax.random.split(jax.random.PRNGKey(10))
    full = core.create_state(key, b.observations, b.actions, cfg)
    single = core.create_state(key, b.observations, b.actions, single_cfg)
    full_fn = jax.jit(partial(core.update_many, config=cfg, batch_size=4, count=2))
    single_fn = jax.jit(partial(core.update_many, config=single_cfg, batch_size=4, count=2))
    full, rk, info = full_fn(full, b, rng)
    single, sk, _ = single_fn(single, b, rng)
    assert all(np.isfinite(float(v)) for v in info.values())
    for field in ("critic", "target_critic", "value"):
        assert_tree_equal(getattr(full, field), getattr(single, field))
    assert_tree_equal(full.actors[1][0], single.actors[0][0])
    assert_tree_equal(rk, sk)
    initial = core.create_state(key, b.observations, b.actions, cfg)
    one_fn = jax.jit(partial(core.update_many, config=cfg, batch_size=4, count=1))
    initial, rng, _ = one_fn(initial, b, rng)
    initial, rng, _ = one_fn(initial, b, rng)
    assert_tree_equal(initial, full)
    assert_tree_equal(rng, rk)
    assert all(int(a.step) == 2 for bank in full.actors for a in bank)


def test_hops_recenter_on_latest_predecessor(monkeypatch):
    b = batch()
    cfg = IQLConfig(variants=(VARIANTS[1],), mpi_steps=2, hidden_dims=(8,))
    state = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    refs = []
    def bump(actor):
        params = jax.tree_util.tree_map(lambda x: x + .01, actor.params)
        return actor.replace(params=params)
    monkeypatch.setattr(core, "update_base", lambda actor, *args: (bump(actor), jnp.array(0.)))
    def refine(actor, reference, *args):
        refs.append(reference[0])
        return bump(actor), {}
    monkeypatch.setattr(core, "refine_actor", refine)
    new, _ = core.update(state, b, jax.random.PRNGKey(1), cfg)
    for hop, ref in enumerate(refs):
        expected = new.actors[0][hop].apply_fn(new.actors[0][hop].params, b.observations)
        np.testing.assert_array_equal(ref, expected)
    assert not np.array_equal(refs[0], state.actors[0][0].apply_fn(state.actors[0][0].params, b.observations))


def test_checkpoint_preserves_next_update_and_rejects_changed_geometry(tmp_path):
    b = batch()
    cfg = IQLConfig(mpi_steps=1, hidden_dims=(8,), mc_samples=2)
    state = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    fn = jax.jit(partial(core.update_many, config=cfg, batch_size=4, count=1))
    state, rng, _ = fn(state, b, jax.random.PRNGKey(2))
    path = tmp_path / "params_1.pkl"
    signature = {"geometry": "test"}
    trainer.save_checkpoint(path, state, rng, 1, np.zeros(3), np.ones(3), signature, {}, "data")
    restored, rng2, step, _, _ = trainer.restore_checkpoint(path, state, signature, {}, "data")
    assert step == 1
    out1, rng1, _ = fn(state, b, rng)
    out2, rng2, _ = fn(restored, b, rng2)
    assert_tree_equal(out1, out2)
    assert_tree_equal(rng1, rng2)
    with pytest.raises(ValueError, match="configuration mismatch"):
        trainer.restore_checkpoint(path, state, {"geometry": "changed"}, {}, "data")


@pytest.mark.parametrize("kwargs", [{"tau": float("nan")}, {"mpi_steps": -1},
                                   {"mc_samples": 0}, {"expectile": 1.},
                                   {"variants": (VARIANTS[0], VARIANTS[0])}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        IQLConfig(**kwargs)
