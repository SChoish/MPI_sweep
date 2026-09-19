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


def test_gaussian_fr_matches_density_overlap_integral():
    from scipy.integrate import quad
    from scipy.stats import norm
    m, s = np.array([[.4, -.3]]), np.array([[.7, 1.2]])
    rm, rs = np.array([[0., .8]]), np.array([[.2, .9]])
    bc = 1.
    for a, b, c, d in zip(m[0], s[0], rm[0], rs[0], strict=True):
        overlap, _ = quad(lambda x: np.sqrt(norm.pdf(x, a, b) * norm.pdf(x, c, d)), -np.inf, np.inf)
        bc *= overlap
    expected = 4 * np.arccos(bc)**2
    actual = core.gaussian_fr_squared(jnp.asarray(m), jnp.asarray(s), jnp.asarray(rm), jnp.asarray(rs))
    np.testing.assert_allclose(actual, [expected], rtol=2e-6)
    np.testing.assert_array_equal(actual, core.gaussian_fr_squared(jnp.asarray(rm), jnp.asarray(rs),
                                                                 jnp.asarray(m), jnp.asarray(s)))
    mean_cost = core.gaussian_fr_squared(jnp.asarray(m), jnp.asarray(s), jnp.asarray(rm), jnp.asarray(rs), "mean")
    np.testing.assert_allclose(mean_cost, actual / 2, rtol=1e-6)
    # Ambient FR of the product distribution is not a sum of marginal FR^2.
    marginal_sum = sum(float(core.gaussian_fr_squared(jnp.asarray(m[:, j:j+1]), jnp.asarray(s[:, j:j+1]),
                                                      jnp.asarray(rm[:, j:j+1]), jnp.asarray(rs[:, j:j+1]))[0])
                       for j in range(2))
    assert not np.isclose(float(actual[0]), marginal_sum)


def test_gaussian_fr_identity_gradient_and_fisher_factor():
    # Coordinates (mean, log sigma), Fisher tensor diag(1/sigma^2, 2).
    center = jnp.array([0.2, jnp.log(0.7)])
    h = .3
    def penalty(x):
        return jnp.sum(core.gaussian_fr_squared(x[:1], jnp.exp(x[1:]), center[:1], jnp.exp(center[1:]))) / (2*h)
    assert float(penalty(center)) == 0.
    np.testing.assert_array_equal(jax.jit(jax.grad(penalty))(center), jnp.zeros_like(center))
    np.testing.assert_allclose(jax.hessian(penalty)(center), np.diag([1/.7**2, 2])/h, rtol=2e-6, atol=1e-6)


def test_gaussian_fr_stays_finite_near_identity_and_zero_overlap():
    displacement = jnp.asarray([[0.], [1e-7], [1e-4], [.1], [10.], [1000.]])
    def distance(m):
        return core.gaussian_fr_squared(m, jnp.ones_like(m), jnp.zeros_like(m), jnp.ones_like(m))
    value = jax.jit(distance)(displacement)
    grad = jax.jit(jax.grad(lambda m: distance(m).sum()))(displacement)
    assert np.isfinite(value).all() and np.isfinite(grad).all()
    assert (np.asarray(value) >= 0).all() and (np.asarray(value) <= np.pi**2 + 1e-5).all()
    np.testing.assert_allclose(value[1:3], displacement[1:3, 0]**2, rtol=2e-6)
    np.testing.assert_allclose(grad[1:3], 2*displacement[1:3], rtol=2e-6)
    np.testing.assert_allclose(value[-1], np.pi**2, rtol=1e-6)


def test_gaussian_fr_derivative_matches_finite_differences():
    if hasattr(jax, "enable_x64"):
        enable_x64 = jax.enable_x64
    else:
        from jax.experimental import enable_x64
    with enable_x64():
        points = np.array([1e-8, 9.99e-5, 1.001e-4, .1, 1., 10.])
        def independent_value(d):
            # Inverse sine is accurate near zero; inverse cosine near saturation.
            return np.where(d < .5, 4 * np.arcsin(np.sqrt(-np.expm1(-2*d)))**2,
                            4 * np.arccos(np.exp(-d))**2)
        epsilon = np.minimum(points * .001, 1e-5)
        finite_diff = (independent_value(points+epsilon) - independent_value(points-epsilon)) / (2*epsilon)
        derivative = jax.vmap(jax.grad(core._fr_squared_from_bhattacharyya_distance))(jnp.asarray(points))
        np.testing.assert_allclose(derivative, finite_diff, rtol=2e-6, atol=1e-8)


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


def test_paper_ddpgbc_base_has_fixed_unit_std_and_matching_refinement_initialization():
    b = batch()
    cfg = IQLConfig(variants=(VARIANTS[2],), mpi_steps=2, hidden_dims=(8,))
    ts = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    actor = ts.actors[0][0]
    refined = ts.actors[0][1]
    reference = core.policy_stats(actor, actor.params, b.observations, True)
    assert_tree_equal(reference, core.policy_stats(refined, refined.params, b.observations, True))
    np.testing.assert_array_equal(reference[1], jnp.ones_like(b.actions))
    assert "log_std" not in actor.params["params"]
    assert "log_std" in refined.params["params"]
    for seed in range(3):
        actor, _ = core.update_base(actor, quadratic_critic(), ts.value, b, VARIANTS[2],
                                    jax.random.PRNGKey(seed), cfg, 1.)
    mean, std = core.policy_stats(actor, actor.params, b.observations, True)
    assert not np.array_equal(mean, reference[0])
    np.testing.assert_array_equal(std, jnp.ones_like(b.actions))
    # The paper mean is unsquashed, even though Q inputs/evaluation are clipped.
    params = jax.tree_util.tree_map(jnp.zeros_like, actor.params)
    params["params"]["Dense_1"]["bias"] = jnp.full((2,), 2.)
    mean, _ = actor.apply_fn(params, b.observations)
    np.testing.assert_array_equal(mean, jnp.full_like(b.actions, 2.))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_paper_ddpgbc_matches_closed_form_loss_and_sgd_gradient(reduction):
    # Two dimensions expose accidental coordinate averaging. The second mean
    # is out of bounds: Q must clip it while NLL must retain its raw value.
    obs = jnp.zeros((2, 1))
    actions = jnp.array([[-.6, .3], [.8, .7]])
    b = Transition(obs, actions, jnp.zeros((2, 1)), obs, jnp.ones((2, 1)))
    def apply(p, states):
        return jnp.broadcast_to(p["mean"], (len(states), 2)), jnp.ones((len(states), 2))
    actor = TrainState.create(apply_fn=apply, params={"mean": jnp.array([.2, 1.4])}, tx=optax.sgd(.1))
    def q_apply(p, states, a):
        q = jnp.sum(a - .5*a**2, axis=-1, keepdims=True)
        return q, q
    critic = SimpleNamespace(params={}, apply_fn=q_apply)
    value = SimpleNamespace(params={}, apply_fn=lambda p, s: jnp.zeros((len(s), 1)))
    cfg = IQLConfig(mpi_steps=1, tau=(1/3 if reduction == "sum" else 1/6),
                    metric_reduction=reduction)
    updated, loss = core.update_base(actor, critic, value, b, VARIANTS[2], jax.random.PRNGKey(0), cfg, 1.)
    # Q(clip([.2,1.4]))=.68; mean squared residual sum=1.35; unit-normal NLL=1.35/2+log(2pi).
    expected_loss = -.68 + 3*(1.35/2 + np.log(2*np.pi))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    # dL/dm = [-.8,0] + 3*([.2,1.4]-[.1,.5]) = [-.5,2.7].
    np.testing.assert_allclose(updated.params["mean"], [.25, 1.13], rtol=1e-6)
    # Base extraction is deterministic and independent of MPI sampling flags.
    alternate = replace(cfg, mc_samples=1, q_action_transform="clip")
    updated2, loss2 = core.update_base(actor, critic, value, b, VARIANTS[2], jax.random.PRNGKey(9), alternate, 1.)
    assert_tree_equal(updated, updated2)
    np.testing.assert_array_equal(loss, loss2)


def test_deterministic_td3bc_detaches_actor_q_scale_and_uses_mean_mse():
    obs = jnp.zeros((2, 1))
    actions = jnp.zeros((2, 2))
    b = Transition(obs, actions, jnp.zeros((2, 1)), obs, jnp.ones((2, 1)))
    actor = TrainState.create(apply_fn=lambda p, s: jnp.broadcast_to(p["mean"], (len(s), 2)),
                              params={"mean": jnp.array([.2, .4])}, tx=optax.sgd(.1))
    def apply_q(p, s, a):
        q = jnp.sum(a, axis=-1, keepdims=True)
        return q, q + 1
    critic = SimpleNamespace(params={}, apply_fn=apply_q)
    value = SimpleNamespace(params={}, apply_fn=lambda p, s: jnp.zeros((len(s), 1)))
    cfg = IQLConfig(mpi_steps=1, tau=1.25)
    updated, loss = core.update_base(actor, critic, value, b, VARIANTS[1], jax.random.PRNGKey(0), cfg, 1.)
    lam = 2.5 / (.6 + 1e-6)
    np.testing.assert_allclose(loss, -lam*.6 + .1, rtol=1e-6)
    np.testing.assert_allclose(updated.params["mean"], np.array([.2,.4]) - .1*(np.array([.2,.4])-lam), rtol=1e-6)


@pytest.mark.parametrize("variant", (VARIANTS[0], VARIANTS[2]))
def test_actual_gaussian_refinement_moves_std_and_stops_reference_gradient(variant):
    b = batch()
    cfg = IQLConfig(variants=(variant,), mpi_steps=2, tau=.5,
                    hidden_dims=(8,), mc_samples=8, inner_updates=2)
    ts = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    base, actor = ts.actors[0]
    mean, std = core.policy_stats(base, base.params, b.observations, True)
    def run(reference):
        return core.refine_actor(actor, reference, quadratic_critic(), b,
                                  variant, jax.random.PRNGKey(1), cfg)
    updated, info = jax.jit(run)((mean, std))
    _, new_std = core.policy_stats(updated, updated.params, b.observations, True)
    assert float(new_std.mean()) < float(std.mean())
    assert float(info["std_shift"]) > 0
    grads = jax.grad(lambda ref: run(ref)[1]["loss"])((mean, std))
    for grad in grads:
        np.testing.assert_array_equal(grad, jnp.zeros_like(grad))


@pytest.mark.parametrize("variant", (VARIANTS[0], VARIANTS[2]))
def test_flat_q_preserves_gaussian_reference(variant):
    b = batch()
    cfg = IQLConfig(variants=(variant,), mpi_steps=2, hidden_dims=(8,))
    ts = core.create_state(jax.random.PRNGKey(0), b.observations, b.actions, cfg)
    base, actor = ts.actors[0]
    reference = core.policy_stats(base, base.params, b.observations, True)
    critic = SimpleNamespace(params={}, apply_fn=lambda p, s, a: (jnp.zeros((*a.shape[:-1], 1)),)*2)
    new_actor, _ = core.refine_actor(actor, reference, critic, b, variant, jax.random.PRNGKey(1), cfg)
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
    single_cfg = replace(cfg, variants=(VARIANTS[1],), mpi_steps=1)
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
    assert_tree_equal(full.baselines[1], single.actors[0][0])
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
    monkeypatch.setattr(core, "update_base", lambda actor, *args, **kwargs: (bump(actor), jnp.array(0.)))
    def refine(actor, reference, *args, **kwargs):
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
                                   {"mpi_steps": 0}, {"log_std_max": -.5},
                                   {"variants": (VARIANTS[0], VARIANTS[0])}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        IQLConfig(**kwargs)


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("dimension", [1, 3, 6])
def test_total_time_coefficients_match_native_actor_losses(k, dimension):
    cfg = IQLConfig(mpi_steps=k, tau=3.)
    assert cfg.h * k == pytest.approx(3.)
    assert cfg.coefficients(VARIANTS[0], dimension)["awr_beta"] == 3 / k
    assert cfg.coefficients(VARIANTS[2], dimension)["bc_coef"] == k / 3
    assert cfg.coefficients(VARIANTS[1], dimension)["td3bc_alpha"] == 6 / k
    # Convert the same physical objectives to a common sum or mean geometry.
    summed = replace(cfg, metric_reduction="sum")
    averaged = replace(cfg, metric_reduction="mean")
    assert summed.coefficients(VARIANTS[1], dimension)["td3bc_alpha"] == pytest.approx(6 / (k * dimension))
    assert averaged.coefficients(VARIANTS[2], dimension)["bc_coef"] == pytest.approx(k / (3 * dimension))
    assert averaged.coefficients(VARIANTS[0], dimension)["awr_beta"] == pytest.approx(3 * dimension / k)


def test_awr_time_and_shared_q_scale_enter_advantage_weights():
    obs = jnp.zeros((2, 1))
    actions = jnp.array([[.2], [.4]])
    b = Transition(obs, actions, obs, obs, jnp.ones_like(obs))
    def apply(p, s):
        return jnp.broadcast_to(p["mean"], (len(s), 1)), jnp.ones((len(s), 1))
    actor = TrainState.create(apply_fn=apply, params={"mean": jnp.array([.1])}, tx=optax.sgd(.1))
    critic = SimpleNamespace(params={}, apply_fn=lambda p, s, a: (a, a))
    value = SimpleNamespace(params={}, apply_fn=lambda p, s: jnp.zeros((len(s), 1)))
    cfg = IQLConfig(tau=3., mpi_steps=2, iql_q_scale_norm=True)
    new, loss = core.update_base(actor, critic, value, b, VARIANTS[0], jax.random.PRNGKey(0),
                                 cfg, 1., q_scale=2.)
    w = np.exp(1.5 * np.array([.2, .4]) / 2)
    expected = np.mean(w * (.5 * (np.array([.2, .4])-.1)**2 + .5*np.log(2*np.pi)))
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
    np.testing.assert_allclose(new.params["mean"], [.1 - .1*np.mean(w*(.1-np.array([.2,.4])))], rtol=1e-6)


def test_k1_is_base_and_full_time_controls_and_q_scale_are_independent_of_k():
    b = batch()
    cfg = IQLConfig(tau=1.25, mpi_steps=1, hidden_dims=(8,), mc_samples=2)
    refined_cfg = replace(cfg, mpi_steps=4)
    key = jax.random.PRNGKey(4)
    initial = core.create_state(key, b.observations, b.actions, cfg)
    deep = core.create_state(key, b.observations, b.actions, refined_cfg)
    assert all(len(bank) == 1 for bank in initial.actors)
    assert all(len(bank) == 4 for bank in deep.actors)
    # Direct native base update must be exactly the only K=1 actor update.
    after_v, _ = core.update_value(initial, b, cfg)
    shallow = initial
    f1 = jax.jit(partial(core.update, config=cfg))
    f4 = jax.jit(partial(core.update, config=refined_cfg))
    shallow, _ = f1(shallow, b, key)
    for i, variant in enumerate(cfg.variants):
        expected, _ = core.update_base(initial.actors[i][0], initial.target_critic,
                                       after_v.value, b, variant, key, cfg, 1.)
        # Separate JIT/eager arithmetic can round differently; compare numerical result.
        for a, e in zip(jax.tree_util.tree_leaves(shallow.actors[i][0].params),
                        jax.tree_util.tree_leaves(expected.params), strict=True):
            np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-7)
        assert int(shallow.actors[i][0].step) == 1
        assert_tree_equal(shallow.actors[i][0], shallow.baselines[i])
    shallow = initial
    for _ in range(3):
        shallow, info1 = f1(shallow, b, key)
        deep, info4 = f4(deep, b, key)
        assert_tree_equal(shallow.baselines, deep.baselines)
        assert_tree_equal(shallow.critic, deep.critic)
        for variant in cfg.variants:
            c = info1[f"{variant}/hop1/q_scale"]
            for hop in range(1, 5):
                np.testing.assert_array_equal(c, info4[f"{variant}/hop{hop}/q_scale"])
    assert all(int(a.step) == 3 for bank in deep.actors for a in bank)


def test_normalized_refinement_requires_shared_scale_and_uses_it_in_gradient():
    b = batch()
    cfg = IQLConfig(variants=(VARIANTS[1],), mpi_steps=2, tau=.4, actor_lr=.1)
    actor = TrainState.create(apply_fn=lambda p, s: jnp.broadcast_to(p["m"], (len(s), 2)),
                              params={"m": jnp.array([.2, .4])}, tx=optax.sgd(.1))
    critic = SimpleNamespace(params={}, apply_fn=lambda p, s, a: (jnp.sum(a, axis=-1, keepdims=True),)*2)
    ref = (jnp.zeros_like(b.actions), jnp.zeros_like(b.actions))
    with pytest.raises(ValueError, match="common"):
        core.refine_actor(actor, ref, critic, b, VARIANTS[1], jax.random.PRNGKey(0), cfg)
    new, info = core.refine_actor(actor, ref, critic, b, VARIANTS[1], jax.random.PRNGKey(0), cfg, q_scale=2.)
    # L=-sum(m)/2 + mean_j(m_j^2)/(2*.2); gradient=-.5 + m/.4.
    np.testing.assert_allclose(new.params["m"], [.2, .35], rtol=1e-6)
    assert float(info["q_scale"]) == 2.
