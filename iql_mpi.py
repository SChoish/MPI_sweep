"""Three actor/geometry paths sharing one actor-independent IQL critic.

Each path has a base actor and K persistent refinement actors. Each minibatch
updates the base, then the refinement actors in order with stopped references.
These are amortized neural proximal updates, not exact minimizers or a return
improvement guarantee. FR denotes the explicit local-KL surrogate below.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from iql_mpi_config import IQLConfig, VARIANTS
from train_td3bc import Actor, Transition, TwinCritic, sample_batch, target_update


class GaussianActor(nn.Module):
    action_dim: int
    max_action: float = 1.0
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations):
        x = observations
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        mean = self.max_action * jnp.tanh(nn.Dense(self.action_dim)(x))
        log_std = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros_init(),
                           bias_init=nn.initializers.constant(self.log_std_init),
                           name="log_std")(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, jnp.exp(log_std)


class Value(nn.Module):
    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, observations):
        x = observations
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(1)(x)


class IQLState(NamedTuple):
    actors: tuple[tuple[TrainState, ...], ...]
    critic: TrainState
    target_critic: TrainState
    value: TrainState


def coordinate_reduce(x, reduction="sum"):
    return jnp.sum(x, axis=-1) if reduction == "sum" else jnp.mean(x, axis=-1)


def gaussian_w2_squared(mean, std, ref_mean, ref_std, reduction="sum"):
    """Exact diagonal-Gaussian W2, before any optional action transform."""
    return coordinate_reduce((mean - ref_mean) ** 2 + (std - ref_std) ** 2, reduction)


def gaussian_kl(mean, std, ref_mean, ref_std, reduction="sum"):
    """KL(new || reference), not squared FR distance at finite displacement."""
    return coordinate_reduce(jnp.log(ref_std / std)
                             + (std**2 + (mean - ref_mean)**2) / (2 * ref_std**2)
                             - 0.5, reduction)


def gaussian_nll(mean, std, actions):
    return jnp.sum(jnp.log(std) + 0.5 * ((actions - mean) / std)**2
                   + 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def expectile_loss(residual, expectile):
    return jnp.mean(jnp.where(residual > 0, expectile, 1 - expectile) * residual**2)


def policy_stats(actor, params, observations, gaussian):
    output = actor.apply_fn(params, observations)
    return output if gaussian else (output, jnp.zeros_like(output))


def min_q(critic, observations, actions):
    q1, q2 = critic.apply_fn(critic.params, observations, actions)
    return jnp.minimum(q1, q2)[..., 0]


def normal_noise(key, samples, shape):
    """Antithetic standard normals; fixed within an inner proximal solve."""
    half = (samples + 1) // 2
    eps = jax.random.normal(key, (half, *shape))
    return jnp.concatenate((eps, -eps), axis=0)[:samples]


def sampled_q(critic, observations, mean, std, eps, transform="identity", max_action=1.0):
    actions = mean[None] + std[None] * eps
    if transform == "clip":
        actions = jnp.clip(actions, -max_action, max_action)
    states = jnp.broadcast_to(observations, (*actions.shape[:-1], observations.shape[-1]))
    qs = min_q(critic, states.reshape(-1, states.shape[-1]), actions.reshape(-1, actions.shape[-1]))
    return qs.reshape(actions.shape[:-1])


def create_state(key, observations, actions, config: IQLConfig, max_action=1.0):
    # Fixed folded keys keep Q/V and the common initial mean independent of
    # the number/order of variants and refinement depth.
    def make(model, tag, lr, *inputs):
        return TrainState.create(apply_fn=model.apply,
                                 params=model.init(jax.random.fold_in(key, tag), *inputs),
                                 tx=optax.adam(lr))
    critic = make(TwinCritic(config.hidden_dims), 1, config.critic_lr, observations, actions)
    value = make(Value(config.hidden_dims), 2, config.value_lr, observations)
    banks = []
    for variant in config.variants:
        if variant == "qbc_deterministic_w2":
            model = Actor(actions.shape[-1], max_action, config.hidden_dims)
        else:
            model = GaussianActor(actions.shape[-1], max_action, config.hidden_dims,
                                  config.log_std_init, config.log_std_min, config.log_std_max)
        base = make(model, 3, config.actor_lr, observations)
        # Same initial parameters, separate persistent optimizer states.
        banks.append(tuple(base.replace() for _ in range(config.mpi_steps + 1)))
    return IQLState(tuple(banks), critic, critic.replace(), value)


def update_value(state, batch, config):
    q = jax.lax.stop_gradient(min_q(state.target_critic, batch.observations, batch.actions))
    def loss_fn(params):
        v = state.value.apply_fn(params, batch.observations)[..., 0]
        return expectile_loss(q - v, config.expectile)
    loss, grad = jax.value_and_grad(loss_fn)(state.value.params)
    return state._replace(value=state.value.apply_gradients(grads=grad)), loss


def update_q(state, batch, config):
    next_v = state.value.apply_fn(state.value.params, batch.next_observations)
    target = jax.lax.stop_gradient(batch.rewards + config.discount * batch.not_dones * next_v)
    def loss_fn(params):
        q1, q2 = state.critic.apply_fn(params, batch.observations, batch.actions)
        return jnp.mean((q1 - target)**2 + (q2 - target)**2)
    loss, grad = jax.value_and_grad(loss_fn)(state.critic.params)
    critic = state.critic.apply_gradients(grads=grad)
    return state._replace(critic=critic, target_critic=target_update(critic, state.target_critic, config.polyak)), loss


def update_base(actor, critic, value, batch, variant, key, config, max_action):
    gaussian = variant != "qbc_deterministic_w2"
    eps = normal_noise(key, config.mc_samples if gaussian else 1, batch.actions.shape)
    q_data = min_q(critic, batch.observations, batch.actions)
    v = value.apply_fn(value.params, batch.observations)[..., 0]
    weights = jax.lax.stop_gradient(jnp.exp(jnp.minimum(config.awr_beta * (q_data - v), jnp.log(100.0))))
    scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_data)) + 1e-6) if config.iql_q_scale_norm else 1.0
    def loss_fn(params):
        mean, std = policy_stats(actor, params, batch.observations, gaussian)
        if variant == "awr_gaussian_fr":
            return jnp.mean(weights * gaussian_nll(mean, std, batch.actions))
        q = sampled_q(critic, batch.observations, mean, std, eps, config.q_action_transform, max_action)
        # E[||a-a_D||^2] under the raw Gaussian, exact rather than noisy MC BC.
        bc = jnp.mean(coordinate_reduce((mean - batch.actions)**2 + std**2, config.metric_reduction))
        return -jnp.mean(q) / scale + config.bc_coef * bc
    loss, grad = jax.value_and_grad(loss_fn)(actor.params)
    return actor.apply_gradients(grads=grad), loss


def refine_actor(actor, reference, critic, batch, variant, key, config, max_action=1.0):
    gaussian = variant != "qbc_deterministic_w2"
    ref_mean, ref_std = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
    eps = normal_noise(key, config.mc_samples if gaussian else 1, batch.actions.shape)
    q_ref = sampled_q(critic, batch.observations, ref_mean, ref_std, eps,
                      config.q_action_transform, max_action)
    scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_ref)) + 1e-6) if config.iql_q_scale_norm else 1.0
    h = config.tau / config.mpi_steps
    def loss_fn(params):
        mean, std = policy_stats(actor, params, batch.observations, gaussian)
        q = sampled_q(critic, batch.observations, mean, std, eps, config.q_action_transform, max_action)
        if variant == "awr_gaussian_fr":
            # d_FR^2 = 2 KL + higher-order terms, hence KL / h, not KL / (2h).
            distance = 2 * gaussian_kl(mean, std, ref_mean, ref_std, config.metric_reduction)
        else:
            distance = gaussian_w2_squared(mean, std, ref_mean, ref_std, config.metric_reduction)
        return -jnp.mean(q) / scale + jnp.mean(distance) / (2 * h)
    def body(_, current):
        _, grad = jax.value_and_grad(loss_fn)(current.params)
        return current.apply_gradients(grads=grad)
    actor = jax.lax.fori_loop(0, config.inner_updates, body, actor)
    mean, std = policy_stats(actor, actor.params, batch.observations, gaussian)
    q = sampled_q(critic, batch.observations, mean, std, eps, config.q_action_transform, max_action)
    raw = mean[None] + std[None] * eps
    metrics = {"loss": loss_fn(actor.params), "q": jnp.mean(q),
               "q_gain": jnp.mean(q - q_ref), "q_scale": jnp.asarray(scale),
               "mean_shift": jnp.mean(jnp.linalg.norm(mean - ref_mean, axis=-1)),
               "std_shift": jnp.mean(jnp.linalg.norm(std - ref_std, axis=-1)),
               "std_mean": jnp.mean(std),
               "out_of_bounds_fraction": jnp.mean(jnp.abs(raw) > max_action)}
    return actor, metrics


def update(state, batch: Transition, key, config: IQLConfig, max_action=1.0):
    # Match IQL's order: update V; extract policies from old target Q/new V;
    # update online Q toward r + gamma V(s'); finally update the target Q EMA.
    state, v_loss = update_value(state, batch, config)
    banks, metrics = [], {"value_loss": v_loss}
    for variant, bank in zip(config.variants, state.actors, strict=True):
        actor_key = jax.random.fold_in(key, VARIANTS.index(variant))
        base, base_loss = update_base(bank[0], state.target_critic, state.value, batch,
                                      variant, jax.random.fold_in(actor_key, 0), config, max_action)
        actors = [base]
        metrics[f"{variant}/base_loss"] = base_loss
        for hop in range(1, len(bank)):
            reference = policy_stats(actors[-1], actors[-1].params, batch.observations,
                                     variant != "qbc_deterministic_w2")
            actor, info = refine_actor(bank[hop], reference, state.target_critic, batch,
                                       variant, jax.random.fold_in(actor_key, hop), config, max_action)
            actors.append(actor)
            metrics.update({f"{variant}/hop{hop}/{k}": v for k, v in info.items()})
        banks.append(tuple(actors))
    state = state._replace(actors=tuple(banks))
    state, q_loss = update_q(state, batch, config)
    metrics["critic_loss"] = q_loss
    return state, metrics


def update_many(state, data, rng, config, batch_size, count):
    def body(carry, _):
        current, key = carry
        key, batch_key, actor_key = jax.random.split(key, 3)
        batch = sample_batch(data, batch_key, batch_size)
        current, metrics = update(current, batch, actor_key, config)
        return (current, key), metrics
    (state, rng), metrics = jax.lax.scan(body, (state, rng), None, length=count)
    return state, rng, jax.tree_util.tree_map(lambda x: x[-1], metrics)
