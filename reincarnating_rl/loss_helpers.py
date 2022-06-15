# coding=utf-8
# Copyright 2022 The Reincarnating RL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for loss computation for various agents."""

import enum
import functools

from absl import logging
from dopamine.jax import losses
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax


class DistillType(enum.IntEnum):
  POLICY_ONLY = 1
  POLICY_AND_VALUE = 2
  VALUE_ONLY = 3


@gin.configurable
@functools.partial(jax.jit, static_argnums=(0, 2, 3))
def persistence_linearly_decaying_epsilon(
    decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy."""
  unused_argv = warmup_steps  # Do not use warmup steps for persistence.
  steps_left = decay_period - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = jnp.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


@gin.configurable
def create_linear_schedule(initial_lr=1e-4, final_lr=1e-5):
  """Decaying learning rate using a multiplier."""
  def learning_rate_decay_fn(multiplier):
    return multiplier * initial_lr + (1-multiplier) * final_lr
  return learning_rate_decay_fn


@gin.configurable
def create_persistence_optimizer(name='adam',
                                 learning_rate=6.25e-5,
                                 beta1=0.9,
                                 beta2=0.999,
                                 eps=1.5e-4,
                                 centered=False,
                                 inject_hparams=False):
  """Create an optimizer for training.

  Currently, only the Adam and RMSProp optimizers are supported.

  Args:
    name: str, name of the optimizer to create.
    learning_rate: float, learning rate to use in the optimizer.
    beta1: float, beta1 parameter for the optimizer.
    beta2: float, beta2 parameter for the optimizer.
    eps: float, epsilon parameter for the optimizer.
    centered: bool, centered parameter for RMSProp.
    inject_hparams: bool, whether to use `optax.inject_hyperparams`.

  Returns:
    An optax optimizer .
  """
  # https://github.com/deepmind/optax/discussions/262 for
  # `optax.inject_hyperparams`
  wrapper_fn = lambda f: optax.inject_hyperparams(f) if inject_hparams else f
  if name == 'adam':
    logging.info('Creating Adam optimizer with settings lr=%f, beta1=%f, '
                 'beta2=%f, eps=%f', learning_rate, beta1, beta2, eps)
    return wrapper_fn(optax.adam)(
        learning_rate, b1=beta1, b2=beta2, eps=eps)
  elif name == 'rmsprop':
    logging.info('Creating RMSProp optimizer with settings lr=%f, beta2=%f, '
                 'eps=%f', learning_rate, beta2, eps)
    return wrapper_fn(optax.rmsprop)(
        learning_rate, decay=beta2, eps=eps, centered=centered)
  else:
    raise ValueError('Unsupported optimizer {}'.format(name))


def expand_dims(arr, n, axis=-1):
  """Expand dims multiple times as specified by `n`."""
  for _ in range(n):
    arr = jnp.expand_dims(arr, axis=axis)
  return arr


@functools.partial(jax.vmap, in_axes=(None, 0))
def get_q_values(model, states):
  return model(states).q_values


def compute_dr3_loss(state_representations, next_state_representations):
  """Minimizes dot product between state and next state representations."""
  dot_products = jnp.einsum(
      'ij,ij->i', state_representations, next_state_representations)
  # Minimize |\phi(s) \phi(s')|
  return jnp.mean(jnp.abs(dot_products))


def kl_divergence_with_logits(target_logits, prediction_logits):
  """Implementation of on-policy distillation loss."""
  out = -nn.softmax(target_logits) * (nn.log_softmax(prediction_logits)
                                      - nn.log_softmax(target_logits))
  return jnp.sum(out)


def stable_scaled_log_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  tau_lse = max_x + tau * jnp.log(
      jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True))
  return x - tau_lse


def stable_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  return jax.nn.softmax(y / tau, axis=axis)


def get_munchausen_reward(q_target, states, actions, tau, clip_value_min=-1):
  r"""Use clipped value of [tau x log\pi(a|s)] as additional reward."""
  target_q_values = get_q_values(q_target, states)
  replay_log_policy = stable_scaled_log_softmax(
      target_q_values, tau, axis=1)
  # replay_next_policy = stable_softmax(  # pi_k+1(s')
  #     target_next_q_values, tau, axis=1)
  # replay_next_qt_softmax = jnp.sum(
  #     (target_next_q_values - replay_next_log_policy) * replay_next_policy,
  #     axis=1)
  tau_log_pi_a = jax.vmap(lambda x, y: x[y])(replay_log_policy, actions)
  # Clip the log-policy.
  return jnp.clip(tau_log_pi_a, a_min=clip_value_min, a_max=1)


def batch_cql_loss(q_values, actions, distill_temperature=1.0):
  q_values /= distill_temperature
  replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
  return jax.scipy.special.logsumexp(q_values, axis=-1) - replay_chosen_q


def create_cql_loss(network_def,
                    states,
                    actions,
                    next_states,
                    cql_coefficient,
                    dr3_coefficient=0.0,
                    distill_temperature=1.0,
                    use_vision_transformer=False):
  """Loss function for training using offline data."""

  def loss_fn(params):
    def q_online(state):
      if use_vision_transformer:
        return network_def.apply(params, state, train=True)
      return network_def.apply(params, state)
    model_output = jax.vmap(q_online)(states)
    # Compute CQL Loss.
    q_values = jnp.squeeze(model_output.q_values)
    cql_loss = jnp.mean(batch_cql_loss(q_values, actions, distill_temperature))

    if dr3_coefficient is not None:
      # Compute DR3 loss.
      representations = jnp.squeeze(model_output.representation)
      next_states_model_output = jax.vmap(q_online)(next_states)
      next_state_representations = jnp.squeeze(
          next_states_model_output.representation)
      dr3_loss = compute_dr3_loss(representations, next_state_representations)
      total_loss = dr3_coefficient * dr3_loss + cql_coefficient * cql_loss
      return total_loss, (cql_loss, dr3_loss)
    else:
      return cql_loss
  return loss_fn


def margin_loss(q_values,
                actions,
                margin,
                dqfd_margin_loss=True):
  """Helper for creating distillation loss."""
  # Compute margin loss.
  replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
  l_margin = margin * (1 - jax.nn.one_hot(actions, q_values.shape[1]))

  if dqfd_margin_loss:
    margin_values = q_values + l_margin
    max_actions = jnp.argmax(margin_values, axis=-1)
    max_margin_vals = jax.vmap(lambda x, y: x[y])(margin_values, max_actions)
    per_state_margin = max_margin_vals - replay_chosen_q
  else:
    # Elementwise max followed by averaging over all (state, action) tuples.
    per_state_margin = jnp.maximum(
        q_values + l_margin - jnp.expand_dims(replay_chosen_q, -1), 0)

  return jnp.mean(per_state_margin)


def distillation_loss(q_values,
                      temperature,
                      target,
                      distill_best_action_only=False,
                      distill_type=DistillType.POLICY_ONLY,
                      return_per_example_loss=False):
  """Helper for creating distillation loss."""
  value_loss = jnp.mean(
      jax.vmap(losses.mse_loss)(target, q_values), axis=-1)

  target_q_values = target / temperature
  student_q_values = q_values / temperature
  if distill_best_action_only:
    actions = jnp.argmax(target_q_values, axis=-1)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(student_q_values, actions)
    policy_loss = (
        jax.scipy.special.logsumexp(student_q_values, axis=-1) -
        replay_chosen_q)
  else:
    policy_loss = jax.vmap(kl_divergence_with_logits)(
        target_q_values, student_q_values)
  if distill_type == DistillType.POLICY_AND_VALUE:
    total_loss = policy_loss + value_loss
  elif distill_type == DistillType.POLICY_ONLY:
    total_loss = policy_loss
  else:
    total_loss = value_loss
  if return_per_example_loss:
    return total_loss
  return jnp.mean(total_loss)


def create_distillation_loss(network_def,
                             states,
                             temperature=1.0,
                             distill_best_action_only=False,
                             distill_type=DistillType.POLICY_ONLY):
  """Loss function for running distillation step."""
  def distillation_loss_fn(params, target):
    """Computes the distillation loss."""

    def q_online(state):
      return network_def.apply(params, state)
    q_values = jnp.squeeze(get_q_values(q_online, states))
    return distillation_loss(
        q_values, temperature, target, distill_best_action_only, distill_type)

  return distillation_loss_fn


def get_q_value_statistics(q_values, replay_chosen_q):
  sorted_values = jnp.argsort(q_values, axis=1)[-2:]
  max_q = jnp.mean(sorted_values[:, 1])
  action_gap = jnp.mean(sorted_values[:, 1] - sorted_values[:, 0])
  return (jnp.mean(replay_chosen_q), action_gap, max_q)


def q_learning_loss(q_values, target, actions, loss_type='huber',
                    return_mean_loss=True):
  """Q-learning loss."""
  replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
  loss_fn = losses.huber_loss if loss_type == 'huber' else losses.mse_loss
  loss = jax.vmap(loss_fn)(target, replay_chosen_q)
  if return_mean_loss:
    loss = jnp.mean(loss)
  q_value_statistics = get_q_value_statistics(q_values, replay_chosen_q)
  return loss, q_value_statistics


def q_learning_loss_fn(network_def, states, actions, loss_type='huber',
                       use_vision_transformer=False):
  """Loss function for running Q-learning step."""

  def loss_fn(params, target):
    def q_online(state):
      if use_vision_transformer:
        return network_def.apply(params, state, train=True)
      return network_def.apply(params, state)
    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    return q_learning_loss(q_values, target, actions, loss_type)
  return loss_fn


@functools.partial(jax.jit, static_argnames=('network_def'))
def q_stats(network_def, online_params, states, actions):
  def q_online(state):
    return network_def.apply(online_params, state)
  q_values = jnp.squeeze(jax.vmap(q_online)(states).q_values)
  data_actions_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
  return get_q_value_statistics(q_values, data_actions_q)
