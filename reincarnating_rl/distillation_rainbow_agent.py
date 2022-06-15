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
"""Rainbow agent that use persistence via Dagger on offline and online data."""

import enum
import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from reincarnating_rl import loss_helpers
from reincarnating_rl import persistent_rainbow_agent


class MethodType(enum.IntEnum):
  REINCARNATION = 1
  QL_PLUS_DAGGER = 2
  DAGGER = 3


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_logits_and_q_values(model, states, rng):
  outputs = model(states, key=rng)
  return (outputs.logits, outputs.q_values)


def _loss_fn_train(loss_fn, loss_weights, distill_target, optimizer,
                   optimizer_state, online_params, target):
  """Helper for creating training loss."""
  # Use the weighted mean loss for gradient computation.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  # Get the unweighted loss without taking its mean for updating priorities.
  # outputs[1] correspond to the per-example TD loss.
  (overall_loss, outputs), grad = grad_fn(online_params, target, distill_target,
                                          loss_weights)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, overall_loss, outputs


@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'optimizer',
                     'double_dqn', 'distributional', 'distill_temperature',
                     'distill_best_action_only', 'distill_loss_coefficient',
                     'td_coefficient'))
def train_and_distill(
    network_def, online_params, target_params, optimizer, optimizer_state,
    teacher_q_values, states, actions, next_states, rewards, terminals,
    loss_weights, support, cumulative_gamma, double_dqn, distributional, rng,
    distill_loss_multiplier, distill_temperature=1.0,
    distill_best_action_only=False, distill_loss_coefficient=0.0,
    td_coefficient=1.0):
  """Run a training step."""

  # Split the current rng into 2 for updating the rng after this call
  rng, rng1, rng2 = jax.random.split(rng, num=3)

  def q_online(state, key):
    return network_def.apply(online_params, state, key=key, support=support)

  def q_target(state, key):
    return network_def.apply(target_params, state, key=key, support=support)

  def loss_fn(params, target, distill_target, loss_multipliers):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_func(state, key):
      return network_def.apply(params, state, key=key, support=support)

    if distributional:
      logits, q_values = get_logits_and_q_values(q_func, states, rng)
      logits, q_values = jnp.squeeze(logits), jnp.squeeze(q_values)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
      loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits)
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
      q_value_statistics = loss_helpers.get_q_value_statistics(
          q_values, replay_chosen_q)
    else:
      q_values = jnp.squeeze(
          full_rainbow_agent.get_q_values(q_func, states, rng))
      loss, q_value_statistics = loss_helpers.q_learning_loss(
          q_values, target, actions, loss_type='huber', return_mean_loss=False)

    # Compute distillation loss
    def distillation_loss_fn():
      distill_loss = loss_helpers.distillation_loss(
          q_values,
          distill_temperature,
          distill_target,
          distill_best_action_only=distill_best_action_only,
          return_per_example_loss=True)
      return jnp.mean(loss_multipliers * distill_loss)

    distill_coeff = distill_loss_multiplier * distill_loss_coefficient
    zero_loss_fn = lambda: jnp.array(0.0)
    mean_distill_loss = jax.lax.cond(
        distill_coeff > 0, distillation_loss_fn, zero_loss_fn)

    mean_td_loss = jnp.mean(loss_multipliers * loss)
    mean_loss = td_coefficient * mean_td_loss + (
        distill_coeff * mean_distill_loss)
    return mean_loss, (loss, mean_td_loss, mean_distill_loss,
                       q_value_statistics)

  target = full_rainbow_agent.target_output(q_online, q_target, next_states,
                                            rewards, terminals, support,
                                            cumulative_gamma, double_dqn,
                                            distributional, rng1)

  optimizer_state, online_params, loss, outputs = _loss_fn_train(
      loss_fn, loss_weights, teacher_q_values, optimizer, optimizer_state,
      online_params, target)
  return optimizer_state, online_params, loss, outputs, rng2


@gin.configurable
class DistillationRainbowAgent(persistent_rainbow_agent.PersistentRainbowAgent):
  """Uses offline pretraining to kickstart learning."""

  def __init__(
      self,
      num_actions,
      distill_loss_coefficient=1.0,
      distill_temperature=1.0,
      distill_decay_period=-1,  # -1 corresponds to no decay.
      distill_best_action_only=False,
      summary_writer=None,
      method_type=MethodType.REINCARNATION,
      seed=None):
    super().__init__(
        num_actions,
        seed=seed,
        summary_writer=summary_writer)

    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t distill_decay_period: %d', distill_decay_period)
    logging.info('\t distill_loss_coefficient: %.4f', distill_loss_coefficient)
    logging.info('\t distill_best_action_only: %.4f', distill_best_action_only)
    logging.info('\t distill_temperature: %.4f', distill_temperature)
    logging.info('\t method_type: %d', method_type)
    # No. of steps within which to decay distillation and dr3 loss coefficients.
    self.distill_decay_period = distill_decay_period
    self.distill_loss_coefficient = distill_loss_coefficient
    self.distill_best_action_only = distill_best_action_only
    self.distill_temperature = distill_temperature
    self.online_training_steps = 0
    self._method_type = method_type
    if method_type == MethodType.DAGGER:
      self._td_coefficient = 0.0
    else:
      self._td_coefficient = 1.0

  def set_phase(self, persistence=False):
    self._persistent_phase = persistence
    if not persistence:
      self._sync_weights()  # Sync online and target network before training.

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.online_params = self.network_def.init(rng, x=self.state,
                                               support=self._support)
    self.optimizer = loss_helpers.create_persistence_optimizer(
        self._optimizer_name, inject_hparams=True)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params
    # Learning rate decay.
    self.loss_decay = 1.0
    self.learning_rate_fn = loss_helpers.create_linear_schedule()
    self.optimizer_state.hyperparams['learning_rate'] = self.learning_rate_fn(
        self.loss_decay)

  def record_score(self, normalized_score):
    # Decay the coefficients of distillation loss.
    super().record_score(normalized_score)
    if self._method_type != MethodType.REINCARNATION:
      return
    if (not self._persistent_phase) and self.loss_decay > 0:
      self.loss_decay = max(1.0 - normalized_score, 0.0)
      if (self.distill_decay_period > 0 and
          self.online_training_steps > self.distill_decay_period):
        self.loss_decay = 0
      # Decay the learning rate based ondistillation loss decay.
      self.optimizer_state.hyperparams['learning_rate'] = self.learning_rate_fn(
          self.loss_decay)

  def _persistence_step(self):
    self._sample_from_teacher_replay_buffer()
    self.replay_elements = self.teacher_replay_elements
    self._distillation_step(self.pretraining_cumulative_gamma)
    if self.training_steps % self.persistence_target_update_period == 0:
      self._sync_weights()

  def _original_train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for _ in range(self._num_updates_per_train_step):
          self._training_step_update()

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
    self.online_training_steps += 1

  def _training_step_update(self):
    self._sample_from_replay_buffer()
    self._distillation_step(self.cumulative_gamma)

  def _distillation_step(self, cumulative_gamma):
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    raw_states = self.replay_elements['state']
    states = self.train_preprocess_fn(raw_states, rng=rng1)
    next_states = self.train_preprocess_fn(
        self.replay_elements['next_state'], rng=rng2)
    # Teacher agent uses its own pre-processing.
    teacher_q_values = jax.lax.stop_gradient(
        self.teacher_agent.get_q_values(raw_states))

    # Whether to use prioritized replay or not.
    use_prioritized_replay = ((not self._persistent_phase) and
                              self._replay_scheme == 'prioritized')
    if use_prioritized_replay:
      probs = self.replay_elements['sampling_probabilities']
      loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
      loss_weights /= jnp.max(loss_weights)
    else:
      # Uniform weights if not using prioritized replay.
      loss_weights = jnp.ones(states.shape[0])

    (self.optimizer_state, self.online_params, overall_loss,
     individual_outputs, self._rng) = train_and_distill(
         self.network_def, self.online_params, self.target_network_params,
         self.optimizer, self.optimizer_state, teacher_q_values, states,
         self.replay_elements['action'], next_states,
         self.replay_elements['reward'],
         self.replay_elements['terminal'], loss_weights,
         self._support, cumulative_gamma, self._double_dqn,
         self._distributional, self._rng,
         distill_temperature=self.distill_temperature,
         distill_loss_coefficient=self.distill_loss_coefficient,
         distill_loss_multiplier=self.loss_decay,
         distill_best_action_only=self.distill_best_action_only,
         td_coefficient=self._td_coefficient)

    (loss, mean_td_loss, mean_distill_loss, (avg_q, action_gap,
                                             max_q)) = individual_outputs
    if use_prioritized_replay:
      self._replay.set_priority(self.replay_elements['indices'],
                                jnp.sqrt(loss + 1e-10))
    teacher_q = jnp.mean(jnp.max(teacher_q_values, axis=-1))

    if (self.summary_writer is not None and
        self.training_steps % self.summary_writing_frequency == 0):
      with self.summary_writer.as_default():
        tf.summary.scalar('Train/OverallLoss', overall_loss,
                          step=self.training_steps)
        tf.summary.scalar('Train/DistillLoss', mean_distill_loss,
                          step=self.training_steps)
        tf.summary.scalar('Train/HuberLoss', mean_td_loss,
                          step=self.training_steps)
        tf.summary.scalar('QL/Q-MaxQ', avg_q-max_q,
                          step=self.training_steps)
        tf.summary.scalar('QL/Q-TeacherQ', avg_q-teacher_q,
                          step=self.training_steps)
        tf.summary.scalar('QL/ActionGap', action_gap,
                          step=self.training_steps)
        tf.summary.scalar('QL/Q', avg_q,
                          step=self.training_steps)
        tf.summary.scalar('QL/MaxQ', max_q,
                          step=self.training_steps)
        tf.summary.scalar('Vars/lr',
                          self.optimizer_state.hyperparams['learning_rate'],
                          step=self.training_steps)
        tf.summary.scalar('Vars/distill_loss_decay', self.loss_decay,
                          step=self.training_steps)
      self.summary_writer.flush()

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state."""
    bundle_dictionary = super().bundle_and_checkpoint(
        checkpoint_dir, iteration_number)
    bundle_dictionary['online_training_steps'] = self.online_training_steps
    bundle_dictionary['loss_decay'] = self.loss_decay
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    if bundle_dictionary is not None:
      self.online_training_steps = bundle_dictionary['online_training_steps']
      self.loss_decay = bundle_dictionary['loss_decay']
      self.optimizer_state.hyperparams['learning_rate'] = self.learning_rate_fn(
          self.loss_decay)
    return super().unbundle(checkpoint_dir, iteration_number, bundle_dictionary)
