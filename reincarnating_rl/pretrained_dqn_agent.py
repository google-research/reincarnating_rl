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
"""DQN agent that use persistence via using values from teacher for bootstrapping."""

import functools

from absl import logging
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.atari_100k import atari_100k_rainbow_agent as augmented_rainbow
import gin
import jax
import jax.numpy as jnp
import optax
from reincarnating_rl import loss_helpers
from reincarnating_rl import persistence_networks  # pylint:disable=unused-import
from reincarnating_rl import persistent_dqn_agent
import tensorflow as tf


def _loss_fn_train(loss_fn, optimizer, optimizer_state, online_params, target):
  """Helper for creating training loss."""
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (overall_loss, individual_losses), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, overall_loss, individual_losses


@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'optimizer', 'cumulative_gamma',
                     'loss_type', 'cql_coefficient', 'distill_temperature',
                     'td_loss_coefficient', 'use_vision_transformer'))
def offline_pretrain(network_def,
                     online_params,
                     target_params,
                     optimizer,
                     optimizer_state,
                     states,
                     actions,
                     next_states,
                     rewards,
                     terminals,
                     cumulative_gamma,
                     loss_type='huber',
                     distill_temperature=1.0,
                     cql_coefficient=1.0,
                     td_loss_coefficient=1.0,
                     use_vision_transformer=False):
  """Run the offline pretraining step."""

  def loss_fn(params, target):
    def q_online(state):
      if use_vision_transformer:
        return network_def.apply(params, state, train=True)
      return network_def.apply(params, state)

    q_values = jnp.squeeze(jax.vmap(q_online)(states).q_values)
    train_loss, q_statistics = loss_helpers.q_learning_loss(
        q_values, target, actions, loss_type=loss_type)
    cql_loss = jnp.mean(loss_helpers.batch_cql_loss(
        q_values, actions, distill_temperature=distill_temperature))
    overall_loss = td_loss_coefficient * train_loss + cql_coefficient * cql_loss
    return overall_loss, (train_loss, cql_loss, q_statistics)

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(q_target, next_states, rewards, terminals,
                              cumulative_gamma)

  return _loss_fn_train(
      loss_fn, optimizer, optimizer_state, online_params, target)


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          loss_type='huber', use_vision_transformer=False):
  """Run the online RL training step."""
  def loss_fn(params, target):
    def q_online(state):
      if use_vision_transformer:
        return network_def.apply(params, state, train=True)
      return  network_def.apply(params, state)

    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    return loss_helpers.q_learning_loss(
        q_values, target, actions, loss_type=loss_type)

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(q_target, next_states, rewards, terminals,
                              cumulative_gamma)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (avg_q, action_gap, max_q)), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, (avg_q, action_gap, max_q)


@gin.configurable
class PretrainedDQNAgent(persistent_dqn_agent.PersistentDQNAgent):
  """Uses offline pretraining to kickstart learning."""

  def __init__(
      self,
      num_actions,
      teacher_replay_scheme='uniform',
      teacher_data_ratio=0.0,
      cql_coefficient=1.0,
      td_loss_coefficient=1.0,
      distill_temperature=1.0,
      summary_writer=None,
      preprocess_fn=augmented_rainbow.preprocess_inputs_with_augmentation,
      seed=None):
    super().__init__(
        num_actions,
        # Set preprocessing function to avoid input preprocessing inside
        # value networks.
        preprocess_fn=preprocess_fn,
        seed=seed,
        summary_writer=summary_writer)

    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t teacher_replay_scheme: %s', teacher_replay_scheme)
    logging.info('\t distill_temperature: %.4f', distill_temperature)
    logging.info('\t cql_coefficient: %.4f', cql_coefficient)
    logging.info('\t teacher_data_ratio: %.4f', teacher_data_ratio)

    self._teacher_replay_scheme = teacher_replay_scheme
    # Probability that a given sample is from the teacher.
    self.teacher_data_ratio = teacher_data_ratio
    self.distill_temperature = distill_temperature
    self.cql_coefficient = cql_coefficient
    self.td_loss_coefficient = td_loss_coefficient

  def _build_networks_and_optimizer(self):
    super()._build_networks_and_optimizer()
    self.pretraining_optimizer = loss_helpers.create_persistence_optimizer(
        self._optimizer_name, inject_hparams=True)
    self.pretraining_optimizer_state = self.pretraining_optimizer.init(
        self.online_params)

  def set_phase(self, persistence=False):
    if self._persistent_phase and not persistence:
      self._sync_weights()
      self.optimizer_state = self.optimizer.init(self.online_params)
    self._persistent_phase = persistence

  def _sample_from_replay_buffer(self):
    super()._sample_from_replay_buffer()
    if self.teacher_data_ratio > 0:
      self._sample_from_teacher_replay_buffer()
      batch_size = len(self.replay_elements['action'])
      self._rng, rng1 = jax.random.split(self._rng, num=2)
      sampling_probs = jax.random.uniform(rng1, shape=(batch_size,))
      condition = (sampling_probs <= self.teacher_data_ratio)
      replay_elements = self.replay_elements.copy()
      for key, elements in replay_elements.items():
        cond_for_key = loss_helpers.expand_dims(
            condition.copy(), len(elements.shape) - 1)
        teacher_elements = self.teacher_replay_elements[key]
        self.replay_elements[key] = jnp.where(
            cond_for_key, teacher_elements, elements)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state."""
    bundle_dictionary = super().bundle_and_checkpoint(
        checkpoint_dir, iteration_number)
    bundle_dictionary.update({
        'pretraining_optimizer_state': self.pretraining_optimizer_state
    })
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint."""
    status = super().unbundle(
        checkpoint_dir, iteration_number, bundle_dictionary)
    if bundle_dictionary is not None:
      if 'pretraining_optimizer_state' in bundle_dictionary:
        self.pretraining_optimizer_state = bundle_dictionary[
            'pretraining_optimizer_state']
      else:
        self.pretraining_optimizer_state = self.optimizer.init(
            self.online_params)
    return status

  def _persistence_step(self):
    self._sample_from_teacher_replay_buffer()
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    states = self.train_preprocess_fn(
        self.teacher_replay_elements['state'], rng=rng1)
    next_states = self.train_preprocess_fn(
        self.teacher_replay_elements['next_state'], rng=rng2)

    (self.pretraining_optimizer_state, self.online_params, overall_loss,
     individual_losses) = offline_pretrain(
         self.network_def, self.online_params, self.target_network_params,
         self.pretraining_optimizer, self.pretraining_optimizer_state, states,
         self.teacher_replay_elements['action'], next_states,
         self.teacher_replay_elements['reward'],
         self.teacher_replay_elements['terminal'],
         self.pretraining_cumulative_gamma,
         self._loss_type, self.distill_temperature, self.cql_coefficient,
         self.td_loss_coefficient,
         use_vision_transformer=self.use_vision_transformer)

    avg_q, action_gap, max_q = individual_losses[2]
    if (self.summary_writer is not None and
        self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      with self.summary_writer.as_default():
        tf.summary.scalar('Pretrain/OverallLoss', overall_loss,
                          step=self.training_steps)
        tf.summary.scalar('Pretrain/CQLLoss', individual_losses[1],
                          step=self.training_steps)
        tf.summary.scalar('Train/HuberLoss', individual_losses[0],
                          step=self.training_steps)
        tf.summary.scalar('OfflineQL/AvgQ', avg_q,
                          step=self.training_steps)
        tf.summary.scalar('OfflineQL/ActionGap', action_gap,
                          step=self.training_steps)
        tf.summary.scalar('OfflineQL/MaxQ', max_q,
                          step=self.training_steps)
        tf.summary.scalar('OfflineQL/Q-MaxQ', avg_q - max_q,
                          step=self.training_steps)
      self.summary_writer.flush()

    if self.training_steps % self.persistence_target_update_period == 0:
      self._sync_weights()

  def training_step(self):
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
        states = self.train_preprocess_fn(
            self.replay_elements['state'], rng=rng1)
        next_states = self.train_preprocess_fn(
            self.replay_elements['next_state'], rng=rng2)
        self.optimizer_state, self.online_params, loss, q_vals = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            self.replay_elements['action'],
            next_states,
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.cumulative_gamma,
            self._loss_type,
            use_vision_transformer=self.use_vision_transformer)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            tf.summary.scalar('Train/HuberLoss', loss,
                              step=self.training_steps)
            tf.summary.scalar('QL/AvgQ', q_vals[0],
                              step=self.training_steps)
            tf.summary.scalar('QL/ActionGap', q_vals[1],
                              step=self.training_steps)
            tf.summary.scalar('QL/MaxQ', q_vals[2],
                              step=self.training_steps)
            tf.summary.scalar('QL/Q-MaxQ', q_vals[0] - q_vals[2],
                              step=self.training_steps)
          self.summary_writer.flush()
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
