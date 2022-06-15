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
"""PersistentDQNAgent that uses distillation for persistence."""

import functools

from absl import logging
from dopamine.jax.agents.dqn import dqn_agent
import gin
import jax
import jax.numpy as jnp
import optax
from reincarnating_rl import loss_helpers
from reincarnating_rl import persistent_dqn_agent
import tensorflow as tf

DistillType = loss_helpers.DistillType


@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'optimizer', 'cumulative_gamma',
                     'loss_type', 'dr3_coefficient', 'distill_loss_coefficient',
                     'distill_temperature', 'distill_best_action_only',
                     'distill_type', 'use_vision_transformer'))
def train_and_distill(network_def,
                      online_params,
                      target_params,
                      optimizer,
                      optimizer_state,
                      target_q_values,
                      states,
                      actions,
                      next_states,
                      rewards,
                      terminals,
                      loss_decay_multiplier,
                      cumulative_gamma,
                      loss_type='huber',
                      dr3_coefficient=0.0,
                      distill_loss_coefficient=0.0,
                      distill_temperature=0.1,
                      distill_best_action_only=False,
                      distill_type=DistillType.POLICY_ONLY,
                      use_vision_transformer=False):
  """Run the training step."""

  def loss_fn(params, target, teacher_q_values):

    def q_online(state):
      if use_vision_transformer:
        return network_def.apply(params, state, train=True)
      return network_def.apply(params, state)

    vmapped_q_online = jax.vmap(q_online)
    model_output = vmapped_q_online(states)
    q_values = jnp.squeeze(model_output.q_values)

    # Compute distillation loss
    def distillation_loss_fn():
      return loss_helpers.distillation_loss(
          q_values,
          distill_temperature,
          teacher_q_values,
          distill_best_action_only=distill_best_action_only,
          distill_type=distill_type)

    distill_coeff = loss_decay_multiplier * distill_loss_coefficient
    zero_loss_fn = lambda: jnp.array(0.0)
    distill_loss = jax.lax.cond(
        distill_coeff > 0, distillation_loss_fn, zero_loss_fn)
    # Compute TD loss
    train_loss, (avg_q, action_gap,
                 max_q) = loss_helpers.q_learning_loss(q_values, target,
                                                       actions, loss_type)

    # Compute DR3 loss
    regularization_loss = distill_coeff * distill_loss
    if not use_vision_transformer:
      state_representations = jnp.squeeze(model_output.representation)
      next_state_output = vmapped_q_online(next_states)
      next_state_representations = jnp.squeeze(next_state_output.representation)
      dr3_loss = loss_helpers.compute_dr3_loss(state_representations,
                                               next_state_representations)
      regularization_loss += loss_decay_multiplier * dr3_coefficient * dr3_loss
    else:
      dr3_loss = 0.0
    overall_loss = train_loss +  regularization_loss
    return overall_loss, (train_loss, distill_loss, dr3_loss, avg_q, action_gap,
                          max_q)

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(q_target, next_states, rewards, terminals,
                              cumulative_gamma)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (overall_loss, individual_losses), grad = grad_fn(online_params, target,
                                                    target_q_values)
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return (optimizer_state, online_params, overall_loss, individual_losses)


@gin.configurable
class DistillationDQNAgent(persistent_dqn_agent.PersistentDQNAgent):
  """Compact implementation of an agent that is reloaded using another Q-agent."""

  def __init__(
      self,
      num_actions,
      distill_loss_coefficient=1.0,
      distill_best_action_only=False,
      summary_writer=None,
      distill_type=DistillType.POLICY_ONLY,
      distill_decay_period=-1,  # -1 corresponds to no decay.
      distill_temperature=1.0,
      dr3_coefficient=0.0,
      seed=None):

    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t distill_loss_coefficient: %.4f', distill_loss_coefficient)
    logging.info('\t distill_best_action_only: %s', distill_best_action_only)
    logging.info('\t distill_type: %s', distill_type)
    logging.info('\t distill_decay_period: %d', distill_decay_period)
    logging.info('\t dr3_coefficient: %.4f', dr3_coefficient)
    logging.info('\t distill_temperature: %.4f', distill_temperature)
    super().__init__(num_actions, summary_writer=summary_writer, seed=seed)
    self.distill_loss_coefficient = distill_loss_coefficient
    self.distill_best_action_only = distill_best_action_only
    self.distill_type = distill_type
    # No. of steps within which to decay distillation loss coefficient.
    self.distill_decay_period = distill_decay_period
    self.online_training_steps = 0
    self.dr3_coefficient = dr3_coefficient
    self.distill_temperature = distill_temperature

  def set_phase(self, persistence=False):
    self._persistent_phase = persistence
    if not persistence:
      self._sync_weights()  # Sync online and target network before training.

  def _build_networks_and_optimizer(self):
    super()._build_networks_and_optimizer()
    self.loss_decay = 1.0
    self.learning_rate_fn = loss_helpers.create_linear_schedule()
    self.optimizer = loss_helpers.create_persistence_optimizer(
        self._optimizer_name, inject_hparams=True)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.optimizer_state.hyperparams['learning_rate'] = self.learning_rate_fn(
        self.loss_decay)

  def training_step(self):
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self._distillation_step(
            dr3_coefficient=0.0, cumulative_gamma=self.cumulative_gamma)
        if self.training_steps % self.target_update_period == 0:
          self._sync_weights()
    self.training_steps += 1
    self.online_training_steps += 1

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state."""
    bundle_dictionary = super().bundle_and_checkpoint(checkpoint_dir,
                                                      iteration_number)
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

  def _persistence_step(self):
    self._sample_from_teacher_replay_buffer()
    self.replay_elements = self.teacher_replay_elements
    self._distillation_step(
        dr3_coefficient=self.dr3_coefficient,
        cumulative_gamma=self.pretraining_cumulative_gamma)
    if self.training_steps % self.persistence_target_update_period == 0:
      self._sync_weights()

  def record_score(self, normalized_score):
    # Decay the coefficients of distillation loss.
    super().record_score(normalized_score)
    if (not self._persistent_phase) and self.loss_decay > 0:
      self.loss_decay = max(1.0 - normalized_score, 0.0)
      if (self.distill_decay_period > 0 and
          self.online_training_steps > self.distill_decay_period):
        self.loss_decay = 0
      # Decay the learning rate based ondistillation loss decay.
      self.optimizer_state.hyperparams['learning_rate'] = self.learning_rate_fn(
          self.loss_decay)

  def _distillation_step(self, dr3_coefficient, cumulative_gamma):
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    raw_states = self.replay_elements['state']
    states = self.train_preprocess_fn(raw_states, rng=rng1)
    next_states = self.train_preprocess_fn(
        self.replay_elements['next_state'], rng=rng2)
    # Teacher agent uses its own pre-processing.
    teacher_q_values = jax.lax.stop_gradient(
        self.teacher_agent.get_q_values(raw_states))

    (self.optimizer_state, self.online_params, overall_loss,
     individual_losses) = train_and_distill(
         self.network_def,
         self.online_params,
         self.target_network_params,
         self.optimizer,
         self.optimizer_state,
         teacher_q_values,
         states,
         self.replay_elements['action'],
         next_states,
         self.replay_elements['reward'],
         self.replay_elements['terminal'],
         self.loss_decay,
         cumulative_gamma,
         self._loss_type,
         dr3_coefficient=dr3_coefficient,
         distill_loss_coefficient=self.distill_loss_coefficient,
         distill_temperature=self.distill_temperature,
         distill_best_action_only=self.distill_best_action_only,
         distill_type=self.distill_type,
         use_vision_transformer=self.use_vision_transformer)

    teacher_q = jnp.mean(jnp.max(teacher_q_values, axis=-1))
    avg_q, action_gap, max_q = individual_losses[3:6]
    if self.training_steps % self.summary_writing_frequency == 0:
      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          tf.summary.scalar('Train/HuberLoss', individual_losses[0],
                            step=self.training_steps)
          tf.summary.scalar('Train/DistillLoss', individual_losses[1],
                            step=self.training_steps)
          tf.summary.scalar('Train/DR3Loss', individual_losses[2],
                            step=self.training_steps)
          tf.summary.scalar('Train/OverallLoss', overall_loss,
                            step=self.training_steps)
          tf.summary.scalar('QL/AvgQ', avg_q,
                            step=self.training_steps)
          tf.summary.scalar('QL/ActionGap', action_gap,
                            step=self.training_steps)
          tf.summary.scalar('QL/MaxQ', max_q,
                            step=self.training_steps)
          tf.summary.scalar('QL/Q-TeacherQ', avg_q - teacher_q,
                            step=self.training_steps)
          tf.summary.scalar('QL/Q-MaxQ', avg_q - max_q,
                            step=self.training_steps)
          tf.summary.scalar('Train/distill_loss_decay', self.loss_decay,
                            step=self.training_steps)
        self.summary_writer.flush()
