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
"""DQN agent that can be restarted from an checkpoint of an another Q-learning agent."""

import functools

from absl import logging
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.atari_100k import atari_100k_rainbow_agent as augmented_rainbow
from flax import core
from flax.training import checkpoints
import gin
import jax
from reincarnating_rl import persistent_dqn_agent
import tensorflow as tf


@gin.configurable
class ReloadedDQNAgent(persistent_dqn_agent.PersistentDQNAgent):
  """A variant of DQN that reloads from an existing DQN agent."""

  def __init__(
      self,
      num_actions,
      reload_optimizer=True,
      load_replay=True,
      summary_writer=None,
      num_updates_per_train_step=1,
      data_augmentation=False,
      preprocess_fn=augmented_rainbow.preprocess_inputs_with_augmentation,
      seed=None):

    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t reload_optimizer: %s', reload_optimizer)
    logging.info('\t num_updates_per_train_step: %d',
                 num_updates_per_train_step)
    super().__init__(num_actions, preprocess_fn=preprocess_fn,
                     summary_writer=summary_writer, seed=seed)
    self._reload_optimizer = reload_optimizer
    self.num_updates_per_train_step = num_updates_per_train_step
    self.load_replay = load_replay
    self.train_preprocess_fn = functools.partial(
        augmented_rainbow.preprocess_inputs_with_augmentation,
        data_augmentation=data_augmentation)

  @property
  def load_teacher_checkpoint(self):
    # Whether to load teacher checkpoint
    return True

  def set_phase(self, persistence=False):  # pylint:disable=unused-error
    pass

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_params to target_network_params if training
    steps is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for _ in range(self.num_updates_per_train_step):
          self._train_update()
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()
    self.training_steps += 1

  def _train_update(self):
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    self._sample_from_replay_buffer()
    self._rng, rng1 = jax.random.split(self._rng, num=2)
    states = self.train_preprocess_fn(
        self.replay_elements['state'], rng=rng1)
    next_states = self.replay_elements['next_state']
    next_states = self.preprocess_fn(next_states)
    self.optimizer_state, self.online_params, loss = dqn_agent.train(
        self.network_def, self.online_params, self.target_network_params,
        self.optimizer, self.optimizer_state, states,
        self.replay_elements['action'], next_states,
        self.replay_elements['reward'], self.replay_elements['terminal'],
        self.cumulative_gamma, self._loss_type)
    if (self.summary_writer is not None and self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      with self.summary_writer.as_default():
        tf.summary.scalar('HuberLoss', loss, step=self.training_steps)
      self.summary_writer.flush()

  def reload_replay_buffer(self, checkpoint_dir, iteration_number):
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError as e:
      logging.warning('Unable to reload replay buffer!')
      raise e

  def reload_checkpoint(self, bundle_dictionary):
    """Reload variables from a fully specified checkpoint.

    Args:
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.: string, full path to a checkpoint to reload.
    """
    if bundle_dictionary is not None:
      self.training_steps = bundle_dictionary['training_steps']
      if isinstance(bundle_dictionary['online_params'], core.FrozenDict):
        self.online_params = bundle_dictionary['online_params']
        self.target_network_params = bundle_dictionary['target_params']
      else:  # Load pre-linen checkpoint.
        self.online_params = core.FrozenDict({
            'params': checkpoints.convert_pre_linen(
                bundle_dictionary['online_params']).unfreeze()
        })
        self.target_network_params = core.FrozenDict({
            'params': checkpoints.convert_pre_linen(
                bundle_dictionary['target_params']).unfreeze()
        })
      # We recreate the optimizer with the new online weights.
      if self._reload_optimizer and ('optimizer_state' in bundle_dictionary):
        self.optimizer_state = bundle_dictionary['optimizer_state']
      else:
        self.optimizer_state = self.optimizer.init(self.online_params)
      if 'state' in bundle_dictionary:
        self.state = bundle_dictionary['state']
      logging.info('Done restoring!')

