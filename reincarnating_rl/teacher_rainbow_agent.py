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
from dopamine.labs.atari_100k import atari_100k_rainbow_agent as augmented_rainbow_agent
from dopamine.replay_memory import sum_tree
from flax import core
from flax.training import checkpoints
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(0,))
def compute_q_values(network_def, online_params, states, key, support):
  def _q_online(state):
    return network_def.apply(online_params, state, key=key, support=support)
  return jnp.squeeze(jax.vmap(_q_online)(states).q_values)


@functools.partial(jax.jit, static_argnums=(0,))
def q_online(network_def, online_params, state, key, support):
  return network_def.apply(
      online_params, state, key=key, support=support).q_values


@functools.partial(jax.jit, static_argnums=(0,))
def argmax_action(network_def, online_params, state, key, support):
  return jnp.argmax(
      network_def.apply(online_params, state, key=key,
                        support=support).q_values)


@gin.configurable
class TeacherRainbowAgent(augmented_rainbow_agent.Atari100kRainbowAgent):
  """A variant of DQN that estimates Q-values of a fixed policy."""

  def __init__(self,
               num_actions,
               data_augmentation=False,
               load_replay=True,
               reload_optimizer=False,
               summary_writer=None,
               seed=None):
    logging.info('Creating a TeacherRainbowAgent with following params: ')
    logging.info('\t load_replay: %s', load_replay)
    logging.info('\t reload_optimizer: %s', reload_optimizer)
    self.load_replay = load_replay  # Used in RunnerWithTeacher.
    self._reload_optimizer = reload_optimizer
    super().__init__(
        num_actions,
        data_augmentation=data_augmentation,
        summary_writer=summary_writer,
        seed=seed)
    self.eval_mode = True

  def get_q_values(self, states):
    preprocessed_states = self.preprocess_fn(states)
    rng, self._rng = jax.random.split(self._rng)
    return compute_q_values(
        self.network_def, self.online_params, preprocessed_states,
        key=rng, support=self._support)

  def q_value(self, state):
    preprocessed_state = self.preprocess_fn(state)
    rng, self._rng = jax.random.split(self._rng)
    return q_online(self.network_def, self.online_params, preprocessed_state,
                    key=rng, support=self._support)

  def get_action(self, state):
    preprocessed_state = self.preprocess_fn(state)
    rng, self._rng = jax.random.split(self._rng)
    return onp.asarray(argmax_action(
        self.network_def, self.online_params, preprocessed_state,
        key=rng, support=self._support))

  def _train_step(self):
    logging.warning("Training step shouldn't be called!")
    pass

  def _original_train_step(self):
    super()._train_step()

  def reload_replay_buffer(self, checkpoint_dir, iteration_number):
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
      # Reset all priorities to be the same.
      self.set_uniform_priorities()
    except tf.errors.NotFoundError as e:
      logging.warning('Unable to reload replay buffer!')
      raise e

  def set_uniform_priorities(self):
    replay_capacity = self._replay._replay_capacity  # pylint: disable=protected-access
    self._replay.sum_tree = sum_tree.SumTree(replay_capacity)

  def reload_checkpoint(self, bundle_dictionary):
    """Reload variables from a fully specified checkpoint.

    Args:
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.: string, full path to a checkpoint to reload.
    """
    if bundle_dictionary is not None:
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
      if self._reload_optimizer:
        # We recreate the optimizer with the new online weights.
        if 'optimizer_state' in bundle_dictionary:
          self.optimizer_state = bundle_dictionary['optimizer_state']
        else:
          self.optimizer_state = self.optimizer.init(self.online_params)
      if 'state' in bundle_dictionary:
        self.state = bundle_dictionary['state']
      logging.info('Done restoring!')
