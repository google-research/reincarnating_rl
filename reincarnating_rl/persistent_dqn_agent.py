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
"""DQN agent that can be restarted from checkpoint of another Q-learning agent."""

import collections
import functools
from absl import logging
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.atari_100k import atari_100k_rainbow_agent as augmented_rainbow
from dopamine.replay_memory import prioritized_replay_buffer
import gin
from reincarnating_rl import impala_networks
from reincarnating_rl import loss_helpers
from reincarnating_rl import persistence_networks  # pylint:disable=unused-import

PRIORITIZED_BUFFERS = [
    prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer,
]


@gin.configurable
class PersistentDQNAgent(dqn_agent.JaxDQNAgent):
  """Compact implementation of an agent that is reloaded using another Q-agent."""

  def __init__(
      self,
      num_actions,
      num_updates_per_persistent_step=1,
      network=impala_networks.JAXDQNNetworkWithRepresentations,
      epsilon_fn=loss_helpers.persistence_linearly_decaying_epsilon,
      data_augmentation=False,
      summary_writer=None,
      summary_writing_frequency=2500,
      seed=None,
      preprocess_fn=augmented_rainbow.preprocess_inputs_with_augmentation):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      num_updates_per_persistent_step: int, Number of gradient updates every
        persistence RL step.
      network: Jax network to use for training.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      data_augmentation: Whether to use data augmentation.
      summary_writer: SummaryWriter object for outputting training statistics.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      seed: int, a seed for DQN's internal RNG, used for initialization and
        sampling actions. If None, will use the current time in nanoseconds.
      preprocess_fn: function expecting the input state as parameter which
        it preprocesses (such as normalizing the pixel values between 0 and 1)
        before passing it to the Q-network. Defaults to None.
    """

    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t num_updates_per_persistent_step: %d',
                 num_updates_per_persistent_step)
    logging.info('\t data_augmentation: %s', data_augmentation)
    logging.info('\t network: %s', network)

    super().__init__(
        num_actions,
        network=network,
        preprocess_fn=preprocess_fn,
        epsilon_fn=epsilon_fn,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        seed=seed)
    self.teacher_agent = None  # To be set explicitly by `set_teacher`
    self._num_updates_per_persistent_step = num_updates_per_persistent_step
    self._persistent_phase = False
    self.teacher_steps = None
    self.data_augmentation = data_augmentation
    self.persistence_target_update_period = (
        self.target_update_period // self.update_period)
    self._teacher_replay = None
    self.train_preprocess_fn = functools.partial(
        preprocess_fn, data_augmentation=data_augmentation)
    # If using Vision Transformer model, then need to pass explicit train flag.
    self.use_vision_transformer = (
        self.network_def.__class__.__name__ == 'AtariVisionTransformer')

  def record_score(self, normalized_score):
    # Performance of agent as a fraction of teacher performance
    self.normalized_score = normalized_score

  @property
  def load_teacher_checkpoint(self):
    # Whether to load teacher checkpoint
    return False

  def set_teacher(self, agent, teacher_steps=None):
    if not agent.eval_mode:
      raise AttributeError('Teacher agent should run in eval mode.')
    self.teacher_agent = agent
    # pylint:disable=protected-access
    if self.teacher_agent.load_replay:
      self._teacher_replay = self.teacher_agent._replay
    else:
      self._teacher_replay = self._replay
    self.pretraining_cumulative_gamma = self.teacher_agent.cumulative_gamma
    # pylint:enable=protected-access
    logging.info('\t teacher_agent: %s', agent)
    if teacher_steps is not None:
      self.teacher_steps = teacher_steps
      logging.info('\t Teacher steps set to: %d', teacher_steps)

  def _sample_from_teacher_replay_buffer(self):
    samples = self._teacher_replay.sample_transition_batch()
    types = self._teacher_replay.get_transition_elements()
    self.teacher_replay_elements = collections.OrderedDict()
    for element, element_type in zip(samples, types):
      self.teacher_replay_elements[element_type.name] = element

  def set_phase(self, persistence: bool = False):
    # Training was in persistent phase but switching to non-persistent phase.
    if self._persistent_phase and not persistence:
      self._sync_weights()  # Sync online and target network before training.
    self._persistent_phase = persistence

  def _train_step(self):
    """Runs a single training step."""
    if self._persistent_phase:
      # Only start updating if replay buffer contains a sufficient number of
      # data points. Multiple updates every step.
      if self._teacher_replay.add_count > self.teacher_steps:
        for _ in range(self._num_updates_per_persistent_step):
          self._persistence_step()
        self.training_steps += 1
    else:
      self.training_step()

  def training_step(self):
    return super()._train_step()

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        *args,
                        priority=None,
                        episode_end=False):
    """Stores a transition in teacher replay."""

    if self.training_steps < self.teacher_steps:
      teacher_priority = priority
      is_prioritized = any(
          isinstance(self._teacher_replay, buffer)
          for buffer in PRIORITIZED_BUFFERS)
      if is_prioritized and priority is None:
        if self._teacher_replay_scheme == 'uniform':
          teacher_priority = 1.
        else:
          teacher_priority = self._teacher_replay.sum_tree.max_recorded_priority

      if not self.eval_mode:
        self._teacher_replay.add(
            last_observation,
            action,
            reward,
            is_terminal,
            *args,
            priority=teacher_priority,
            episode_end=episode_end)
    else:
      # Add to standard replay buffer too.
      super()._store_transition(
          last_observation,
          action,
          reward,
          is_terminal,
          *args,
          priority=priority,
          episode_end=episode_end)

  def _original_train_step(self):
    super()._train_step()

  def _teacher_step(self, reward, observation):
    """Records the most recent transition and returns the teacher's next action."""
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()
    return self.teacher_agent.step(None, observation)

  def _persistence_step(self):
    raise NotImplementedError
