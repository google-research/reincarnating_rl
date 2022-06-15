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
"""Change how we explore based on teacher."""

import functools

from absl import logging
import gin
import jax
import numpy as onp
from reincarnating_rl import loss_helpers
from reincarnating_rl import persistent_dqn_agent
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def select_roll_out_action(roll_out_guide_prob, roll_out_decay_fn,
                           roll_out_decay_period, training_steps, rng):
  """Select an action from teacher vs student."""
  roll_out_prob = roll_out_guide_prob * roll_out_decay_fn(
      roll_out_decay_period, training_steps, None, 0.0)
  rng, rng1 = jax.random.split(rng, num=2)
  return rng, jax.random.uniform(rng1) <= roll_out_prob


@gin.configurable
class LOLSDQNAgent(persistent_dqn_agent.PersistentDQNAgent):
  """Uses guide policy for rolling in to kickstart learning."""

  def __init__(
      self,
      num_actions,
      summary_writer=None,
      roll_in_guide=False,
      roll_out_guide_prob=1.0,
      roll_out_decay_fn=loss_helpers.persistence_linearly_decaying_epsilon,
      roll_out_decay_period=250000,
      decay_roll_in_steps=1.0,
      max_roll_in_steps=100,
      seed=None):
    super().__init__(
        num_actions,
        seed=seed,
        summary_writer=summary_writer)

    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('roll_in_guide: %s', roll_in_guide)
    logging.info('roll_out_guide_prob %.4f:', roll_out_guide_prob)
    logging.info('decay_roll_in_steps %.4f:', decay_roll_in_steps)
    logging.info('max_roll_in_steps %d:', max_roll_in_steps)
    logging.info('roll_out_decay_period %d:', roll_out_decay_period)
    self.roll_out_guide_prob = roll_out_guide_prob
    self.roll_in_guide = roll_in_guide
    self.max_roll_in_steps = max_roll_in_steps  # set to 1 / (1 - γ) by default
    self.roll_out_decay_period = roll_out_decay_period
    self.roll_out_decay_fn = roll_out_decay_fn
    self._decay_roll_in_steps = decay_roll_in_steps

  def record_score(self, normalized_score):
    # Performance of agent as a fraction of teacher performance
    super().record_score(normalized_score)
    # Decay number of `max_roll_in_steps` every iteration
    self.max_roll_in_steps = int(
        self.max_roll_in_steps * self._decay_roll_in_steps)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode."""
    # Randomly selected number of roll in steps for this episode.
    if self.max_roll_in_steps >= 0:
      self.num_roll_in_steps = onp.random.randint(0, self.max_roll_in_steps) + 1
    else:
      self.num_roll_in_steps = 0.0
    # Count of roll in steps executed so far.
    self.roll_in_steps_executed = 1
    # Run the agent in eval mode.
    if not self.eval_mode and self.roll_in_guide:
      return self.guide_begin_episode(observation)
    return super().begin_episode(observation)

  def guide_begin_episode(self, observation):
    self._reset_state()
    self._record_observation(observation)

    # Guide begins episode in the train mode.
    self._train_step()
    self.action = self.teacher_agent.get_action(self.state)
    return self.action

  def guide_step(self, reward, observation):
    self._last_observation = self._observation
    self._record_observation(observation)

    # Guide step is always run in the train mode.
    self._store_transition(self._last_observation, self.action, reward, False)
    self._train_step()

    self.action = self.teacher_agent.get_action(self.state)
    return self.action

  def step(self, reward, observation):
    if self.eval_mode:
      return super().step(reward, observation)
    if self.roll_in_steps_executed < self.num_roll_in_steps:
      # If we use guide policy for rolling in.
      self.roll_in_steps_executed += 1
      if self.roll_in_guide:
        return self.guide_step(reward, observation)
      return super().step(reward, observation)

    # Roll out mixture policy.
    self._rng, select_teacher_action = select_roll_out_action(
        self.roll_out_guide_prob, self.roll_out_decay_fn,
        self.roll_out_decay_period, self.training_steps, self._rng)
    if select_teacher_action:
      return self.guide_step(reward, observation)
    return super().step(reward, observation)

  def _train_step(self):
    """Runs a single training step."""
    for _ in range(self._num_updates_per_persistent_step):
      super()._original_train_step()
    if self.training_steps % self.summary_writing_frequency == 0:
      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          tf.summary.scalar('Vars/max_roll_in_steps', self.max_roll_in_steps,
                            step=self.training_steps)
          tf.summary.scalar('Vars/num_roll_in_steps', self.num_roll_in_steps,
                            step=self.training_steps)
