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
"""Runner for launching persistent experiments."""

import copy
import os
import sys
import time

from absl import logging
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
import gin
import numpy as onp
from reincarnating_rl import atari_scores
from reincarnating_rl import checkpoint_helpers
import tensorflow as tf


@gin.configurable
def get_all_checkpoint_numbers(base_directory,
                               sentinel_file_identifier='checkpoint'):
  """Returns the version numbers of all the saved checkpoints."""

  sentinel = 'sentinel_{}_complete.*'.format(sentinel_file_identifier)
  glob = os.path.join(base_directory, sentinel)
  def extract_iteration(x):
    return int(x[x.rfind('.') + 1:])
  try:
    checkpoint_files = tf.io.gfile.glob(glob)
  except tf.errors.NotFoundError:
    logging.info('Could not find any checkpoint file')
    return [-1]
  try:
    iterations = sorted(
        [extract_iteration(x) for x in checkpoint_files], reverse=True)
    return iterations
  except ValueError:
    logging.info('Could not extract any checkpoint versions')
    return [-1]


@gin.configurable
class PersistentRunner(run_experiment.Runner):
  """Object that handles running Dopamine experiments for persistence agents."""

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_teacher_agent_fn,
               teacher_checkpoint_dir,
               num_persistence_steps=100000,
               teacher_steps=100000,
               num_persistence_iterations=1,
               teacher_checkpoint_number=None,
               evaluate_teacher=True,
               checkpoint_file_prefix='ckpt',
               teacher_checkpoint_file_prefix='ckpt'):

    super().__init__(
        base_dir,
        create_agent_fn,
        checkpoint_file_prefix=checkpoint_file_prefix)
    logging.info('\t Base directory: %s', base_dir)
    logging.info('\t Teacher checkpoint directory: %s', teacher_checkpoint_dir)
    logging.info('\t Num distillation iterations: %d',
                 num_persistence_iterations)
    logging.info('\t num_persistence_steps: %d', num_persistence_steps)
    logging.info('\t teacher_steps: %d', teacher_steps)
    self._teacher_checkpoint_dir = teacher_checkpoint_dir
    self._teacher_agent = create_teacher_agent_fn(
        self._environment, summary_writer=None)
    self._initialize_agent(self._teacher_agent, teacher_checkpoint_file_prefix,
                           teacher_checkpoint_number)
    if self._agent.load_teacher_checkpoint:
      # To be used for ReloadedDQNAgent.
      logging.info('Loading checkpoint for base agent: %d', teacher_steps)
      self._initialize_agent(self._agent, teacher_checkpoint_file_prefix,
                             teacher_checkpoint_number)
    self._agent.set_teacher(self._teacher_agent, teacher_steps)
    self._num_persistence_iterations = num_persistence_iterations
    self._num_persistence_steps = num_persistence_steps
    self.teacher_steps = teacher_steps
    self._evaluate_teacher = evaluate_teacher
    # Score of a random agent.
    game_name = gin.query_parameter('create_atari_environment.game_name')
    self._random_score = atari_scores.RANDOM_SCORES[game_name]

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists."""
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    # List of all checkpoints sorted from latest to oldest checkpoint.
    checkpoint_versions = get_all_checkpoint_numbers(self._checkpoint_dir)
    for checkpoint_version in checkpoint_versions:
      if checkpoint_version >= 0:
        experiment_data = self._checkpointer.load_checkpoint(checkpoint_version)
        if self._agent.unbundle(self._checkpoint_dir, checkpoint_version,
                                experiment_data):
          if experiment_data is not None:
            if ('logs' in experiment_data) and ('current_iteration'
                                                in experiment_data):
              self._logger.data = experiment_data['logs']
              self._start_iteration = experiment_data['current_iteration'] + 1
            else:
              logging.info(
                  'logs or current iteration not found for checkpoint %d',
                  checkpoint_version)
              continue
          logging.info('Reloaded checkpoint and will start from iteration %d',
                       self._start_iteration)
          return
        else:
          logging.info('Could not reload checkpoint %d', checkpoint_version)

  def _set_teacher_as_agent(self):
    self._current_agent = copy.copy(self._agent)
    self._agent = self._teacher_agent

  def _reset_agent(self):
    self._agent = self._current_agent

  def _initialize_agent(self,
                        agent,
                        checkpoint_file_prefix,
                        checkpoint_number=None):
    if not self._teacher_checkpoint_dir:
      logging.info(
          'No checkpoint directory passed for teacher agent. Using a randomly '
          'initialized teacher. Use this functionality only for testing.')
      return
    teacher_checkpointer = checkpointer.Checkpointer(
        self._teacher_checkpoint_dir, checkpoint_file_prefix)
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_number = checkpointer.get_latest_checkpoint_number(
        self._teacher_checkpoint_dir)
    if checkpoint_number is None:
      checkpoint_number = latest_checkpoint_number
    assert checkpoint_number <= latest_checkpoint_number, (
        f"checkpoint_number {checkpoint_number} doesn't exist")
    if checkpoint_number >= 0:
      agent_name = agent.__class__.__name__
      if 'tf' in checkpoint_file_prefix:
        if 'DQN' in agent_name:  # DQN Agent
          experiment_data = checkpoint_helpers.create_dqn_checkpoint_data(
              self._teacher_checkpoint_dir, checkpoint_file_prefix,
              checkpoint_number)
        else:
          raise ValueError(f'Cannot load tf checkpoint for {agent_name}')
      else:
        experiment_data = teacher_checkpointer.load_checkpoint(
            checkpoint_number)
      agent.reload_checkpoint(experiment_data)
      logging.info('Reloaded %s agent with checkpoint at iteration %d',
                   agent_name, experiment_data['current_iteration'])
      if agent.load_replay:
        try:
          agent.reload_replay_buffer(self._teacher_checkpoint_dir,
                                     checkpoint_number)
          logging.info('Reloaded %s agent replay corresponding to checkpoint'
                       ' %d', agent.__class__.__name__, checkpoint_number)
        except tf.errors.NotFoundError:
          replay_dir = os.path.join(
              os.path.dirname(self._teacher_checkpoint_dir), 'replay_logs')
          # Every checkpoint stores 1M transitions = 4M frames
          agent.reload_replay_buffer(replay_dir, (checkpoint_number + 1) // 4)
    else:
      logging.warning('Pass checkpoint number is less than 0.')

  def _run_teacher_evaluation(self):
    """Runs one iteration of teacher agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting teacher evaluation at iteration 0.')
    self._set_teacher_as_agent()
    num_episodes_eval, self._teacher_score = self._run_eval_phase(
        statistics)
    self._reset_agent()
    self._save_teacher_tensorboard_summaries(
        num_episodes_eval, self._teacher_score)
    return statistics.data_lists

  def _save_teacher_tensorboard_summaries(
      self, num_episodes_eval, average_reward_eval):
    """Save teacher statistics as tensorboard summaries."""

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Eval/Teacher/NumEpisodes', num_episodes_eval, step=0)
        tf.summary.scalar('Eval/Teacher/AverageReturns', average_reward_eval,
                          step=0)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Eval/Teacher/NumEpisodes', simple_value=num_episodes_eval),
          tf.compat.v1.Summary.Value(
              tag='Eval/Teacher/AverageReturns',
              simple_value=average_reward_eval)
      ])
      self._summary_writer.add_summary(summary, 0)  # Iteration 0

  def _record_score(self, iteration, statistics):
    # Performance of agent as a fraction of teacher performance.
    score = onp.mean(statistics['eval_episode_returns'])
    self.normalized_score = ((score - self._random_score) /
                             (self._teacher_score - self._random_score + 1e-8))
    if self._sess is None:
      tf.summary.scalar(
          'Eval/Normalized_Score', self.normalized_score, step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Eval/Normalized_Score', simple_value=self.normalized_score),
      ])
      self._summary_writer.add_summary(summary, iteration)

  def _run_one_iteration(self, iteration):
    statistics = super()._run_one_iteration(iteration)
    self._record_score(iteration, statistics)
    return statistics

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    if self._evaluate_teacher:
      self._run_teacher_evaluation()

    total_iterations = self._num_iterations + self._num_persistence_iterations
    if total_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    original_training_steps = self._training_steps
    for iteration in range(self._start_iteration, total_iterations):
      if iteration < self._num_persistence_iterations:
        self._agent.set_phase(persistence=True)
        self._training_steps = self._num_persistence_steps
        logging.info('Persistence iteration %d', iteration)
      elif iteration == self._num_persistence_iterations:
        self._agent.set_phase(persistence=False)
        self._training_steps = original_training_steps
        logging.info('Beginning training at iteration %d', iteration)
      statistics = self._run_one_iteration(iteration)
      self._agent.record_score(self.normalized_score)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
    self._summary_writer.flush()


@gin.configurable
class OfflinePretrainingRunner(PersistentRunner):
  """Persistent Runner for Offline Pretraining."""

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_teacher_agent_fn,
               teacher_checkpoint_dir,
               teacher_checkpoint_number=None,
               teacher_checkpoint_file_prefix='ckpt',
               use_offline_samples_only=False):
    super().__init__(
        base_dir,
        create_agent_fn,
        create_teacher_agent_fn,
        teacher_checkpoint_dir,
        teacher_checkpoint_file_prefix=teacher_checkpoint_file_prefix,
        teacher_checkpoint_number=teacher_checkpoint_number)
    logging.info('\t use_offline_samples_only: %s', use_offline_samples_only)
    self.use_offline_samples_only = use_offline_samples_only
    self._agent.use_offline_samples_only = use_offline_samples_only

  def _run_pretrain_phase(self):
    """Run pre-training phase."""
    self._agent.eval_mode = False
    start_time = time.time()

    for i in range(self._training_steps):
      if i % 1000 == 0:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Training step: {}/{}\r'.format(
            i, self._training_steps))
        sys.stdout.flush()
      self._agent._train_step()  # pylint: disable=protected-access

    time_delta = time.time() - start_time
    average_steps_per_second = self._training_steps / time_delta
    logging.info('Average training steps per second: %.2f',
                 average_steps_per_second)
    # Return dummy values to be logged.
    return 0, 0, average_steps_per_second

  def _run_train_phase(self, statistics):
    if self._offline_pretraining or self.use_offline_samples_only:
      return self._run_pretrain_phase()
    return super()._run_train_phase(statistics)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    if self._evaluate_teacher:
      self._run_teacher_evaluation()
    self._offline_pretraining = False

    total_iterations = (
        self._num_iterations + self._num_persistence_iterations + 1)
    if total_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    original_training_steps = self._training_steps
    for iteration in range(self._start_iteration, total_iterations):
      if iteration < self._num_persistence_iterations + 1:
        self._agent.set_phase(persistence=True)
        if iteration == 0:
          self._training_steps = self.teacher_steps
          logging.info(
              'Data collection iteration 0: Steps %d', self.teacher_steps)
          if self.teacher_steps <= 0:
            continue
        else:
          self._offline_pretraining = True
          self._training_steps = self._num_persistence_steps
          logging.info('Offline pretraining iteration %d: Steps %d', iteration,
                       self._num_persistence_steps)
      else:
        if iteration == self._num_persistence_iterations + 1:
          self._agent.set_phase(persistence=False)
        self._offline_pretraining = False
        logging.info('Beginning training at iteration %d', iteration)
        self._training_steps = original_training_steps
      statistics = self._run_one_iteration(iteration)
      self._agent.record_score(self.normalized_score)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
    self._summary_writer.flush()
