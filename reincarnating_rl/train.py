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

r"""Entry point for Reincarnating RL experiments.

# pylint: disable=line-too-long

"""

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains import train as base_train
from jax.config import config
import numpy as np
from reincarnating_rl import dqfd_dqn_agent
from reincarnating_rl import jsrl_dqn_agent
from reincarnating_rl import pretrained_dqn_agent
from reincarnating_rl import qdagger_dqn_agent
from reincarnating_rl import qdagger_rainbow_agent
from reincarnating_rl import reloaded_dqn_agent
from reincarnating_rl import run_experiment
from reincarnating_rl import teacher_dqn_agent
from reincarnating_rl import teacher_rainbow_agent
import tensorflow as tf


FLAGS = flags.FLAGS
AGENTS = [
    'qdagger_dqn',
    'reloaded_dqn',
    'pretrained_dqn',
    'jsrl_dqn',
    'qdagger_rainbow',
    'dqfd_dqn',
]
TEACHER_AGENTS = ['dqn']
PRETRAINING_AGENTS = [
    'pretrained_dqn',
    'qdagger_dqn',
    'qdagger_rainbow',
    'dqfd_dqn',
]

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent', 'qdagger_dqn', AGENTS, 'Name of the agent.')
flags.DEFINE_boolean('disable_jit', False, 'Whether to use jit or not.')
flags.DEFINE_enum('teacher_agent', 'dqn', TEACHER_AGENTS, 'Teacher agent name.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_string(
    'teacher_checkpoint_dir', None,
    'Directory from which to load the teacher agent checkpoints.')
flags.DEFINE_integer(
    'teacher_checkpoint_number', None, 'Checkpoint number of the teacher agent '
    'that needs to be loaded.')
flags.DEFINE_string(
    'teacher_checkpoint_file_prefix', 'ckpt', 'Checkpoint prefix')




def create_agent(
    sess,  # pylint: disable=unused-argument
    environment,
    seed,
    agent='rainbow',
    summary_writer=None):
  """Create persistent agent which pretrains using a teacher agent."""

  if agent == 'qdagger_dqn':
    agent_fn = qdagger_dqn_agent.QDaggerDQNAgent
  elif agent == 'qdagger_rainbow':
    # Pass a separate gin config for DrQ/Full Rainbow agent.
    agent_fn = qdagger_rainbow_agent.QDaggerRainbowAgent
  elif agent == 'reloaded_dqn':
    agent_fn = reloaded_dqn_agent.ReloadedDQNAgent
  elif agent == 'pretrained_dqn':
    agent_fn = pretrained_dqn_agent.PretrainedDQNAgent
  elif agent == 'jsrl_dqn':
    agent_fn = jsrl_dqn_agent.JSRLAgent
  elif agent == 'dqfd_dqn':
    agent_fn = dqfd_dqn_agent.DQfDAgent
  else:
    raise ValueError(f'{agent} is not defined.')

  return agent_fn(
      num_actions=environment.action_space.n,
      seed=seed,
      summary_writer=summary_writer)


def create_teacher_agent(environment,
                         teacher_agent='dqn',
                         summary_writer=None):
  """Helper function for creating teacher agent."""

  if teacher_agent == 'dqn':
    return teacher_dqn_agent.TeacherDQNAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  elif teacher_agent == 'rainbow':
    return teacher_rainbow_agent.TeacherRainbowAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  else:
    raise ValueError(f'{teacher_agent} is not a defined agent.')


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  logging.info('Setting random seed: %d', seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  if FLAGS.disable_jit:
    config.update('jax_disable_jit', True)
  base_dir = FLAGS.base_dir
  gin_files = FLAGS.gin_files
  gin_bindings = FLAGS.gin_bindings
  if FLAGS.teacher_checkpoint_dir is not None:
    teacher_checkpoint_dir = os.path.join(FLAGS.teacher_checkpoint_dir,
                                          'checkpoints')
  else:
    teacher_checkpoint_dir = None
  # Add code for setting random seed using the run_number
  set_random_seed(FLAGS.run_number)
  base_run_experiment.load_gin_configs(gin_files, gin_bindings)
  # Set the Jax agent seed using the run number
  create_agent_fn = functools.partial(
      create_agent, agent=FLAGS.agent, seed=FLAGS.run_number)
  create_teacher_agent_fn = functools.partial(
      create_teacher_agent, teacher_agent=FLAGS.teacher_agent)

  if FLAGS.agent in PRETRAINING_AGENTS:
    runner_fn = run_experiment.ReincarnationRunner
  else:
    runner_fn = run_experiment.RunnerWithTeacher
  runner = runner_fn(
      base_dir,
      create_agent_fn,
      create_teacher_agent_fn=create_teacher_agent_fn,
      teacher_checkpoint_dir=teacher_checkpoint_dir,
      teacher_checkpoint_file_prefix=FLAGS.teacher_checkpoint_file_prefix,
      teacher_checkpoint_number=FLAGS.teacher_checkpoint_number)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
