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
"""Helpers for loading tf checkpoints for JAX Dopamine agents."""

import os
import flax
import jax
import jax.numpy as jnp
import tensorflow as tf


def load_tf_nature_dqn_weights(checkpoint_path,
                               prefix='Online') -> flax.core.FrozenDict:
  """Load the TF NatureDQNNetwork weights and convert to a JAX array."""

  ckpt_reader = tf.train.load_checkpoint(checkpoint_path)
  jax_to_tf_layer_mapping = {
      'Conv_0': 'Conv',
      'Conv_1': 'Conv_1',
      'Conv_2': 'Conv_2',
      'Dense_0': 'fully_connected',
      'Dense_1': 'fully_connected_1',
  }
  params = {}
  for jax_layer, tf_layer in jax_to_tf_layer_mapping.items():
    params[jax_layer] = {
        'bias': ckpt_reader.get_tensor(f'{prefix}/{tf_layer}/biases'),
        'kernel': ckpt_reader.get_tensor(f'{prefix}/{tf_layer}/weights'),
    }
  jax_params = jax.tree_map(jnp.asarray, {'params': params})
  return flax.core.FrozenDict(jax_params)


def create_dqn_checkpoint_data(checkpoint_path, checkpoint_file_prefix,
                               iteration_number, auxiliary_info=True):
  """Loads tf Nature DQN weights and creates a dict to restore a JAX agent."""
  bundle_dictionary = {}
  tf_checkpoint_path = os.path.join(
      checkpoint_path, f'{checkpoint_file_prefix}-{iteration_number}')
  if auxiliary_info:
    bundle_dictionary['current_iteration'] = iteration_number
    # This assumes that the agent was trained using standard Dopamine params.
    bundle_dictionary['training_steps'] = (iteration_number + 1) * 250000
  bundle_dictionary['online_params'] = load_tf_nature_dqn_weights(
      tf_checkpoint_path, 'Online')
  bundle_dictionary['target_params'] = load_tf_nature_dqn_weights(
      tf_checkpoint_path, 'Target')
  return bundle_dictionary
