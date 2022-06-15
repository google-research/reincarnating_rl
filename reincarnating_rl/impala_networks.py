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
"""Networks for Impala-based agents."""

import collections
from typing import Tuple
from flax import linen as nn
import gin
import jax.numpy as jnp
NetworkType = collections.namedtuple('network', ['q_values', 'representation'])


def preprocess_atari_inputs(x):
  """Input normalization for Atari 2600 input frames."""
  return x.astype(jnp.float32) / 255.


@gin.configurable
class Stack(nn.Module):
  """Stack of pooling and convolutional blocks with residual connections."""
  num_ch: int
  num_blocks: int
  use_max_pooling: bool = True

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    conv_out = nn.Conv(
        features=self.num_ch,
        kernel_size=(3, 3),
        strides=1,
        kernel_init=initializer,
        padding='SAME')(
            x)
    if self.use_max_pooling:
      conv_out = nn.max_pool(
          conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2))

    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(features=self.num_ch, kernel_size=(3, 3),
                         strides=1, padding='SAME')(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(features=self.num_ch, kernel_size=(3, 3),
                         strides=1, padding='SAME')(conv_out)
      conv_out += block_input

    return conv_out


@gin.configurable
class ImpalaNetworkWithRepresentations(nn.Module):
  """Impala Network which also outputs penultimate representation layers."""
  num_actions: int
  inputs_preprocessed: bool = False
  nn_scale: int = 1
  stack_sizes: Tuple[int, ...] = (16, 32, 32)
  num_blocks: int = 2

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    conv_out = x

    for stack_size in self.stack_sizes:
      conv_out = Stack(num_ch=stack_size*self.nn_scale,
                       num_blocks=self.num_blocks,)(conv_out)

    conv_out = nn.relu(conv_out)
    conv_out = conv_out.reshape(-1)

    conv_out = nn.Dense(features=512, kernel_init=initializer)(conv_out)
    representation = nn.relu(conv_out)
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=initializer)(representation)
    return NetworkType(q_values, representation)


@gin.configurable
class JAXDQNNetworkWithRepresentations(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    representation = nn.relu(x)  # Use penultimate layer as representation
    q_values = nn.Dense(
        features=self.num_actions, kernel_init=initializer)(
            representation)
    return NetworkType(q_values, representation)


@gin.configurable
class JAXLargeDQNNetworkWithRepresentations(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    representation = nn.relu(x)  # Use penultimate layer as representation
    q_values = nn.Dense(
        features=self.num_actions, kernel_init=initializer)(
            representation)
    return NetworkType(q_values, representation)


@gin.configurable
class JAXSmallDQNNetworkWithRepresentations(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=256, kernel_init=initializer)(x)
    representation = nn.relu(x)  # Use penultimate layer as representation
    q_values = nn.Dense(
        features=self.num_actions, kernel_init=initializer)(
            representation)
    return NetworkType(q_values, representation)
