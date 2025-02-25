# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for rl_continuous experiments."""

from acme import wrappers
import dm_env
import gym

_VALID_TASK_SUITES = ('gym', 'control')


def make_environment(suite: str, task: str) -> dm_env.Environment:
  """Makes the requested continuous control environment.

  Args:
    suite: One of 'gym' or 'control'.
    task: Task to load. If `suite` is 'control', the task must be formatted as
      f'{domain_name}:{task_name}'

  Returns:
    An environment satisfying the dm_env interface expected by Acme agents.
  """

  if suite not in _VALID_TASK_SUITES:
    raise ValueError(
        f'Unsupported suite: {suite}. Expected one of {_VALID_TASK_SUITES}')

  if suite == 'gym':
    env = gym.make(task)
    # Make sure the environment obeys the dm_env.Environment interface.
    env = wrappers.GymWrapper(env)

  elif suite == 'control':
    # Load dm_suite lazily not require Mujoco license when not using it.
    from dm_control import suite as dm_suite  # pylint: disable=g-import-not-at-top
    domain_name, task_name = task.split(':')
    env = dm_suite.load(domain_name, task_name)
    env = wrappers.ConcatObservationWrapper(env)

  # Wrap the environment so the expected continuous action spec is [-1, 1].
  # Note: this is a no-op on 'control' tasks.
  env = wrappers.CanonicalSpecWrapper(env, clip=True)
  env = wrappers.SinglePrecisionWrapper(env)
  return env


import functools
from typing import Any, Dict, NamedTuple, Sequence

from acme import specs
from acme.agents.jax import ppo
from acme.agents.jax.multiagent.decentralized import factories
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_jax_utils
from acme.multiagent import types as ma_types

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

class CategoricalParams(NamedTuple):
  """Parameters for a categorical distribution."""
  logits: jnp.ndarray
  
def make_dram_ppo_networks(
    environment_spec: specs.EnvironmentSpec,
    hidden_layer_sizes: Sequence[int] = (64, 64),
) -> ppo.PPONetworks:
  """Returns PPO networks used by the agent in the dram environments."""

  # Check that dram environment is defined with discrete actions, 0-indexed
  assert np.issubdtype(environment_spec.actions.dtype, np.integer), (
      'Expected multigrid environment to have discrete actions with int dtype'
      f' but environment_spec.actions.dtype == {environment_spec.actions.dtype}'
  )
  assert environment_spec.actions.minimum == 0, (
      'Expected dram environment to have 0-indexed action indices, but'
      f' environment_spec.actions.minimum == {environment_spec.actions.minimum}'
  )
  num_actions = environment_spec.actions.maximum + 1

  def forward_fn(inputs):
    processed_inputs = inputs
    trunk = hk.nets.MLP(hidden_layer_sizes, activation=jnp.tanh)
    h = trunk(processed_inputs)
    logits = hk.Linear(num_actions)(h)
    values = hk.Linear(1)(h)
    values = jnp.squeeze(values, axis=-1)
    return (CategoricalParams(logits=logits), values)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = acme_jax_utils.zeros_like(environment_spec.observations)
  dummy_obs = acme_jax_utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
  return make_categorical_ppo_networks(network)  # pylint:disable=undefined-variable

def make_categorical_ppo_networks(
    network: networks_lib.FeedForwardNetwork) -> ppo.PPONetworks:
  """Constructs a PPONetworks for Categorical Policy from FeedForwardNetwork.

  Args:
    network: a transformed Haiku network (or equivalent in other libraries) that
      takes in observations and returns the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """

  def log_prob(params: CategoricalParams, action):
    return tfd.Categorical(logits=params.logits).log_prob(action)

  def entropy(params: CategoricalParams):
    return tfd.Categorical(logits=params.logits).entropy()

  def sample(params: CategoricalParams, key: networks_lib.PRNGKey):
    return tfd.Categorical(logits=params.logits).sample(seed=key)

  def sample_eval(params: CategoricalParams, key: networks_lib.PRNGKey):
    del key
    return tfd.Categorical(logits=params.logits).mode()

  return ppo.PPONetworks(
      network=network,
      log_prob=log_prob,
      entropy=entropy,
      sample=sample,
      sample_eval=sample_eval)

def init_default_dram_network(
    agent_type: str,
    agent_spec: specs.EnvironmentSpec) -> ma_types.Networks:
  """Returns default networks for multigrid environment."""
  if agent_type == factories.DefaultSupportedAgent.PPO:
    return make_dram_ppo_networks(agent_spec)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')