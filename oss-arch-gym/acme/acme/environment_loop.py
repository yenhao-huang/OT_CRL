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

"""A simple agent-environment training loop."""

import operator
import time
from typing import Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import signals

import dm_env
from dm_env import specs
import numpy as np
import gym
import tree

COUNT = 0
execution_time_all = []

class EnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. Agent is updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.

  A list of 'Observer' instances can be specified to generate additional metrics
  to be logged by the logger. They have access to the 'Environment' instance,
  the current timestep datastruct and the current action.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'environment_loop',
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      agent_type: str = "multiagent",
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self.label = label
    self._logger = logger or loggers.make_default_logger(
        label, steps_key=self._counter.get_steps_key())
    self._should_update = should_update
    self._observers = observers
    self._agent_type = agent_type

  def single_agent_sample_random_action(self):
      action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
      action = action_space.sample()
      return action

  def single_agent_explore(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """

    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    # Make the first observation.
    timestep = self._environment.reset()
    self._actor.observe_first(timestep)
      
    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      #print(f"[obs:{timestep.observation}]")
      action = self.single_agent_sample_random_action()
      timestep = self._environment.step(action)
      # Have the agent observe the timestep and let the actor update itself.
      # self._observers default: None
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)
        
      if self._should_update:
        self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd, episode_return, timestep.reward)
      
      # [JONY]
      latency, energy, err_rate = self._environment.record_perfmetric()
    
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    if self._environment.rl_form == "macme" or self._environment.rl_form == "macme_continuous":
      result = {
        'episode_length': episode_steps,
        'episode_return': episode_return[0],
        'steps_per_second': steps_per_second,
      }
    else:
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'steps_per_second': steps_per_second,
      }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())

    # [JONY]
    result.update({
        'energy': energy,
        'latency': latency,
        'err_rate': err_rate,
    })
    pagepolicy = {"open" : 0, "closed" : 1}
    addressmapping = {"bank_row_col" : 0, "row_bank_col" : 1, "permutation" : 2, "row_bank_col2" : 3, "row_bank_col3" : 4}
    action_dict = self._environment.action_decoded
    action_dict["pagepolicy"] = pagepolicy[action_dict["pagepolicy"]]
    action_dict["addressmapping"] = addressmapping[action_dict["addressmapping"]]
    result.update(action_dict)

    return result

  def sample_random_action(self):
    action_space = [
        gym.spaces.Discrete(2),  # page policy agent
        gym.spaces.Discrete(5),  # address mapping agent
        gym.spaces.Discrete(5),  # rcd agent
        gym.spaces.Discrete(5),  # rp agent
        gym.spaces.Discrete(21), # ras agent
        gym.spaces.Discrete(4),  # rrd agent
        gym.spaces.Discrete(10), # refreshinterval agent 
    ]
    
    action = {}
    for i, space in enumerate(action_space):
        action[str(i)] = np.array(space.sample(), dtype="int32")
    
    return action

  def explore(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    # Make the first observation.
    timestep = self._environment.reset()
    self._actor.observe_first(timestep)
      
    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      #print(f"[obs:{timestep.observation}]")
      action = self.sample_random_action()
      timestep = self._environment.step(action)
      # Have the agent observe the timestep and let the actor update itself.
      # self._observers default: None
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)
        
      if self._should_update:
        self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd, episode_return, timestep.reward)
      
      # [JONY]
      latency, energy, err_rate = self._environment.record_perfmetric()
    
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    if self._environment.rl_form == "macme" or self._environment.rl_form == "macme_continuous":
      result = {
        'episode_length': episode_steps,
        'episode_return': episode_return[0],
        'steps_per_second': steps_per_second,
      }
    else:
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'steps_per_second': steps_per_second,
      }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())

    # [JONY]
    result.update({
        'energy': energy,
        'latency': latency,
        'err_rate': err_rate,
    })
    pagepolicy = {"open" : 0, "closed" : 1}
    addressmapping = {"bank_row_col" : 0, "row_bank_col" : 1, "permutation" : 2, "row_bank_col2" : 3, "row_bank_col3" : 4}
    action_dict = self._environment.action_decoded
    action_dict["pagepolicy"] = pagepolicy[action_dict["pagepolicy"]]
    action_dict["addressmapping"] = addressmapping[action_dict["addressmapping"]]
    result.update(action_dict)

    return result


  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    episode_steps = 0
    start_time = time.time()

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        self._environment.reward_spec())
    # Make the first observation.
    timestep = self._environment.reset()
    self._actor.observe_first(timestep)

    # self._observers default: None
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      observer.observe_first(self._environment, timestep)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      #print(f"[obs:{timestep.observation}]")
      '''
      #measure inference time
      import time
      global COUNT, execution_time_all
      start_time = time.time()
      action = self._actor.select_action(timestep.observation)
      COUNT+=1
      end_time = time.time()
      execution_time_all.append(end_time - start_time)
      if COUNT >= 5:
        for execution_time in execution_time_all:
          print(f"Execution time: {execution_time:.6f} seconds")
        exit(0)
      '''
      action = self._actor.select_action(timestep.observation)
      timestep = self._environment.step(action)
      # Have the agent observe the timestep and let the actor update itself.

      # self._observers default: None
      self._actor.observe(action, next_timestep=timestep)
      for observer in self._observers:
        # One environment step was completed. Observe the current state of the
        # environment, the current timestep and the action.
        observer.observe(self._environment, timestep, action)

      if self._should_update:
        self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd, episode_return, timestep.reward)
      
      # [JONY]
      latency, energy, err_rate = self._environment.record_perfmetric()
    
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    if self._environment.rl_form == "macme" or self._environment.rl_form == "macme_continuous":
      result = {
        'episode_length': episode_steps,
        'episode_return': episode_return[0],
        'steps_per_second': steps_per_second,
      }
    else:
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'steps_per_second': steps_per_second,
      }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())

    # [JONY]
    result.update({
        'energy': energy,
        'latency': latency,
        'err_rate': err_rate,
    })
    pagepolicy = {"open" : 0, "closed" : 1}
    addressmapping = {"bank_row_col" : 0, "row_bank_col" : 1, "permutation" : 2, "row_bank_col2" : 3, "row_bank_col3" : 4}
    action_dict = self._environment.action_decoded
    action_dict["pagepolicy"] = pagepolicy[action_dict["pagepolicy"]]
    action_dict["addressmapping"] = addressmapping[action_dict["addressmapping"]]
    result.update(action_dict)
    return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Returns:
      Actual number of steps the loop executed.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))
    if self._agent_type == "multiagent":
      episode_count, step_count = 0, 0
      with signals.runtime_terminator():
        while not should_terminate(episode_count, step_count):
          if step_count < 2 and self.label != 'eval_loop':
            result = self.explore()
          else:
            result = self.run_episode()
          episode_count += 1
          step_count += result['episode_length']
          # Log the given episode results.
          self._logger.write(result)
          print(f'Episode {episode_count} finished after {result["episode_length"]} steps.')
    else:
      episode_count, step_count = 0, 0
      with signals.runtime_terminator():
        while not should_terminate(episode_count, step_count):
          if step_count < 50 and self.label != 'eval_loop':
            result = self.single_agent_explore()
          else:
            result = self.run_episode()
          episode_count += 1
          step_count += result['episode_length']
          # Log the given episode results.
          self._logger.write(result)
          print(f'Episode {episode_count} finished after {result["episode_length"]} steps.')      

    return step_count


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
