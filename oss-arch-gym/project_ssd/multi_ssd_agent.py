import gym
from gym import spaces
import numpy as np
from absl import flags
import os
import json
import subprocess 
from pathlib import Path
import time

from concurrent import futures
import grpc
import portpicker
import sys
import os

from absl import app
from absl import logging
import envlogger
import numpy as np
import pandas as pd
from vizier._src.algorithms.designers.random import RandomDesigner
from vizier._src.algorithms.designers.emukit import EmukitDesigner

# from arch_gym.envs import customenv_wrapper
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

from typing import Any, Dict, List, Optional

from acme import specs
from acme import types
from acme import wrappers
import dm_env
import gym
from gym import spaces
import numpy as np
import tree


# Define global variables
PERF_LOG_PATH = './profiling_tools/results/performance_log.csv'
CONFIG_PATH = Path("./tuning_tools/config.json")
LINUX_TUNE_CONFIG_PATH = './tune_config_wprof.sh'
POWERSHELL_LINUX_TUNE_CONFIG_PATH = r"C:\Users\Administrator\Desktop\wsl\arch-gym\main.ps1"
'''
ALGORITHM = "bo"
TRAJECT_DIR = 'bo_search_trajectories'
LOG_SUBDIR = 'bo_search_log'
'''
ALGORITHM = "random"
TRAJECT_DIR = 'random_search_trajectories'
LOG_SUBDIR = 'random_search_log'

WORKLOAD = 'Webserver'
NUM_STEPS = 1000
NUM_EPISODES = 2

USE_ENVLOGGER = False
LOG_DIR = './log'
REWARD_FORMULATION = 'throughput'
SEED = 110

class CustomEnv(gym.Env):
    def __init__(self, max_steps=10):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Dict({
          "bandwidth": spaces.Box(low=0, high=np.inf, dtype=np.float32),
          "throughput": spaces.Box(low=0, high=np.inf, dtype=np.float32), 
        })
        
        self.action_space = spaces.Dict({
          "qw": spaces.Discrete(6),
          "ab": spaces.Box(low=0, high=7, dtype=int),
          "wa": spaces.Discrete(2),
          "ic": spaces.Box(low=0, high=5, dtype=int),
          "ps": spaces.Discrete(5),
        })
         

        self.max_steps = max_steps
        self.counter = 0
        self.bandwidth = 0
        self.throughput = 0
        self.initial_state = np.array([self.bandwidth, self.throughput])
        self.observation = None
        self.done = False
        self.ideal = np.array([6000.0], dtype=np.float32) #ideal values for throughput

    def reset(self):
        return self.initial_state

    # Input : action; Output : observation & reward &
    def step(self, action):
        '''
        qd = action['qd']
        mnpd = action['mnpd']
        mc = action['mc']
        pc = action['pc'] 
        dpo = action['dpo']
        irp = action['irp']
        dwc = action['dwc']
        smartpath_ac = action['smartpath_ac'] 

        action = np.array([qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac])
        '''
        if (self.counter == self.max_steps):
            self.done = True
            print("Maximum steps reached")
            self.reset()
        else:
            self.counter += 1
        
        # Compute the new state based on the action (random formulae for now)
        # TODO: Replace with actual reward function
        df = pd.read_csv(PERF_LOG_PATH) 
        df = df[1:]
        df = df.apply(lambda col: col.astype('float32') if col.name != 'Device' else col, axis=0)
        df["iops"] = df["r/s"] + df["w/s"]
        df["bandwidth"] =df["rkB/s"] + df["wkB/s"]
        self.throughput = df["iops"].mean()
        self.bandwidth = df["bandwidth"].mean()
        print("Throughput: ", self.throughput)
        print("Bandwidth(kb): ", self.bandwidth)

        observation = np.array([self.bandwidth, self.throughput], dtype=np.float32)
        self.observation = observation
        objective = np.array([self.throughput], dtype=np.float32)
        reward = objective - self.ideal

        return observation, reward, self.done, {}

    def render(self, mode='human'):
        print (f'Energy: {self.energy}, Area: {self.area}, Latency: {self.latency}')

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

"""Wraps an OpenAI Gym environment to be used as a dm_env environment."""


class CustomEnvWrapper(dm_env.Environment):
  """Environment wrapper for OpenAI Gym environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self, environment: gym.Env):

    self._environment = environment
    self._reset_next_step = True
    self._last_info = None
    # self.helper = helpers()

    # Convert action and observation specs.
    obs_space = self._environment.observation_space
    act_space = self._environment.action_space
    self._observation_spec = _convert_to_spec(obs_space, name='observation')
    self._action_spec = _convert_to_spec(act_space, name='action')
    

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    observation = self._environment.reset()
    # Reset the diagnostic information.
    self._last_info = None
    a = dm_env.restart(observation)

    return a

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(action)
    self._reset_next_step = done
    self._last_info = info

    # Convert the type of the reward based on the spec, respecting the scalar or
    # array property.
    reward = tree.map_structure(
        lambda x, t: (  # pylint: disable=g-long-lambda
            t.dtype.type(x)
            if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
        reward,
        self.reward_spec())

    if done:
      truncated = info.get('TimeLimit.truncated', False)
      if truncated:
        return dm_env.truncation(reward, observation)
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._action_spec

  def reward_spec(self):
    return specs.Array(shape=(), dtype=float, name='reward')

  def get_info(self) -> Optional[Dict[str, Any]]:
    """Returns the last info returned from env.step(action).
    Returns:
      info: dictionary of diagnostic information from the last environment step
    """
    return self._last_info

  @property
  def environment(self) -> gym.Env:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    if name.startswith('__'):
      raise AttributeError(
          "attempted to get missing private attribute '{}'".format(name))
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def _convert_to_spec(space: gym.Space,
                     name: Optional[str] = None) -> types.NestedSpec:
  """
  Box, MultiBinary and MultiDiscrete -> BoundedArray
  Discrete -> DiscreteArray
  Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.
  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).
  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec - 1,
        name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(_convert_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: _convert_to_spec(value, key)
        for key, value in space.spaces.items()
    }

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))

def make_custom_env(seed: int = 12234,
                     max_steps: int = 100,
                     reward_formulation: str = 'throughput',
                     ) -> dm_env.Environment:
  """Returns DRAMSys environment."""
  environment = CustomEnvWrapper(CustomEnv(max_steps=max_steps))
  environment = wrappers.SinglePrecisionWrapper(environment)
#   if(arch_gym_configs.rl_agent):
#     environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  return environment

def log_fitness_to_csv(filename, fitness_dict):
    """Logs fitness history to a CSV file.

    Args:
        filename (str): Path to the CSV file.
        fitness_dict (dict): Dictionary containing the fitness history.
    """
    df = pd.DataFrame([fitness_dict['reward']])
    csvfile = os.path.join(filename, "fitness.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

    # Append to CSV
    df = pd.DataFrame([fitness_dict])
    csvfile = os.path.join(filename, "trajectory.csv")
    df.to_csv(csvfile, index=False, header=False, mode='a')

def wrap_in_envlogger(env, envlogger_dir):
    """Wraps the environment in envlogger.

    Args:
        env (gym.Env): Gym environment.
        envlogger_dir (str): Path to the directory where the data will be logged.
    """
    metadata = {
        'agent_type': 'RandomSearch',
        'num_steps': NUM_STEPS,
        'env_type': type(env).__name__,
    }
    if USE_ENVLOGGER:
        logging.info('Wrapping environment with EnvironmentLogger...')
        env = envlogger.EnvLogger(env,
                                  data_directory=envlogger_dir,
                                  max_episodes_per_file=1000,
                                  metadata=metadata)
        logging.info('Done wrapping environment with EnvironmentLogger.')
        return env
    else:
        return env


# call bash script to tune config
# use subprocess to run bash script
def tune_config_linux(config):
  with CONFIG_PATH.open('w') as file:
      json.dump(config, file)

  process = subprocess.Popen(["sudo", "bash", LINUX_TUNE_CONFIG_PATH], stdout=subprocess.PIPE)
  stdout, stderr = process.communicate()

  return_code = process.returncode
  print("stdout:", stdout.decode('utf-8'))
  if stderr:
    print("stderr:", stderr.decode('utf-8'))
  print("return code:", return_code)

def tune_config_windows(config):
  with CONFIG_PATH.open('w') as file:
      json.dump(config, file)

  result = subprocess.run(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", POWERSHELL_LINUX_TUNE_CONFIG_PATH], capture_output=True, text=True)

  if result.returncode == 0:
      print("PowerShell 腳本成功執行，輸出為：")
      print(result.stdout)
  else:
      print("PowerShell 腳本執行失敗，錯誤輸出為：")
      print(result.stderr)

def main():
    """Trains the custom environment using random actions for a given number of steps and episodes
    """
    
    env = make_custom_env(max_steps=NUM_STEPS)
    fitness_hist = {}
    problem = vz.ProblemStatement()
    arb_list = [i for i in range(0, 8)]
    arw_list = [i + 1 for i in range(6)]
    problem.search_space.select_root().add_discrete_param(name='QueueWeight', feasible_values=arw_list)
    problem.search_space.select_root().add_discrete_param(name='ArbriterBurst', feasible_values=arb_list)
    problem.search_space.select_root().add_discrete_param(name='WrAtomicity', feasible_values=[0, 1])
    problem.search_space.select_root().add_discrete_param(name='IrCoalescing', feasible_values=[0, 1, 2, 3, 4, 5])
    problem.search_space.select_root().add_categorical_param(name='PowerState', feasible_values=[0, 1, 2, 3, 4])

    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))




    study_config = vz.StudyConfig.from_problem(problem)
    random_designer = RandomDesigner(problem.search_space, seed = SEED)
    bo_designer = EmukitDesigner(problem)

    port = portpicker.pick_unused_port()
    address = f'localhost:{port}'

    # Setup server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    # Setup Vizier Service.
    servicer = vizier_server.VizierService()
    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(servicer, server)
    server.add_secure_port(address, grpc.local_server_credentials())

    # Start the server.
    server.start()

    clients.environment_variables.service_endpoint = address  # Server address.
    study = clients.Study.from_study_config(
        study_config, owner='owner', study_id='example_study_id')

     # experiment name
    exp_name = "_num_steps_" + str(NUM_STEPS) + "_num_episodes_" + str(NUM_EPISODES)

    # append logs to base path
    log_path = os.path.join(LOG_DIR, LOG_SUBDIR, REWARD_FORMULATION, exp_name)
    

    # get the current working directory and append the exp name
    global TRAJECT_DIR
    TRAJECT_DIR = os.path.join(LOG_DIR, TRAJECT_DIR, REWARD_FORMULATION, exp_name)
    
    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if USE_ENVLOGGER:
        if not os.path.exists(TRAJECT_DIR):
            os.makedirs(TRAJECT_DIR)
    env = wrap_in_envlogger(env, TRAJECT_DIR)

    # """
    # This loop runs for num_steps * num_episodes iterations.
    # """
    env.reset()

    count = 0
    # Designer TODO
    if ALGORITHM == "bo":
      suggestions = bo_designer.suggest(count=NUM_STEPS)
    elif ALGORITHM == "random":
      suggestions = random_designer.suggest(count=NUM_STEPS)
    else:
      raise ValueError("Algorithm not supported")

    start_time = time.time()
    for suggestion in suggestions:
        count += 1
        # Get action
        qw = str(suggestion.parameters['QueueWeight'])
        ab = str(suggestion.parameters['ArbriterBurst'])
        wa = str(suggestion.parameters['WrAtomicity'])
        ic = str(suggestion.parameters['IrCoalescing'])
        ps = str(suggestion.parameters["PowerState"])
        action = {
          "qw" : float(qw),
          "ab" : float(ab),
          "wa" : float(wa),
          "ic" : float(ic),
          "ps" : float(ps),
        }
        
        print("Suggested Parameters")
        config = {
          "configuration" : [int(qw), int(ab), int(ps), -1, int(ic), -1, int(wa)]
        }
        print(config["configuration"])
        # Tune config
        tune_config_linux(config)

        # Iterate with environment
        done, reward, info, obs = (env.step(action))
        
        # Save trajectory and reward to log_path
        fitness_hist['reward'] = reward
        fitness_hist['action'] = action
        fitness_hist['obs'] = obs
        if count == NUM_STEPS:
            done = True

        log_fitness_to_csv(log_path, fitness_hist)
        print("Observation: ",obs)
        
        # Update agent
        final_measurement = vz.Measurement({'Reward': reward})
        suggestion = suggestion.to_trial()
        suggestion.complete(final_measurement)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

if __name__ == '__main__':
  main()
