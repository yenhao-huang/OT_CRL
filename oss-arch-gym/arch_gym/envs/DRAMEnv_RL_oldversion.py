import os
import sys
import csv

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)

os.sys.path.insert(0, settings_dir_path + '/../../')

from configs import arch_gym_configs
from pathlib import Path
import gym
from gym.utils import seeding
from envHelpers import helpers

from loggers import write_csv
import numpy as np

# ToDo: Have a configuration for Arch-Gym to manipulate this methods

import sys

import subprocess
import time
import re
import numpy
import pickle

import random

LOG_DIR = "logs/Details"
N_CONFIG = 7

class DRAMEnv_Base(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()
        
    '''
    # DRAMSYS
    def get_observation(self, outstream):
        
        #converts the std out from DRAMSys to observation of energy, power, latency
        #[Energy (PJ), Power (mW), Latency (ns)]

        obs = []
        
        keywords = ["Total Energy", "Average Power", "Total Time"]

        energy = re.findall(keywords[0],outstream)
        all_lines = outstream.splitlines()
        for each_idx in range(len(all_lines)):
            
            if keywords[0] in all_lines[each_idx]:
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)
            if keywords[1] in all_lines[each_idx]:
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e3)
            if keywords[2] in all_lines[each_idx]:
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e6)

        obs = np.asarray(obs)
        print('[Environment] Observation:', obs)
        
        if(len(obs)==0):
             print(outstream)
        return obs
    '''
    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}//{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_obs_reward(self, action_decoded):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        obs = np.append(obs, error_rate)

        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        self._observation = obs
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        '''
        base_rcd, base_rp, base_ras, base_rrd, base_refi, refi_range = 12, 12, 19, 3, 4680, 46800
        if self.workload == "xz":
            action_list = [1, 1, 12-base_rcd, 15-base_rp, 38-base_ras, 6-base_rrd, (332280-base_refi)/refi_range]
        elif self.workload == "namd":
            action_list = [0, 1, 12-base_rcd, 13-base_rp, 32-base_ras, 5-base_rrd, (332280-base_refi)/refi_range]
        elif self.workload == "sphinx3":
            action_list = [0, 1, 12-base_rcd, 13-base_rp, 32-base_ras, 5-base_rrd, (332280-base_refi)/refi_range]
        else:
            raise "no target default config"
        '''
        #action_list = [0] * 7
        action_list = [0,2,0,3,15,0,5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[0][:3]
        else:
            return self._observation[:3]
        
    def reset(self):
        '''
        #print("Reseting Environment!")
        self.steps = 0
        self.cur_config_idx = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.observation_space[0].sample()] * self.num_agents
        else:
            # return zeros of shape of observation space
            return np.zeros(self.observation_space.shape)
            #return self.observation_space.sample()
        '''

        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:lat/energy/action1-7    
class DRAMEnv_V2(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(9,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self._observation = None
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_env_v2/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_env_v2/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()
        
    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        return 1e-4
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[0][:3]
        else:
            return self._observation[:3]
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}//{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_obs_reward(self, action_decoded, action):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        action = np.array(action, dtype=np.float)
        obs = np.append(obs, action)

        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        return obs, reward

    def step(self, action_enconded):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_enconded)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded, action_enconded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        self._observation = obs

        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        action_list = [0, 2, 0, 3, 15, 0, 5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded, action_list)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs

    def reset(self):
        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:lat/energy/err_rate/action1-7
class DRAMEnv_V3(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(10,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self._observation = None
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_env_v3/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_env_v3/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()
        
    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        return 1e-4
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[0][:3]
        else:
            return self._observation[:3]
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}//{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_obs_reward(self, action_decoded, action):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        action = np.array(action, dtype=np.float)
        obs = np.append(obs, error_rate)
        obs = np.append(obs, action)

        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        return obs, reward

    def step(self, action_enconded):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_enconded)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded, action_enconded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        self._observation = obs

        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        action_list = [0, 2, 0, 3, 15, 0, 5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded, action_list)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs

    def reset(self):
        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:lat/energy
class DRAMEnv_V4(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(2,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.error_rate = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v4/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v4/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()
        
    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}//{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_obs_reward(self, action_decoded):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        self.error_rate = error_rate
        
        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        self._observation = obs
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        #action_list = [0] * 7
        action_list = [0, 2, 0, 3, 15, 0, 5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return np.append(self._observation[0], self.error_rate)
        else:
            return self._observation
        
    def reset(self):
        '''
        #print("Reseting Environment!")
        self.steps = 0
        self.cur_config_idx = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.observation_space[0].sample()] * self.num_agents
        else:
            # return zeros of shape of observation space
            return np.zeros(self.observation_space.shape)
            #return self.observation_space.sample()
        '''

        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:lat/energy/action1-7 ; same as v2
class DRAMEnv_V5(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(9,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.error_rate = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v5/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v5/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()
        
    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}//{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_obs_reward(self, action_decoded, action):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        self.error_rate = error_rate
        obs = np.append(obs, action)
        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        return obs, reward

    def step(self, action_encoded):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_encoded)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded, action_encoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        self._observation = obs
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        #action_list = [0] * 7
        action_list = [0,2,0,3,15,0,5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded, action_list)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return np.append(self._observation[0][:2], self.error_rate)
        else:
            return self._observation[:2]
        
    def reset(self):
        '''
        #print("Reseting Environment!")
        self.steps = 0
        self.cur_config_idx = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.observation_space[0].sample()] * self.num_agents
        else:
            # return zeros of shape of observation space
            return np.zeros(self.observation_space.shape)
            #return self.observation_space.sample()
        '''

        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

## Base + err_reward : 1e6 -> 0
class DRAMEnv_V6(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 expertdata_path = None,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v6/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v6/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.scaler, self.is_norm = self.get_scaler(expertdata_path)
        self.default_obs = self.get_default_obs()
        self.reset()
        
    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open(f"/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target_{self.thread_idx}.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")

        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}//{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_obs_reward(self, action_decoded):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        obs = np.append(obs, error_rate)

        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [0.0] * self.num_agents
            else:
                reward = 0.0
        return obs, reward

    def get_scaler(self, expertdata_path):
        if expertdata_path != None and expertdata_path.exists():
            is_norm = True
            with open(expertdata_path, 'rb') as f:
                _, _, metadata = pickle.load(f)
            scaler = metadata["scaler"]
        else:
            is_norm = False
            scaler = None
        return scaler, is_norm

    def normalize(self, obs):
        obs = self.scaler.transform([obs])
        obs = obs[0]
        return obs
        
    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)

            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        if self.is_norm:
            obs_denrom = obs
            obs = self.normalize(obs)

        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        if self.is_norm:
            self._observation = [obs_denrom.copy()] * self.num_agents
        else:
            self._observation = obs
        return obs, reward, done, {}

    def get_default_obs(self):
        action_list = [0,2,0,3,15,0,5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)
        
        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)

        '''
        # 立即修改
        obs, reward = self.get_obs_reward(action_decoded)
        self.buffer.update({bufferid:(obs, reward)})
        with open(self.load_buf_path, 'wb') as f:
            pickle.dump(self.buffer, f)
        '''

        if self.is_norm:
            obs = self.normalize(obs)

        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[0][:3]
        else:
            return self._observation[:3]
        
    def reset(self):
        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:lat/energy/err_rate/best_action1-7
class DRAMEnv_V7(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(10,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v7/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v7/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()

    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-5
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_workload_mapping(self):
        mapping = {'hmmer': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
        5.0000e+00, 3.3228e+05]),
        'xz': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
                5.0000e+00, 3.3228e+05]),
        'lbm': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.9000e+01,
                6.0000e+00, 3.7908e+05]),
        'swaptions': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.6000e+01, 3.9000e+01,
                4.0000e+00, 4.2588e+05]),
        'ferret': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.6000e+01,
                5.0000e+00, 2.3868e+05]),
        'zeusmp': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                3.0000e+00, 3.3228e+05]),
        'namd': np.array([0.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                5.0000e+00, 3.3228e+05]),
        'freqmine': np.array([0.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 2.0000e+01,
                5.0000e+00, 3.3228e+05])}
        return mapping[self.workload]

    def get_obs_reward(self, action_decoded):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_decoded)
        obs = np.append(obs, error_rate)
        obs = np.concatenate([obs, self.get_workload_mapping()])
        
        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        self._observation = obs
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        #action_list = [0] * 7
        action_list = [0,2,0,3,15,0,5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[0][:3]
        else:
            return self._observation[:3]
        
    def reset(self):

        self.steps = 0
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs

        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:best_action_i for each agent i
class DRAMEnv_V8(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(1,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v8/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v8/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()

    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_workload_mapping(self):
        mapping = {
            'hmmer': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
        5.0000e+00, 3.3228e+05]),
        'xz': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
                5.0000e+00, 3.3228e+05]),
        'lbm': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.9000e+01,
                6.0000e+00, 3.7908e+05]),
        'swaptions': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.6000e+01, 3.9000e+01,
                4.0000e+00, 4.2588e+05]),
        'ferret': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.6000e+01,
                5.0000e+00, 2.3868e+05]),
        'zeusmp': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                3.0000e+00, 3.3228e+05]),
        'namd': np.array([0.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                5.0000e+00, 3.3228e+05]),
        'freqmine': np.array([0.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 2.0000e+01,
                5.0000e+00, 3.3228e+05])
        }
        return mapping[self.workload]

    def get_obs_reward(self, action_decoded):
        perf = self.runDRAMEnv()

        error_rate = self.helpers.get_err_rate(action_decoded)
        perf = np.append(perf, error_rate)
        best_action = self.get_workload_mapping()
        obs = best_action.reshape(-1, 1).astype('float')
        obs = [np.array(o) for o in obs]
        if error_rate < self.err_limit:
            reward = self.calculate_reward(perf)
            
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        
        self._observation = perf
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        #if self.rl_form == "macme" or self.rl_form == "macme_continuous":
        #    obs = [obs.copy()] * self.num_agents

        print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        #action_list = [0] * 7
        action_list = [0, 2, 0, 3, 15, 0, 5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[:3]
        else:
            return self._observation[:3]
        
    def reset(self):

        self.steps = 0
        return self.default_obs
        '''
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs
        '''
        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded


# state:best_action_i for each agent i
class DRAMEnv_V9(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(4,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v8/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v8/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()

    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_workload_mapping(self):
        mapping = {
            'hmmer': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
        5.0000e+00, 3.3228e+05]),
        'xz': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
                5.0000e+00, 3.3228e+05]),
        'lbm': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.9000e+01,
                6.0000e+00, 3.7908e+05]),
        'swaptions': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.6000e+01, 3.9000e+01,
                4.0000e+00, 4.2588e+05]),
        'ferret': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.6000e+01,
                5.0000e+00, 2.3868e+05]),
        'zeusmp': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                3.0000e+00, 3.3228e+05]),
        'namd': np.array([0.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                5.0000e+00, 3.3228e+05]),
        'freqmine': np.array([0.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 2.0000e+01,
                5.0000e+00, 3.3228e+05])
        }
        return mapping[self.workload]

    def get_obs_reward(self, action_decoded):
        perf = self.runDRAMEnv()

        error_rate = self.helpers.get_err_rate(action_decoded)
        perf = np.append(perf, error_rate)
        best_action = self.get_workload_mapping()
        obs = [perf] * self.num_agents
        act = best_action.reshape(-1, 1).astype('float')
        obs = [np.concatenate([o, a]) for o, a in zip(obs, act)]

        if error_rate < self.err_limit:
            reward = self.calculate_reward(perf)
            
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        
        self._observation = perf
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        #if self.rl_form == "macme" or self.rl_form == "macme_continuous":
        #    obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        #action_list = [0] * 7
        action_list = [0, 2, 0, 3, 15, 0, 5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[:3]
        else:
            return self._observation[:3]
        
    def reset(self):

        self.steps = 0
        return self.default_obs
        '''
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs
        '''
        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# state:best_action_i for each agent i
class DRAMEnv_V10(gym.Env):
    def __init__(self,
                 workload,
                 rl_form: str = 'tdm',
                 rl_algo: str = 'ppo',
                 max_steps: int = 5,
                 num_agents: int = 1,
                 reward_formulation: str = 'latency',
                 reward_scaling: str = 'false',
                 is_update_buffer: bool = False,
                 ):

        if(rl_form == 'tdm'):
            # observation space  for TDM
            # Simulator output: [Energy (PJ), Power (mW), Latency (ns)]
            # TDM Action = [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]
            # Observation to agent at each time step
            # [Energy (PJ), Power (mW), Latency (ns), param1, param2, param3, param4, param5, param6, param7, param8, param9, param10]

            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
            # Agent is Time division multiplexed. Hence it will take only one of the many discrete actions
            self.action_space = gym.spaces.Discrete(8)
            if (rl_algo == 'sac'):
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        elif(rl_form == 'sa'):
            self.observation_space = gym.spaces.Box(low=0, high=1e3, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_CONFIG,))
        elif (rl_form == 'macme'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            
            self.action_space = [
                # page policy agent
                gym.spaces.Discrete(2),
                
                # address mapping agent
                gym.spaces.Discrete(5),
                
                # rcd agent
                gym.spaces.Discrete(5),
                
                # rp agent
                gym.spaces.Discrete(5),
                
                # ras agent
                gym.spaces.Discrete(21),
                
                # rrd agent
                gym.spaces.Discrete(4),
                
                # refreshinterval agent 
                gym.spaces.Discrete(10),
            ]
        elif (rl_form == 'macme_continuous'):
            self.observation_space = [gym.spaces.Box(low=0, high=1e3, shape=(3,))] * num_agents
            self.action_space = [
                # page policy agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # address mapping agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # rcd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rp agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # ras agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),

                # rrd agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                
                # refreshinterval agent
                gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            ]
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(10,))
            
        self.rl_form = rl_form
        self.num_agents = num_agents
        self._observation = None
        self.binary_name = arch_gym_configs.binary_name
        self.exe_path = arch_gym_configs.exe_path
        self.sim_config = arch_gym_configs.sim_config
        self.sim_config_dir = arch_gym_configs.sim_config_dir
        self.experiment_name = arch_gym_configs.experiment_name
        self.logdir = arch_gym_configs.logdir
        self.reward_form = reward_formulation
        self.reward_scaling = reward_scaling
        self.total_step = 0
        self.thread_idx = self.read_thread_idx()
        self.target_latency, self.target_energy = self.read_target()
        
        print("[DEBUG][Reward Scaling]", self.reward_scaling)
        self.max_steps = max_steps
        self.steps = 0
        self.max_episode_len = 10
        self.episode = 0
        self.reward_cap = 1e3
        self.helpers = helpers()
        self.algorithm = "GA"
        self.goal_latency = 2e7
        self.prev_latency = 1e9
        self.prev_power = 0
        self.workload = workload
        self.action_decoded = None
        self.err_limit = self.read_upper_err()
        self.is_update_buffer = is_update_buffer
        self.load_buf_path = f'buffer_v8/merge_for_rl/{self.workload}.pkl'
        self.save_buf_path = f'buffer_v8/rl/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.default_obs = self.get_default_obs()
        self.reset()

    def read_thread_idx(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/thread.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            int_value = int(lines[0])
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return lines[0].strip()
    
    def read_target(self):
        with open("/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec_ramulator/target.txt", "r") as f:
            lines = f.readlines()

        try:
            # Attempt to convert lines[0] to an integer
            target_latency, target_energy = lines[1].split(",")
        except ValueError:
            # This block executes if there is a ValueError (e.g., converting non-numeric string to int)
            raise ValueError("THREADIDX ERROR: Could not convert lines[0] to integer.")
        
        return float(target_latency), float(target_energy)
    
    def read_upper_err(self):
        #TODO
        return 1e-4
    
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}/{self.thread_idx}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
        

        return np.array([latency_us, energy_uj])
        
    def calculate_reward(self, obs):
        latency_normalize = self.target_latency / abs(obs[0]-self.target_latency)
        energy_normalize = self.target_energy / abs(obs[1]-self.target_energy)
        reward = latency_normalize * energy_normalize
        
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            reward = [np.copy(reward)] * self.num_agents

        return reward

    def read_config_dir(self):
        sim_config_arr = []
        for f in Path(self.sim_config_dir).iterdir():
            sim_config_arr.append(f)
        return sim_config_arr
        

    def runDRAMEnv(self):
        '''
        Method to launch the DRAM executables given an action
        '''
        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4_multithread/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
                
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main_multithread.py", "--workload", self.workload, "--threadidx", self.thread_idx],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        obs = self.get_observation()
        return obs

    def is_buffered(self, bufferid):
        if bufferid in self.buffer.keys():
              return True
        else:
            return False

    def load_buf(self):
        if not os.path.exists(self.load_buf_path):
            data = {}
        else:
            with open(self.load_buf_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def update_buf(self, bufferid, obs, reward):
        self.buffer.update({bufferid:(obs, reward)})
        self.save_buffer()

    def save_buffer(self):
        if (self.total_step+1) % 100 == 0:
            with open(self.save_buf_path, 'wb') as f:
                pickle.dump(self.buffer, f)

    def get_from_buf(self, bufferid):
        return self.buffer[bufferid]

    def get_workload_mapping(self):
        mapping = {
            'hmmer': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
        5.0000e+00, 3.3228e+05]),
        'xz': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.3000e+01, 3.9000e+01,
                5.0000e+00, 3.3228e+05]),
        'lbm': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.9000e+01,
                6.0000e+00, 3.7908e+05]),
        'swaptions': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.6000e+01, 3.9000e+01,
                4.0000e+00, 4.2588e+05]),
        'ferret': np.array([1.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 3.6000e+01,
                5.0000e+00, 2.3868e+05]),
        'zeusmp': np.array([1.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                3.0000e+00, 3.3228e+05]),
        'namd': np.array([0.0000e+00, 1.0000e+00, 1.2000e+01, 1.2000e+01, 3.8000e+01,
                5.0000e+00, 3.3228e+05]),
        'freqmine': np.array([0.0000e+00, 2.0000e+00, 1.2000e+01, 1.2000e+01, 2.0000e+01,
                5.0000e+00, 3.3228e+05])
        }
        return mapping[self.workload]

    def get_obs_reward(self, action_decoded):
        perf = self.runDRAMEnv()

        error_rate = self.helpers.get_err_rate(action_decoded)
        perf = np.append(perf, error_rate)
        best_action = self.get_workload_mapping()
        obs = [perf[:2]] * self.num_agents
        act = best_action.reshape(-1, 1).astype('float')
        obs = [np.concatenate([o, a]) for o, a in zip(obs, act)]

        if error_rate < self.err_limit:
            reward = self.calculate_reward(perf)
            
        else:
            if self.rl_form == "macme" or self.rl_form == "macme_continuous":
                reward = [-1e6] * self.num_agents
            else:
                reward = -1e6
        
        self._observation = perf
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        self.steps += 1
        done = False
        
        status, action_decoded = self.actionToConfigs(action_dict)
        if not status:
            print("Error in writing configs")

        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)


        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
                
        if(self.steps == self.max_steps):
            self.default_obs = obs
            done = True
            print("Maximum steps per episodes reached!")
            self.reset()
            self.episode +=1
        
        #if self.rl_form == "macme" or self.rl_form == "macme_continuous":
        #    obs = [obs.copy()] * self.num_agents

        #print('[Environment] Observation:', obs)
        print("Episode:", self.episode, " Rewards:", reward)
        self.total_step += 1
        return obs, reward, done, {}

    def get_default_obs(self):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''

        #action_list = [0] * 7
        action_list = [0, 2, 0, 3, 15, 0, 5]
        status, action_decoded = self.actionToConfigs(action_list)
        if not status:
            print("Error in writing configs")


        action_all = [str(act) for act in action_decoded.values()]
        bufferid = '_'.join(action_all)

        if self.is_buffered(bufferid) and self.rl_form == "macme":
            obs, reward = self.get_from_buf(bufferid)
        else:
            obs, reward = self.get_obs_reward(action_decoded)
            if self.is_update_buffer:
                self.update_buf(bufferid, obs, reward)
        
        return obs
    
    def record_perfmetric(self):
        if self.rl_form == "macme" or self.rl_form == "macme_continuous":
            return self._observation[:3]
        else:
            return self._observation[:3]
        
    def reset(self):

        self.steps = 0
        return self.default_obs
        '''
        if (self.rl_form == "macme" or self.rl_form == "macme_continuous"):
            return [self.default_obs] * self.num_agents
        else:
            return self.default_obs
        '''
        
    def actionToConfigs(self, action):
        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        print("[Env][Action]", action)
        # if discrete else continuous
        if self.rl_form == "macme":
            action_decoded = self.helpers.action_decoder_marl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action, self.rl_form)
            self.action_decoded = action_decoded
            write_ok = self.helpers.read_modify_write_ramulator(action_decoded, self.thread_idx)
        return write_ok, action_decoded

# For testing

if __name__ == "__main__":
    
    dramObj = DRAMEnv()
    helpers = helpers()
    logs = []

    obs = dramObj.runDRAMEnv()

    
 
     
    


