import os
import sys
import sys
import subprocess
import time
import re
import random
import numpy as np
import pandas as pd

settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
os.sys.path.insert(0, settings_dir_path)
os.sys.path.insert(0, settings_dir_path + '/../../')
os.sys.path.insert(0, settings_dir_path + '/../../sims/DRAM/binary/DRAMSys_Proxy_Model')

from DRAMSys_Proxy_Model import DRAMSysProxyModel
from configs.sims        import DRAMSys_config
from configs.algos       import rl_config
import gym
from gym.utils           import seeding
from envHelpers          import helpers
import pickle
from loggers             import write_csv

class DRAMEnv(gym.Env):
    def __init__(self,
                workload,
                reward_formulation = "power",
                cost_model = "simulator",):
        # Todo: Change the values if we normalize the observation space
        self.observation_space = gym.spaces.Box(low=0, high=1e10, shape=(1,3))
        self.action_space = gym.spaces.Box(low=0, high=9, shape=(9,))
        self.binary_name = DRAMSys_config.binary_name
        self.exe_path = DRAMSys_config.exe_path
        self.sim_config = DRAMSys_config.sim_config
        self.experiment_name = DRAMSys_config.experiment_name
        self.logdir = DRAMSys_config.logdir

        self.cost_model = cost_model

        self.reward_formulation = reward_formulation
        self.max_steps = 100
        self.steps = 0
        self.total_step = 0
        self.max_episode_len = 10
        self.error_rate = 10
        self.episode = 0
        self.reward_cap = sys.float_info.epsilon
        self.workload = workload
        self.thread_idx = self.read_thread_idx()
        self.err_limit = self.read_upper_err()
        self.helpers = helpers()
        self.load_buf_path = f'buffer/merge_for_rd/{self.workload}.pkl'
        self.save_buf_path = f'buffer/random/{self.workload}_{self.thread_idx}.pkl'
        self.buffer = self.load_buf()
        self.reset()
    '''
    def get_observation(self):
        # Get latency
        stat_path = f"/home/user/ramulator/stats/{self.workload}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.read()
            pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
            matches = pattern.findall(file_content)
            if matches:
                latency = float(matches[0])  # Assuming we only care about the first match
                latency_us = latency / 1e3
        
        # Get energy
        stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{self.workload}/sim.stat"
        with open(stat_path, "r") as f:
            file_content = f.readlines()
            energy, power = file_content[0].split(",")
            energy_uj = float(energy) / 1e6
            power = float(power)
    
        return np.array([latency_us, energy_uj])
    '''
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

    def obs_to_dict(self, obs):
        obs_dict = {}
        obs_dict["Energy"] = obs[0]
        obs_dict["Power"] = obs[1]
        obs_dict["Latency"] = obs[2]

        return obs_dict
    
    def calculate_reward(self, obs):
        reward = -1 * obs[0] * obs[1]
        if reward > self.reward_cap:
            reward = self.reward_cap

        print("Reward:", reward)
        return reward


    '''
    def runDRAMEnv(self):

        #Method to launch the DRAM executables given an action

        env = os.environ.copy()
        working_dir = "/home/user/ramulator/lib/tuner_ddr4/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        
        working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
        _ = subprocess.run(
            ["python3", "main.py", "--workload", self.workload],
            cwd=working_dir,
            env=env,
            timeout=1500
        )
        obs = self.get_observation()
        return obs
    '''
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
        
    def read_upper_err(self):
        #TODO
        return 1e-4

    def get_obs_reward(self, action_dict):
        obs = self.runDRAMEnv()
        error_rate = self.helpers.get_err_rate(action_dict)
        obs = np.append(obs, error_rate)
        
        if error_rate < self.err_limit:
            reward = self.calculate_reward(obs)
        else:
            reward = -1e6
        
        return obs, reward

    def step(self, action_dict):
        '''
        Step method takes action as input and outputs observation
        rewards
        '''
        print("Action Dict",action_dict)

        self.steps += 1
        done = False

        if self.cost_model == "simulator":
            status = self.actionToConfigs(action_dict)
            if(status):
                obs = self.runDRAMEnv()
            else:
                print("Error in writing configs")
            
            error_rate = self.helpers.get_err_rate(action_dict)
            obs = np.append(obs, error_rate)
            
            if error_rate < self.err_limit:
                reward = self.calculate_reward(obs)
            else:
                reward = -1e6
        elif self.cost_model == "proxy_model":
            proxy_model = DRAMSysProxyModel()
            obs = proxy_model.run_proxy_model(action_dict)
            raise "NO IMPLEMENTATION"
        elif self.cost_model == "buffer":
            status = self.actionToConfigs(action_dict)
            action_all = [str(act) for act in action_dict.values()]
            bufferid = '_'.join(action_all)
            if not self.is_buffered(bufferid):
                obs, reward = self.get_obs_reward(action_dict)
                self.update_buf(bufferid, obs, reward)
            else:
                obs, reward = self.get_from_buf(bufferid) 
            self.total_step += 1
            
        print("Episode:", self.episode, " Rewards:", reward)
        return obs, reward, done, {}
    
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

    def reset(self):
        #print("Reseting Environment!")
        self.steps = 0
        return self.observation_space.sample()

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

    def actionToConfigs(self,action):

        '''
        Converts actions output from the agent to update the configuration files.
        '''
        write_ok = False

        if(type(action) == dict):
            #write_ok = self.helpers.read_modify_write_dramsys(action)
            #write_ok = self.helpers.read_modify_write_ramulator_random(action)
            write_ok = self.helpers.read_modify_write_ramulator(action, self.thread_idx)
        else:
            action_decoded = self.helpers.action_decoder_rl(action)
            write_ok = self.helpers.read_modify_write_dramsys(action_decoded)
            raise "No implement"
        return write_ok
    


# For testing

if __name__ == "__main__":
    print("Hey")
    
    dramObj = DRAMEnv(cost_model="proxy_model")
    helpers = helpers()
    logs = []

    obs = dramObj.runDRAMEnv()

    
 
     
    


