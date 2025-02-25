import json
import re
import subprocess
import sys
import numpy as np  
import os
import pandas as pd

'''
0. Require: default setting、dynamic setting
1. change n_groups
2. change the path of config and trace directory: 
e.g., mv DRAMSys/library/traces/ours/canny_groups=10 DRAMSys/library/simulations/ours/canny
e.g., mv DRAMSys/library/simulations/ours/canny_groups=10 DRAMSys/library/simulations/ours/canny
'''

def get_observation(outstream):
    '''
    converts the std out from DRAMSys to observation of energy, power, latency
    [Energy (PJ), Power (mW), Latency (ns)]
    '''
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
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)

    obs = np.asarray(obs)
    print('[Environment] Observation:', obs)

    return obs

def tune_config(config_arr):
    mem_spec_file = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/configs/memspecs/MICRON_4Gb_DDR4-2400_8bit_A.json"
    with open (mem_spec_file, "r") as JsonFile:
        data = json.load(JsonFile)
    
    data['memspec']['memtimingspec']['CL'] = config_arr[0]
    data['memspec']['memtimingspec']['WL'] = config_arr[1]
    data['memspec']['memtimingspec']['RCD'] = config_arr[2]
    data['memspec']['memtimingspec']['RP'] = config_arr[3]
    data['memspec']['memtimingspec']['RAS'] = config_arr[4]
    data['memspec']['memtimingspec']['RRD_L'] = config_arr[5]
    data['memspec']['memtimingspec']['FAW'] = config_arr[6]
    data['memspec']['memtimingspec']['REFI'] = config_arr[7]

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(data, JsonFile)    

def evaluation(sim_name):
    exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
    
    try:
        env = os.environ.copy()
        working_dir = '/home/user/Desktop/oss-arch-gym/sims/DRAM'
        # Run the command and capture the output
        result = subprocess.run(
            [exe_path, sim_name],
            cwd=working_dir,
            env=env,
            timeout=10,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
            
        outstream = result.stdout.decode()

        obs = get_observation(outstream)
        obs = obs.reshape(1,3)
        return obs
    except subprocess.TimeoutExpired:
        raise "Process terminated due to timeout."
     
if __name__ == "__main__":

    df = pd.read_csv("/home/user/Desktop/oss-arch-gym/sims/DRAM/configs/add/config_add.csv")
    config1 = df.to_numpy().tolist()[0]
    sim_name1 = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/add.json"
    df = pd.read_csv("/home/user/Desktop/oss-arch-gym/sims/DRAM/configs/copy/config_copy.csv")
    config2 = df.to_numpy().tolist()[0]
    sim_name2 = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/copy.json"

    '''
    tune_config(canny_config)
    obs_app1_same = evaluation(sim_name_canny)
    tune_config(blackscoles_config)
    obs_app1_diff = evaluation(sim_name_canny)
    
    same_latency = obs_app1_same[0][2]
    same_energy = obs_app1_same[0][0]
    diff_latency = obs_app1_diff[0][2]
    diff_energy = obs_app1_diff[0][0]
    print("On app1, latency using app1 config : {}  latency using app2 config latency: {}".format(same_latency, diff_latency))
    print("On app1, latency using app1 energy : {}  latency using app2 config energy: {}".format(same_energy, diff_energy))
    '''
    
    tune_config(config2)
    obs_app2_same = evaluation(sim_name2)
    tune_config(config1)
    obs_app2_diff = evaluation(sim_name2)

    same_latency = obs_app2_same[0][2]
    same_energy = obs_app2_same[0][0]
    diff_latency = obs_app2_diff[0][2]
    diff_energy = obs_app2_diff[0][0]
    print("On app2, latency using app2 config : {}  latency using app1 config : {}".format(same_latency, diff_latency))
    print("On app2, energy using app2 config : {}  energy using app1 config : {}".format(same_energy, diff_energy))

