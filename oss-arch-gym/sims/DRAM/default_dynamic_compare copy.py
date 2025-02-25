import argparse
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
    print(config_arr)
    data['memspec']['memtimingspec']['CL'] = config_arr[0]
    data['memspec']['memtimingspec']['WL'] = config_arr[1]
    data['memspec']['memtimingspec']['RCD'] = config_arr[2]
    data['memspec']['memtimingspec']['RP'] = config_arr[3]
    data['memspec']['memtimingspec']['RAS'] = config_arr[4]
    data['memspec']['memtimingspec']['RRD_L'] = config_arr[5]
    data['memspec']['memtimingspec']['FAW'] = config_arr[6]
    data['memspec']['memtimingspec']['RFC'] = config_arr[7]
    data['memspec']['memtimingspec']['REFI'] = config_arr[8]

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(data, JsonFile)    

def evaluation(group_idx, n_groups, config_name):
    exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
    
    try:
        env = os.environ.copy()
        working_dir = '/home/user/Desktop/oss-arch-gym/sims/DRAM'
        # Run the command and capture the output
        result = subprocess.run(
            [exe_path, config_name],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=str)
    args = parser.parse_args()
    
    n_groups = 1
    df = pd.read_csv("configs/add/default_config.csv".format(n_groups))
    defaut_value = df.to_numpy().tolist()[0]
    #df = pd.read_csv("configs/canny_groups={}_sequence=20_replay/config.csv".format(n_groups))
    df = pd.read_csv("configs/add/config_single.csv")
    dynamic_config_arr = df.to_numpy().tolist()

    if args.workload == "add":
        config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/add.json"
    else:
        raise "No supported this workload"

    obs_default_arr, obs_dynamic_arr = [], []
    for i in range(n_groups):
        tune_config(defaut_value)
        
        obs_default = evaluation(group_idx=i, n_groups=n_groups, config_name=config_name)
        obs_default_arr.append(obs_default)
        
        tune_config(dynamic_config_arr[i])
        obs_dynamic = evaluation(group_idx=i, n_groups=n_groups, config_name=config_name)
        obs_dynamic_arr.append(obs_dynamic)

    total_default_energy, total_default_latency = 0, 0
    total_dynamic_energy, total_dynamic_latency = 0, 0
    for default, dynamic in zip(obs_default_arr, obs_dynamic_arr):
        default_latency = default[0][2]
        default_energy = default[0][0]
        dynamic_latency = dynamic[0][2]
        dynamic_energy = dynamic[0][0]
        print("default latency: {}  dynamic latency: {}".format(default_latency, dynamic_latency))
        print("default energy: {}  dynamic energy: {}".format(default_energy, dynamic_energy))
        total_default_energy += default_energy
        total_default_latency += default_latency
        total_dynamic_energy += dynamic_energy
        total_dynamic_latency += dynamic_latency

    energy_improve = (total_default_energy-total_dynamic_energy)/total_default_energy * 100
    latency_improve = (total_default_latency-total_dynamic_latency)/total_default_latency * 100
    print("Improvement(%): {}, total default energy: {} dynamic energy: {}".format(energy_improve, total_default_energy, total_dynamic_energy))
    print("Improvement(%): {}, Toal default latency: {} dynamic latency: {}".format(latency_improve, total_default_latency, total_dynamic_latency))

