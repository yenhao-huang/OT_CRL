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

def evaluation(config_name):
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
    
    df = pd.read_csv(f"/home/user/Desktop/oss-arch-gym/sims/DRAM/configs/{args.workload}/default_config.csv")
    default_config = df.to_numpy().tolist()[0]
    df = pd.read_csv(f"/home/user/Desktop/oss-arch-gym/sims/DRAM/configs/{args.workload}/config_{args.workload}.csv")
    tuned_config = df.to_numpy().tolist()[0]
    
    if args.workload == "add":
        config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/add.json"
    elif args.workload == "scale":
        config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/scale.json"
    elif args.workload == "copy":
        config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/copy.json"
    elif args.workload == "triad":
        config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/stream/triad.json"
    else:
        raise "No supported this workload"

    tune_config(default_config)
    obs_default = evaluation(config_name=config_name)
    
    tune_config(tuned_config)
    obs_tuned = evaluation(config_name=config_name)

    total_default_energy, total_default_latency = 0, 0
    total_dynamic_energy, total_dynamic_latency = 0, 0
    default_latency = obs_default[0][2]
    default_energy = obs_default[0][0]
    tuned_latency = obs_tuned[0][2]
    tuned_energy = obs_tuned[0][0]

    energy_improve = (default_energy-tuned_energy)/default_energy * 100
    latency_improve = (default_latency-tuned_latency)/default_latency * 100
    print("Improvement(%): {}, total default energy: {} tuned energy: {}".format(energy_improve, default_energy, tuned_energy))
    print("Improvement(%): {}, Toal default latency: {} tuned latency: {}".format(latency_improve, default_latency, tuned_latency))