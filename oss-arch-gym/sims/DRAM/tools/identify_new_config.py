import json
import re
import subprocess
import sys
import numpy as np  
import os
import pandas as pd

def get_observation(outstream):
    '''
    converts the std out from DRAMSys to observation of energy, power, latency
    [Energy (PJ), Power (mW), Latency (ns)]
    '''
    obs = []
    
    keywords = ["Total Energy", "Average Power", "Total Time"]

    all_lines = outstream.splitlines()
    for each_idx in range(len(all_lines)):
        if keywords[0] in all_lines[each_idx]:
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)
        if keywords[1] in all_lines[each_idx]:
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e3)
        if keywords[2] in all_lines[each_idx]:
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)

    obs = np.asarray(obs)
    print('[Environment] Observation(Energy/Power/Latency):', obs)

    return obs

def tune_config(config_arr, n_config=9):
    mem_spec_file = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/configs/memspecs/MICRON_4Gb_DDR4-2400_8bit_A.json"
    with open (mem_spec_file, "r") as JsonFile:
        data = json.load(JsonFile)
    
    for i in range(n_config):
        if i == 0:
            data['memspec']['memtimingspec']['CL'] = config_arr[0]
        elif i == 1:
            data['memspec']['memtimingspec']['WL'] = config_arr[1]
        elif i == 2:
            data['memspec']['memtimingspec']['RCD'] = config_arr[2]
        elif i == 3:
            data['memspec']['memtimingspec']['RP'] = config_arr[3]
        elif i == 4:
            data['memspec']['memtimingspec']['RAS'] = config_arr[4]
        elif i == 5:
            data['memspec']['memtimingspec']['RRD_L'] = config_arr[5]
        elif i == 6:
            data['memspec']['memtimingspec']['FAW'] = config_arr[6]
        elif i == 7:
            data['memspec']['memtimingspec']['RFC'] = config_arr[7]
        elif i == 8:
            data['memspec']['memtimingspec']['REFI'] = int(config_arr[8])
        elif i == 9:
             data['memspec']['memtimingspec']['clkMhz'] = config_arr[9]

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(data, JsonFile)    

def evaluation():
    exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
    config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/synthetic/random_access=1e4.json"
    try:
        env = os.environ.copy()
        working_dir = '/home/user/Desktop/oss-arch-gym/sims/DRAM'
        # Run the command and capture the output
        result = subprocess.run(
            [exe_path, config_name],
            cwd=working_dir,
            env=env,
            timeout=20,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
            
        outstream = result.stdout.decode()
        obs = get_observation(outstream)
        return obs
    except subprocess.TimeoutExpired:
        raise "Process terminated due to timeout."


def generate_changed_config(min_value, max_value):
    step = (max_value-min_value)//10
    values = list(range(min_value, max_value, step))
    return values
    
if __name__ == "__main__":
    n_config=10
    
    defaut_value = [16, 16, 16, 16, 39, 6, 26, 313, 4680, 1200]

    frequencys = generate_changed_config(1000, 10000)
    for frequency in frequencys:
        defaut_value = [16, 16, 16, 16, 39, 6, 26, 313, 4680, frequency]
        tune_config(defaut_value, n_config=10)
        obs = evaluation()
        obs_all.append(obs)