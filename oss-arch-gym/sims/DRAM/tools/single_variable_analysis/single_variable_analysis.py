import copy
import json
import re
import subprocess
import sys
import numpy as np  
import os
import pandas as pd
from pathlib import Path

# convert to s/j/w
ENERGY_UNIT_CONVERT = {
    "pJ" : 1e-12,
    "nJ" : 1e-9,
}
Latency_UNIT_CONVERT = {
    "ps" : 1e-12,
    "ns" : 1e-9,
}
Power_UNIT_CONVERT = {
    "mW" : 1e-3,
    "W" : 1, 
}

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
            unit = all_lines[each_idx].split(":")[1].split()[1]
            if unit in ENERGY_UNIT_CONVERT.keys():
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])* ENERGY_UNIT_CONVERT[unit])
            else:
                raise "Wrong format"
        elif keywords[1] in all_lines[each_idx]:
            unit = all_lines[each_idx].split(":")[1].split()[1]
            if unit in Power_UNIT_CONVERT.keys():
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])* Power_UNIT_CONVERT[unit])
            else:
                raise "Wrong format"
        elif keywords[2] in all_lines[each_idx]:
            unit = all_lines[each_idx].split(":")[1].split()[1]
            if unit in Latency_UNIT_CONVERT.keys():
                obs.append(float(all_lines[each_idx].split(":")[1].split()[0])* Latency_UNIT_CONVERT[unit])
            else:
                raise "Wrong format"
    print('[Environment] Observation(Energy/Power/Latency):', obs)

    return obs

def tune_config(config_arr):
    mem_spec_file = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/configs/memspecs/MICRON_4Gb_DDR4-2400_8bit_A.json"
    with open (mem_spec_file, "r") as JsonFile:
        lines_mspec = json.load(JsonFile)
    
    mc_file = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/configs/mcconfigs/fr_fcfs.json"
    with open (mc_file, "r") as JsonFile:
        lines_mc = json.load(JsonFile)
    
    for i in range(len(config_arr)):
        if i == 0:
            lines_mspec['memspec']['memtimingspec']['CL'] = config_arr['CL']
        elif i == 1:
            lines_mspec['memspec']['memtimingspec']['WL'] = config_arr['WL']
        elif i == 2:
            lines_mspec['memspec']['memtimingspec']['RCD'] = config_arr['RCD']
        elif i == 3:
            lines_mspec['memspec']['memtimingspec']['RP'] = config_arr['RP']
        elif i == 4:
            lines_mspec['memspec']['memtimingspec']['RAS'] = config_arr['RAS']
        elif i == 5:
            lines_mspec['memspec']['memtimingspec']['RRD_L'] = config_arr['RRD_L']
        elif i == 6:
            lines_mspec['memspec']['memtimingspec']['FAW'] = config_arr['FAW']
        elif i == 7:
            lines_mspec['memspec']['memtimingspec']['RFC'] = config_arr['RFC']
        elif i == 8:
            lines_mspec['memspec']['memtimingspec']['REFI'] = config_arr['REFI']
        elif i == 9:
            lines_mspec['memspec']['memtimingspec']['clkMhz'] = config_arr['clkMhz']
        elif i == 10:
            lines_mc['mcconfig']['PagePolicy'] = config_arr['pagepolicy']

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(lines_mspec, JsonFile)    
    
    with open (mc_file, "w") as JsonFile:
        json.dump(lines_mc, JsonFile)    

def evaluation():
    exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
    config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/parsec/blackscholes_dy/ratio=0.5/part2.json"
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


def generate_changed_config(min_value, max_value, n_comb):
    step = (max_value-min_value)//n_comb
    values = list(range(min_value, max_value, step))
    return values

def single_variable_analysis(changed_config_list, option_name):
    default_values = copy.deepcopy(DEFAULT_VALUES)
    
    obs_all = []
    for config in changed_config_list:
        default_values[option_name] = config
        tune_config(default_values)
        obs = [config]
        obs += evaluation()
        obs_all.append(np.array(obs))
    df = pd.DataFrame(obs_all)
    df.columns = [option_name, "Energy", "Power", "Latency"]
    o_dir = Path("/home/user/Desktop/oss-arch-gym/sims/DRAM/tools/single_variable_analysis/results")
    df.to_csv(o_dir / f"{option_name}.csv", index=False)

DEFAULT_VALUES = {
    "CL" : 16,
    "WL" : 16,
    "RCD" : 16,
    "RP" : 16,
    "RAS" : 39,
    "RRD_L" : 6,
    "FAW" : 26,
    "RFC" : 313,
    "REFI" : 4680,
    "clkMhz" : 1200,
    "pagepolicy" : "Open",
}

if __name__ == "__main__":
    print("Page Policy")
    pagepolicy_all = ["Open", "OpenAdaptive", "Closed", "ClosedAdaptive"]
    single_variable_analysis(pagepolicy_all, "pagepolicy")
    
    '''
    print("RAS")
    ras_all = generate_changed_config(19, 42, n_comb=5)
    single_variable_analysis(ras_all, "RAS")

    print("CL")
    cas_all = generate_changed_config(8, 24, n_comb=5)
    single_variable_analysis(cas_all, "CL")
    
    print("WL")
    cwl_all = generate_changed_config(8, 24, n_comb=5)
    single_variable_analysis(cwl_all, "WL")
    
    print("RCD")
    rcd_all = generate_changed_config(8, 24, n_comb=5)
    single_variable_analysis(rcd_all, "RCD")
    
    print("RP")
    rp_all = generate_changed_config(8, 24, n_comb=5)
    single_variable_analysis(rp_all, "RP")
    
    print("RAS")
    ras_all = generate_changed_config(19, 42, n_comb=5)
    single_variable_analysis(ras_all, "RAS")
    
    print("RRD_L")
    rrd_all = generate_changed_config(3, 9, n_comb=5)
    single_variable_analysis(rrd_all, "RRD_L")
    
    print("FAW")
    faw_all = generate_changed_config(13, 39, n_comb=5)
    single_variable_analysis(cas_all, "FAW")
    
    print("REFI")
    refi_all = generate_changed_config(4680, 468000, n_comb=10)
    single_variable_analysis(refi_all, "REFI")
    '''
    