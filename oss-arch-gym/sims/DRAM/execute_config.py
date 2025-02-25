import os
import subprocess
import re

import numpy as np
import pandas as pd

def read_modify_write_ramulator(action):
    print("[envHelpers][Action]", action)
    op_success = False
    mem_spec_file = f"/home/user/ramulator/lib/tuner_ddr4/config.txt"

    lines = []
    lines.append(f"pagepolicy,{action['pagepolicy']}\n")
    lines.append(f"addressmapping,{action['addressmapping']}\n")
    lines.append(f"rcd,{str(action['rcd'])}\n")
    lines.append(f"rp,{str(action['rp'])}\n")
    lines.append(f"ras,{str(action['ras'])}\n")
    lines.append(f"rrd,{str(action['rrd'])}\n")
    lines.append(f"refi,{str(action['refi'])}\n")

    try:
        with open (mem_spec_file, "w") as f:
            f.writelines(lines)
        
        op_success = True
    except Exception as e:
        print(str(e))
        op_success = False

    return op_success

def runDRAMEnv(workload):
    '''
    Method to launch the DRAM executables given an action
    '''
    env = os.environ.copy()
    working_dir = "/home/user/ramulator/lib/tuner_ddr4/"
    _ = subprocess.run(
        ["python3", "main.py", "--workload", workload],
        cwd=working_dir,
        env=env,
        timeout=1500
    )
            
    working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
    _ = subprocess.run(
        ["python3", "main.py", "--workload", workload],
        cwd=working_dir,
        env=env,
        timeout=1500
    )
    
    obs = get_observation(workload)
    return obs

def get_observation(workload):
    # Get latency
    stat_path = f"/home/user/ramulator/stats/{workload}/sim.stat"
    with open(stat_path, "r") as f:
        file_content = f.read()
        pattern = re.compile(r"ramulator\.dram_cycles\s+(\d+)")
        matches = pattern.findall(file_content)
        if matches:
            latency = float(matches[0])  # Assuming we only care about the first match
            latency_us = latency / 1e3
    
    # Get energy
    stat_path = f"/home/user/ramulator-pim/common/DRAMPower/stats/{workload}/sim.stat"
    with open(stat_path, "r") as f:
        file_content = f.readlines()
        energy, power = file_content[0].split(",")
        energy_uj = float(energy) / 1e6
        power = float(power)
    
    edp = latency_us * energy_uj
    return np.array([latency_us, energy_uj, edp])

def save_obs(obs_all, name):
    df = pd.DataFrame(obs_all)
    df.index = ["Latency", "Energy", "EDP"]
    df.to_csv(f"configs/{name}.csv")

if __name__ == '__main__':
    '''
    workload = "ferret"
    action = {"pagepolicy":"closed", "addressmapping":"permutation", "rcd":12, "rp":12, "ras":36, "rrd":5, "refi":191880}
    read_modify_write_ramulator(action)
    obs = runDRAMEnv(workload)
    print(f"[Workload: {workload}; Observation: {obs}]")

    #archgym_action = {"pagepolicy":"open", "addressmapping":"permutation", "rcd":12, "rp":15, "ras":34, "rrd":3, "refi":238680}
    micron_action = {"pagepolicy":"open", "addressmapping":"row_bank_col", "rcd":16, "rp":16, "ras":39, "rrd":6, "refi":4680}
    name = "micron_config"
    action = micron_action
    workload_all = [
        'cactusADM', 'copy', 'ferret', 'fotonik3d', 'freqmine',
        'GemsFDTD', 'hmmer', 'lbm', 'namd', 'parest', 'sphinx3', 
        'swaptions','triad', 'xz', 'zeusmp'
    ]

    obs_all = {}
    for workload in workload_all:
        read_modify_write_ramulator(action)
        obs = runDRAMEnv(workload)
        obs_all.update({workload: obs})
    save_obs(obs_all, name)
    '''
    pagepolicy_map = {0: "open", 1: "closed"}
    addressmapping_map = {0: "bank_row_col", 
                        1: "row_bank_col", 
                        2: "permutation", 
                        3: "row_bank_col2", 
                        4: "row_bank_col3"
                        }

    dreamcrl_action = [
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 14, "ras": 39, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 13, "ras": 38, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 13, "rp": 12, "ras": 39, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[0], "addressmapping": addressmapping_map[1], "rcd": 12, "rp": 12, "ras": 37, "rrd": 5, "refi": 51480},
        {"pagepolicy": pagepolicy_map[0], "addressmapping": addressmapping_map[2], "rcd": 15, "rp": 14, "ras": 20, "rrd": 5, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[1], "rcd": 12, "rp": 13, "ras": 38, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 12, "ras": 38, "rrd": 5, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 12, "ras": 39, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[0], "addressmapping": addressmapping_map[1], "rcd": 12, "rp": 12, "ras": 37, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[0], "addressmapping": addressmapping_map[1], "rcd": 12, "rp": 12, "ras": 39, "rrd": 5, "refi": 51480},
        {"pagepolicy": pagepolicy_map[0], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 12, "ras": 39, "rrd": 6, "refi": 51480},
        {"pagepolicy": pagepolicy_map[0], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 13, "ras": 38, "rrd": 3, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 12, "rp": 12, "ras": 39, "rrd": 3, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[2], "rcd": 13, "rp": 13, "ras": 38, "rrd": 5, "refi": 51480},
        {"pagepolicy": pagepolicy_map[1], "addressmapping": addressmapping_map[1], "rcd": 12, "rp": 12, "ras": 38, "rrd": 3, "refi": 51480}
    ]
    workload_all = [
        'cactusADM', 'copy', 'ferret', 'fotonik3d', 'freqmine',
        'GemsFDTD', 'hmmer', 'lbm', 'namd', 'parest', 'sphinx3', 
        'swaptions','triad', 'xz', 'zeusmp'
    ]
    name = "dreamcrl_config"

    obs_all = {}
    for workload, action in zip(workload_all, dreamcrl_action):
        read_modify_write_ramulator(action)
        obs = runDRAMEnv(workload)
        obs_all.update({workload: obs})
    save_obs(obs_all, name)
