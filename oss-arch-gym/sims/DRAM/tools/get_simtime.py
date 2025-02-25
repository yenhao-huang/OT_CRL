import argparse
import numpy as np
import re
import os
import subprocess
import time

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
    

    return np.array([latency_us, energy_uj])

def runDRAMEnv(workload):
    '''
    Method to launch the DRAM executables given an action
    '''
    #Time1
    env = os.environ.copy()
    start_time_1 = time.time()
    working_dir = "/home/user/ramulator/lib/tuner_ddr4/"
    _ = subprocess.run(
        ["python3", "main.py", "--workload", workload],
        cwd=working_dir,
        env=env,
        timeout=1500
    )
    end_time_1 = time.time()
    execution_time_1 = end_time_1 - start_time_1
    
    #Time2
    start_time_2 = time.time()
    working_dir = "/home/user/ramulator-pim/common/DRAMPower/lib/"
    _ = subprocess.run(
        ["python3", "main.py", "--workload", workload],
        cwd=working_dir,
        env=env,
        timeout=1500
    )
    end_time_2 = time.time()
    execution_time_2 = end_time_2 - start_time_2

    # Print out the execution times
    print(f"Execution time for first subprocess: {execution_time_1:.2f} seconds")
    print(f"Execution time for second subprocess: {execution_time_2:.2f} seconds")
    obs = get_observation(workload)
    return obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--workload', type=str, required=True, help='workload')
    args = parser.parse_args()
    runDRAMEnv(args.workload)