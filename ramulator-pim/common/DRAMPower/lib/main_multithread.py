import argparse
from pathlib import Path
import subprocess
import os

HOME_DIR = "/home/user/ramulator-pim/common/DRAMPower/"
DATA_DIR = "/mnt/nvme3n1/benchmark/drampower/"
SIMSPEC_PATH = HOME_DIR + "memspecs/ours/MICRON_4Gb_DDR4-2400_8bit_A.xml"
STATS_DIR = (HOME_DIR+f'stats/')
REFI = None
ADDRESSMAPPING = None
THREAD_IDX = None

def execute_simualtor(workload):
    # Command components
    trace_path = DATA_DIR + f"{workload}/{ADDRESSMAPPING}_{REFI}.cmdtrace"
    sim_path = HOME_DIR + "drampower"
    command = [
        sim_path,
        '-m', SIMSPEC_PATH,
        '-c', trace_path
    ]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # Check if there was an error
    if result.returncode != 0:
        print("Error in command execution!")
        print("Error:", result.stderr)
        return None

    # Process the output to find the total trace energy and average power
    for line in result.stdout.split('\n'):
        if "Total Trace Energy:" in line:
            energy_value = float(line.split(':')[1].strip().split(' ')[0])
        elif "Average Power:" in line:
            average_power = float(line.split(':')[1].strip().split(' ')[0])
    
    return energy_value, average_power

def write_stat(energy, average_power):
    o_path = os.path.join(STATS_DIR, "sim.stat")
    with open(o_path, "w") as f:
        f.writelines(f"{energy},{average_power}")

def get_config():
    option_path = f"/home/user/ramulator/lib/tuner_ddr4_multithread/config/config_{THREAD_IDX}.txt" 
    with Path(option_path).open("r") as f:
        lines = f.readlines()
    
    addressmapping = lines[1].split(",")[1].strip()
    refi = lines[6].split(",")[1].strip()

    if int(refi) > 191880:
        refi = "191880"
        
    return addressmapping, refi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--workload', type=str, required=True, help='workload')
    parser.add_argument('--threadidx', type=str, required=True, help='threadidx')
    args = parser.parse_args()    
    THREAD_IDX = args.threadidx
    STATS_DIR = os.path.join(STATS_DIR, args.workload, THREAD_IDX)
    os.makedirs(STATS_DIR, exist_ok=True)
    ADDRESSMAPPING, REFI = get_config()
    energy, average_power = execute_simualtor(args.workload)
    print("Default config, Total Trace Energy: {} pJ".format(energy))
    write_stat(energy, average_power)