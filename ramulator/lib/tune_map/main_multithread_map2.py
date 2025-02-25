import argparse
import subprocess
import os
import multiprocessing
from pathlib import Path

'''
total_bit  26 bits
'''

HOME_DIR = "/home/user/ramulator/"
DATA_DIR = "/mnt/nvme3n1/benchmark/"
STATS_DIR = (HOME_DIR+f'stats/')
SIM_MODE = "dram"

def modify_file(i_path, o_path, workload_idx):
    with open(i_path, 'r') as file:
        filedata = file.read()    
    
    filedata = filedata.replace('{mapping}', f"/home/user/ramulator/mappings/ours/ddr4/generate_{workload_idx}.map") 
    filedata = filedata.replace('{pagepolicy}', "open") 
    with open(o_path, 'w') as file:
        file.write(filedata)

def assign_bits(start_bit):
    total_bits = 26
    Ro_bits = 15
    Co_bits = 7
    Ba_bits = 2
    Bg_bits = 2
    
    lines = []
    if Ro_bits + start_bit > total_bits:
        raise "no this mapping"
    
    # Calculate Ro range
    Ro_start = start_bit
    Ro_end = start_bit + Ro_bits - 1
    lines.append(f"Ro {Ro_bits - 1}:0 = {Ro_end}:{Ro_start}\n")
    
    # Initialize remaining bits
    remaining_bits = list(range(total_bits))
    
    # Remove Ro bits from the remaining bits
    for bit in range(Ro_start, Ro_end + 1):
        remaining_bits.remove(bit)
    
    # Assign Co, Ba, and Bg bits
    Co_assign = remaining_bits[:Co_bits]
    Ba_assign = remaining_bits[Co_bits:Co_bits + Ba_bits]
    Bg_assign = remaining_bits[Co_bits + Ba_bits:Co_bits + Ba_bits + Bg_bits]
    
    # Format the assignments
    for i, co in enumerate(Co_assign):
        lines.append(f"Co {i} = {co}\n")
    for i, ba in enumerate(Ba_assign):
        lines.append(f"Ba {i} = {ba}\n")
    for i, bg in enumerate(Bg_assign):
        lines.append(f"Bg {i} = {bg}\n")
        
    return lines

def change_config(config_pair, workload_pair):
    config, _ = config_pair
    _, workload_idx = workload_pair
    
    # modify mapping
    lines = assign_bits(config)
    o_path = Path(HOME_DIR) / f"mappings/ours/ddr4/generate_{workload_idx}.map"
    
    with o_path.open("w") as f:
        f.writelines(lines)
    
    # modify sim
    i_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', f'DDR4-sample.cfg')
    o_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', f'DDR4-config_{workload_idx}.cfg')
    modify_file(i_path, o_path, workload_idx)

def execute_sim(config_pair, workload_pair):
    _, config_idx = config_pair
    workload, workload_idx = workload_pair
    config_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', f'DDR4-config_{workload_idx}.cfg')
    stat_path = os.path.join(STATS_DIR, f"workload={os.path.basename(workload)}_config={config_idx}.stat")
    if os.path.isfile(workload):
        ramulator_cmd = [
            os.path.join(HOME_DIR, "ramulator"),
            config_path,
            f"--mode={SIM_MODE}",
            "--stats",
            stat_path,
            workload
        ]
        subprocess.run(ramulator_cmd)
    else:
        raise FileNotFoundError("No this workload")

def tune(config_pair, workload_pair):
    change_config(config_pair, workload_pair)
    execute_sim(config_pair, workload_pair)

# Return array of tupple, e.g., [(mapping1, pagepolicy1), ...]
def get_config_set():
    mapping_all = []
    for mapping in range(0, MAX_MAPPING_VALUE):
        mapping_all.append(mapping)

    return mapping_all

def get_workload_set(benchmark_name):
    workload_dir = os.path.join(DATA_DIR, benchmark_name)
    return [os.path.join(workload_dir, workload_name) for workload_name in os.listdir(workload_dir)]

# workload_pair: (workload, workload_idx)
def tune_workload(workload_pair):
    mapping_all = get_config_set()
    for config_idx, config in enumerate(mapping_all):
        config_pair = (config, config_idx)
        tune(config_pair, workload_pair)

MAX_MAPPING_VALUE = 12
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--benchmark', type=str, required=True, help='pagepolicy')
    args = parser.parse_args()
    workload_set = get_workload_set(args.benchmark)
    if args.benchmark == "cpu2017":
        SIM_MODE = "cpu"
    
    STATS_DIR = STATS_DIR + args.benchmark
    os.makedirs(STATS_DIR, exist_ok=True)
    
    n_cores = 6
    workload_pairs = [(workload, idx) for idx, workload in enumerate(workload_set)]

    with multiprocessing.Pool(n_cores) as pool:
        pool.map(tune_workload, workload_pairs)

