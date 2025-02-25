import argparse
import subprocess
import os
import multiprocessing

HOME_DIR = "/home/user/ramulator/"
DATA_DIR = "/mnt/nvme3n1/benchmark/"
STATS_DIR = (HOME_DIR+f'stats/')
SIM_MODE = "dram"

def modify_file(i_path, o_path, config):
    with open(i_path, 'r') as file:
        filedata = file.read()    
    filedata = filedata.replace('{mapping}', config[0])
    filedata = filedata.replace('{pagepolicy}', config[1]) 
    with open(o_path, 'w') as file:
        file.write(filedata)

def change_config(config_pair, workload_pair):
    config, _ = config_pair
    _, workload_idx = workload_pair
    i_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', f'DDR4-sample.cfg')
    o_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', f'DDR4-config_{workload_idx}.cfg')
    map_config_path = {
        "row_bank_col": "row_bank_col.map",
        "bank_row_col": "bank_row_col.map",
        "permutation": "permutation.map",
        "row_bank_col2": "row_bank_col2.map",
        "row_bank_col3": "row_bank_col3.map",
    }
    
    if config[0] in map_config_path:
        map_config = os.path.join(HOME_DIR, 'mappings', 'ours', 'ddr4', map_config_path[config[0]])
    else:
        raise ValueError("No such config")

    if config[1] == "open":
        pagepolicy_config = "open" 
    elif config[1] == "closed": 
        pagepolicy_config = "closed"
    else:
        raise ValueError("No such config")

    config_all = (map_config, pagepolicy_config)
    modify_file(i_path, o_path, config_all)

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
    mappings = ["bank_row_col", "row_bank_col", "permutation", "row_bank_col2", "row_bank_col3"]
    pagepolicys = ["open", "closed"]
    
    config_all = []
    for mapping in mappings:
        for pagepolicy in pagepolicys:
            config_all.append((mapping, pagepolicy))
    
    return config_all

def get_workload_set(benchmark_name):
    workload_dir = os.path.join(DATA_DIR, benchmark_name)
    return [os.path.join(workload_dir, workload_name) for workload_name in os.listdir(workload_dir)]

# workload_pair: (workload, workload_idx)
def tune_workload(workload_pair):
    config_all = get_config_set()
    for config_idx, config in enumerate(config_all):
        config_pair = (config, config_idx)
        tune(config_pair, workload_pair)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--benchmark', type=str, required=True, help='pagepolicy')
    args = parser.parse_args()
    workload_set = get_workload_set(args.benchmark)
    if args.benchmark == "cpu2017" or args.benchmark == "cpu2017_benchmark":
        SIM_MODE = "cpu"
    
    STATS_DIR = STATS_DIR + args.benchmark
    os.makedirs(STATS_DIR, exist_ok=True)
    
    n_cores = 10
    workload_pairs = [(workload, idx) for idx, workload in enumerate(workload_set)]

    with multiprocessing.Pool(n_cores) as pool:
        pool.map(tune_workload, workload_pairs)

