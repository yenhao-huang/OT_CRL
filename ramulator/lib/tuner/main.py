import subprocess
import os
import argparse

HOME_DIR = "/home/user/ramulator/"
DATA_DIR = "/mnt/nvme3n1/benchmark/"
STATS_DIR = (HOME_DIR+'stats/')

def change_mapping(i_path, o_path, config):
    with open(i_path, 'r') as file:
        filedata = file.read()
        
    filedata = filedata.replace('{mapping}', config)
        
    with open(o_path, 'w') as file:
        file.write(filedata)

def change_pagepolicy(i_path, o_path, config):
    with open(i_path, 'r') as file:
        filedata = file.read()
        
    filedata = filedata.replace('{pagepolicy}', config)
        
    with open(o_path, 'w') as file:
        file.write(filedata)

def change_config(config):
    i_path=(HOME_DIR+'configs/DDR3-sample.cfg')
    o_path=(HOME_DIR+'configs/DDR3-config.cfg')
    if config[0] == "row_bank_col":
        map_config = "/home/user/ramulator/mappings/ours/row_bank_col.map"
    elif config[0] == "bank_row_col":
        map_config = "/home/user/ramulator/mappings/ours/bank_row_col.map"
    elif config[0] == "permutation":
        map_config = "/home/user/ramulator/mappings/ours/permutation.map"
    else:
        raise "No ths config"
    
    change_mapping(i_path, o_path, map_config)
    
    i_path=(HOME_DIR+'/lib/option_pagepolicy_sample.txt')
    o_path=(HOME_DIR+'/lib/option_pagepolicy.txt')
    if config[1] == "open":
        pagepolicy_config = "1"
    elif config[1] == "closed":
        pagepolicy_config = "0"
    else:
        raise "No ths config"
    
    change_pagepolicy(i_path, o_path, pagepolicy_config)

def execute_sim(workload):
        if os.path.isfile(workload):
            # 构建ramulator命令
            ramulator_cmd = [
                os.path.join(HOME_DIR, "ramulator"),
                os.path.join(HOME_DIR, "configs/DDR3-config.cfg"),
                "--mode=dram",
                workload
            ]
            
            # 执行ramulator命令
            subprocess.run(ramulator_cmd)
        else:
            raise "No this workload"

def extract_stats(o_path):
    # 处理DDR3.stats文件
    with open("DDR3.stats", "r") as stats_file:
        dram_cycles = [line for line in stats_file if "dram_cycles" in line]
    
    # 写入结果文件
    
    with open(o_path, "w") as result_file:
        result_file.writelines(dram_cycles)

def tune(config, workload):
    change_config(config)
    execute_sim(workload)

# Return array of tupple, e.g., [(mapping1, pagepolicy1), ...]
def get_config_set():
    mappings = ["bank_row_col", "row_bank_col", "permutation"]
    pagepolicys = ["open", "closed"]
    
    config_all = []
    for mapping in mappings:
        for pagepolicy in pagepolicys:
            config_all.append((mapping, pagepolicy))
    
    return config_all

def get_workload_set(benchmark_name):
    workload_dir = f"{DATA_DIR}{benchmark_name}/"
    workloads = [os.path.join(workload_dir, workload_name) for workload_name in os.listdir(workload_dir)]
    return workloads
    
if __name__ == "__main__":
    '''
    E.g., python3 main.py --benchmark comm
    '''
    
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--benchmark', type=str, required=True, help='pagepolicy')
    args = parser.parse_args()
    
    config_all = get_config_set()
    workload_set = get_workload_set(args.benchmark)

    for workload in workload_set:
        for config_idx, config in enumerate(config_all):
            print(config)
            tune(config, workload)
            workload_name = workload.split("/")[-1]
            o_path = os.path.join(STATS_DIR, f"{workload_name}_{config_idx}.stl")
            extract_stats(o_path)