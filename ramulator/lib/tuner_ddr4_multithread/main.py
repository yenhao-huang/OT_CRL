import argparse
import subprocess
import os
from pathlib import Path

HOME_DIR = "/home/user/ramulator/"
DATA_DIR = "/mnt/nvme3n1/benchmark/"
STATS_DIR = (HOME_DIR+f'stats/')
SIM_MODE = "dram"
THREAD_IDX = None

def execute_sim(workload_path):
    config_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4_multithread', f'DDR4-config_{THREAD_IDX}.cfg')
    stat_path = os.path.join(STATS_DIR, f"sim.stat")
    print(workload_path)
    if os.path.isfile(workload_path):
        ramulator_cmd = [
            os.path.join(HOME_DIR, "ramulator"),
            config_path,
            f"--mode={SIM_MODE}",
            "--stats",
            stat_path,
            workload_path
        ]
        subprocess.run(ramulator_cmd)
    else:
        raise FileNotFoundError("No this workload")

def modify_file(i_path, o_path, config):
    with open(i_path, 'r') as file:
        filedata = file.read()    
    
    filedata = filedata.replace('{pagepolicy}', config[0])
    filedata = filedata.replace('{mapping}', config[1]) 
    filedata = filedata.replace('{timing}', config[2])
     
    with open(o_path, 'w') as file:
        file.write(filedata)

def change_sim(config):
    i_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4_multithread', 'DDR4-sample.cfg')
    o_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4_multithread', f'DDR4-config_{THREAD_IDX}.cfg')
    map_config_path = {
        "row_bank_col": "row_bank_col.map",
        "bank_row_col": "bank_row_col.map",
        "permutation": "permutation.map",
        "row_bank_col2": "row_bank_col2.map",
        "row_bank_col3": "row_bank_col3.map",
    }
    
    if config[0] == "open":
        pagepolicy_config = "open" 
    elif config[0] == "closed": 
        pagepolicy_config = "closed"
    else:
        raise ValueError("No such config")
    
    if config[1] in map_config_path:
        map_config = os.path.join(HOME_DIR, 'mappings', 'ours', 'ddr4', map_config_path[config[1]])
    else:
        raise ValueError("No such config")

    timing = f"{config[2]},{config[3]},{config[4]},{config[5]},{config[6]}"
    config_all = (pagepolicy_config, map_config, timing)
    modify_file(i_path, o_path, config_all)

def tune(config, workload_path):
    change_sim(config)
    execute_sim(workload_path)

def get_config():
    option_path = os.path.join(HOME_DIR, f"lib/tuner_ddr4_multithread/config/config_{THREAD_IDX}.txt")
    with Path(option_path).open("r") as f:
        lines = f.readlines()
    
    cg1 = lines[0].split(",")[1].strip()
    cg2 = lines[1].split(",")[1].strip()
    cg3 = lines[2].split(",")[1].strip()
    cg4 = lines[3].split(",")[1].strip()
    cg5 = lines[4].split(",")[1].strip()
    cg6 = lines[5].split(",")[1].strip()
    cg7 = lines[6].split(",")[1].strip()

    return [cg1, cg2, cg3, cg4, cg5, cg6, cg7]

    
def tune_workload(workload_path):
    config = get_config()
    tune(config, workload_path)

def get_workload_path(workload):
    if workload == "GemsFDTD":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/459.GemsFDTD"
    elif workload == "cactusADM":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/436.cactusADM"
    elif workload == "zeusmp":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/434.zeusmp"
    elif workload == "lbm":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/470.lbm"
    elif workload == "xz":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/557.xz"    
    elif workload == "fotonik3d":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/549.fotonik3d"
    elif workload == "mcf":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/429.mcf"
    elif workload == "hmmer":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/456.hmmer"    
    elif workload == "namd":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/508.namd" 
    elif workload == "parest":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/510.parest" 
    elif workload == "sphinx3":
        benchmark = "cpu2017"
        workload_path = DATA_DIR + benchmark + "/482.sphinx3" 
    elif workload == "ferret":
        benchmark = "parsec"
        workload_path = DATA_DIR + benchmark + "/ferret.trace"             
    elif workload == "freqmine":
        benchmark = "parsec"
        workload_path = DATA_DIR + benchmark + "/freqmine.trace"      
    elif workload == "swaptions":
        benchmark = "parsec"
        workload_path = DATA_DIR + benchmark + "/swaptions.trace" 
    elif workload == "copy":
        benchmark = "stream"
        workload_path = DATA_DIR + benchmark + "/copy.trace" 
    elif workload == "triad":
        benchmark = "stream"
        workload_path = DATA_DIR + benchmark + "/triad.trace"
        
    return workload_path, benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--workload', type=str, required=True, help='workload')
    parser.add_argument('--threadidx', type=str, required=True, help='threadidx')
    args = parser.parse_args()
    
    THREAD_IDX = args.threadidx

    STATS_DIR = os.path.join(STATS_DIR, args.workload, THREAD_IDX)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    workload_path, benchmark = get_workload_path(args.workload)
    if benchmark == "cpu2017":
        SIM_MODE = "cpu"
        
    tune_workload(workload_path)

