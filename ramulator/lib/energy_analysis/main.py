import argparse
import subprocess
import shutil
import os
from pathlib import Path

HOME_DIR = "/home/user/ramulator/"
DATA_DIR = "/mnt/nvme3n1/benchmark/"
STATS_DIR = os.path.join(HOME_DIR, "stats", "record_cmd")
SIM_MODE = "dram"
OPTION_PATH = "config.txt"
CONFIG_NAME = None

def execute_sim(workload_path):
    config_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', f'record_cmd-config.cfg')
    stat_path = os.path.join(STATS_DIR, f"sim.stat")
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
    i_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', 'record_cmd.cfg')
    o_path = os.path.join(HOME_DIR, 'configs', 'ours', 'DDR4', 'record_cmd-config.cfg')
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

def get_config_all():
    default = ["open", "row_bank_col", 16, 16, 39, 6, 468000]
    cofigs = []
    if CONFIG_NAME == "refi":
        #possible_values = list(range(4680, 468000, 234000))
        possible_values = [4680, 234000, 468000]
        for refi_value in possible_values:
            cofigs.append(["open", "row_bank_col", 16, 16, 39, 6, refi_value])
    elif CONFIG_NAME == "rcd":
        possible_values = [12, 14, 16]
        for rcd_value in possible_values:
            cofigs.append(["open", "row_bank_col", rcd_value, 16, 39, 6, 468000])
    elif CONFIG_NAME == "rp":
        possible_values = [12, 14, 16]
        for rp_value in possible_values:
            cofigs.append(["open", "row_bank_col", 16, rp_value, 39, 6, 468000])
    elif CONFIG_NAME == "ras":
        possible_values = [19, 25, 30, 35, 39]
        for ras_value in possible_values:
            cofigs.append(["open", "row_bank_col", 16, 16, ras_value, 6, 468000])
    elif CONFIG_NAME == "rrd":
        possible_values = [3, 5, 6]
        for rrd_value in possible_values:
            cofigs.append(["open", "row_bank_col", 16, 16, 39, rrd_value, 468000])  
    elif CONFIG_NAME == "pagepolicy":
        possible_values = ["open", "closed"]
        for pp_value in possible_values:
            cofigs.append([pp_value, "row_bank_col", 16, 16, 39, 6, 468000])  
    elif CONFIG_NAME == "addressmapping":
        possible_values = ["row_bank_col", "bank_row_col", "permutation"]
        for am_value in possible_values:
            cofigs.append(["open", am_value, 16, 16, 39, 6, 468000])  
    
    return cofigs


CMDIR = "cmd"

def save_cmdtrace(config_idx):
    src = "./cmd-trace-chan-0-rank-0.cmdtrace"
    dst_dir = os.path.join(CMDIR, CONFIG_NAME)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{config_idx}.cmdtrace")
    shutil.move(src, dst)
    
def tune_workload(workload_path):
    configs = get_config_all()
    for config_idx, config in enumerate(configs):
        tune(config, workload_path)
        save_cmdtrace(config_idx)

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
    parser.add_argument('--config', type=str, required=True, help='configs')
    args = parser.parse_args()
    STATS_DIR = os.path.join(STATS_DIR, args.workload)
    os.makedirs(STATS_DIR, exist_ok=True)
    CONFIG_NAME = args.config
    workload_path, benchmark = get_workload_path(args.workload)
    if benchmark == "cpu2017":
        SIM_MODE = "cpu"
        
    tune_workload(workload_path)

