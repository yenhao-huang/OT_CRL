import subprocess
import os
import argparse

HOME_DIR = "/home/user/ramulator/"
DATA_DIR = "/mnt/nvme3n1/benchmark/"
STATS_DIR = (HOME_DIR+'stats/')
MODE = "dram"
def execute_sim(workload):
        if os.path.isfile(workload):
            # 构建ramulator命令
            ramulator_cmd = [
                os.path.join(HOME_DIR, "ramulator"),
                os.path.join(HOME_DIR, "configs/ours/DDR3-config.cfg"),
                f"--mode={MODE}",
                workload
            ]
            
            # 执行ramulator命令
            subprocess.run(ramulator_cmd)
        else:
            raise "No this workload"

def tune(workload):
    execute_sim(workload)


def get_workload_set(benchmark_name):
    workload_dir = f"{DATA_DIR}{benchmark_name}/"
    workloads = [os.path.join(workload_dir, workload_name) for workload_name in os.listdir(workload_dir)]
    return workloads

if __name__ == "__main__":
    '''
    E.g., python3 main.py --benchmark comm
    '''
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--benchmark', type=str, required=True, help='benchmark')
    parser.add_argument('--workload_name', type=str, required=True, help='workload_name')
    args = parser.parse_args()
    if args.benchmark == "cpu2017":
        MODE = "cpu"
        
    workload = os.path.join(DATA_DIR, args.benchmark, args.workload_name)
    tune(workload)