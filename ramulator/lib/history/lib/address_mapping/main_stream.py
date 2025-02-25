import subprocess
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--config_idx', type=int, required=True, help='Workload')
    args = parser.parse_args()
    
    # 设置变量
    HOME_DIR = "/home/user/ramulator/"
    benchmark = "stream"

    workloads_dir = f"{HOME_DIR}../zsim_ramulator/ramulator-pim/zsim-ramulator/shell/share/mem_trace/ramulator/{benchmark}/"
    workload_name1 = os.listdir(workloads_dir)
    workloads = [os.path.join(workloads_dir, workload_name) for workload_name in workload_name1]
    results_dir = f"results/{benchmark}/{args.config_idx}"

    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)

    # 遍历所有工作负载文件
    for workload_path in workloads:
        workload = workload_path.split("/")[-1]

        if os.path.isfile(workload_path):
            # 构建ramulator命令
            ramulator_cmd = [
                os.path.join(HOME_DIR, "ramulator"),
                os.path.join(HOME_DIR, "configs/DDR3-config.cfg"),
                "--mode=dram",
                workload_path
            ]
            
            # 执行ramulator命令
            subprocess.run(ramulator_cmd)
            
            # 处理DDR3.stats文件
            with open("DDR3.stats", "r") as stats_file:
                dram_cycles = [line for line in stats_file if "dram_cycles" in line]
            
            # 写入结果文件
            
            output_file = os.path.join(results_dir, f"{workload}.stl")
            with open(output_file, "w") as result_file:
                result_file.writelines(dram_cycles)
