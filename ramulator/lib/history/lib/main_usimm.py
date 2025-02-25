import subprocess
import os

if __name__ == "__main__":
    # 设置变量
    HOME_DIR = "/home/user/ramulator/"
    benchmark = "usimm"

    workloads_dir = f"/mnt/nvme3n1/benchmark/{benchmark}/jwac_msc_workloads_first18_workloads/"
    workload_name1 = os.listdir(workloads_dir)
    workloads1 = [os.path.join(workloads_dir, workload_name) for workload_name in workload_name1]
    workloads_dir = f"/mnt/nvme3n1/benchmark/{benchmark}/jwac_msc_workloads_last_14_workloads/"
    workload_name2 = os.listdir(workloads_dir)
    workloads2 = [os.path.join(workloads_dir, workload_name) for workload_name in workload_name2]
    workloads = workloads1 + workloads2
    results_dir = f"results/{benchmark}"

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
