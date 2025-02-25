import argparse
import os
    
def modify_cfg_file_canny(i, n_groups):
    # 读取 cfg 文件内容
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ddr4-example-canny.json', 'r') as file:
        filedata = file.read()

    # 替换占位符 {input1} 为实际的值
    new_data = filedata.replace('{workload_name}', f"dramsys_input_{i}")
    new_data = new_data.replace('{n_groups}', str(n_groups))
    
    path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny_groups={}".format(n_groups)
    os.makedirs(path, exist_ok=True)
    # 将修改后的内容写回 cfg 文件
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny_groups={}/ddr4-{}.json'.format(n_groups, i), 'w') as file:
        file.write(new_data)

def modify_cfg_file_canny_replay(i, n_groups):
    # 读取 cfg 文件内容
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ddr4-example-canny_replay.json', 'r') as file:
        filedata = file.read()

    # 替换占位符 {input1} 为实际的值
    new_data = filedata.replace('{workload_name}', f"dramsys_input_{i}_replay")
    new_data = new_data.replace('{n_groups}', str(n_groups))
    
    path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny_groups={}_replay".format(n_groups)
    os.makedirs(path, exist_ok=True)
    # 将修改后的内容写回 cfg 文件
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny_groups={}_replay/ddr4-{}.json'.format(n_groups, i), 'w') as file:
        file.write(new_data)

def modify_cfg_file_blackscholes(i):

    # 读取 cfg 文件内容
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ddr4-example-blackscholes.json', 'r') as file:
        filedata = file.read()

    # 替换占位符 {input1} 为实际的值
    new_data = filedata.replace('{workload_name}', f"dramsys_input_{i}")

    # 将修改后的内容写回 cfg 文件
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/blackscholes/ddr4-{}.json'.format(i), 'w') as file:
        file.write(new_data)

def main():
    parser = argparse.ArgumentParser(description='Example script to demonstrate the use of argparse.')
    parser.add_argument('--n_groups', required=True, type=int, help='Number of groups')
    parser.add_argument('--workload', required=True, type=str, help='Workload')
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    workload = args.workload
    if workload == "canny":
        n_groups = args.n_groups
        for i in range(n_groups):
            modify_cfg_file_canny(i, n_groups)
            modify_cfg_file_canny_replay(i, n_groups)
            
        modify_cfg_file_canny("all", n_groups)
        modify_cfg_file_canny_replay("all", n_groups)
    elif workload == "blackscholes":
        n_groups = args.n_groups
        for i in range(n_groups):
            modify_cfg_file_blackscholes(i)
            
        modify_cfg_file_blackscholes("all")

if __name__ == "__main__":
    main()