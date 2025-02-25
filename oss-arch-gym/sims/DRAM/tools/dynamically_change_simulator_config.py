import argparse

def change_config(i_path, o_path):
    with open(i_path, 'r') as file:
        filedata = file.read()
        
    filedata = filedata.replace('{group_idx}', str(args.group_idx))
        
    with open(o_path, 'w') as file:
        file.write(filedata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--workload', type=str, required=True, help='Workload')
    parser.add_argument('--group_idx', type=int, required=True, help='Configuration index to set in the config file')
    
    args = parser.parse_args()
    workload_name = args.workload
    # start from 1
    if workload_name == "canny_05":
        i_path='/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/others/canny/ratio=0.5/canny_sample.py'
        o_path='/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py'
        change_config(i_path, o_path)
    elif workload_name == "canny_075":
        i_path='/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/others/canny/ratio=0.75/canny_sample.py'
        o_path='/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py'
        change_config(i_path, o_path)
    elif workload_name == "blackscholes_05":
        i_path='/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/parsec/blackscholes/ratio=0.5/sample.py'
        o_path='/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py'
        change_config(i_path, o_path)
    else:
        raise "No this workload!"