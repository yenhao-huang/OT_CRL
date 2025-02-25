import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--workload', type=str, required=True, help='Workload')
    parser.add_argument('--config_idx', type=int, help='Configuration index to set in the config file')

    args = parser.parse_args()
    workload_name = args.workload

    if workload_name == "add":
        with open(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/stream/add.py', 'r') as file:
            filedata = file.read()
    elif workload_name == "copy":
        with open(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/stream/copy.py', 'r') as file:
            filedata = file.read()
    elif workload_name == "scale":
        with open(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/stream/scale.py', 'r') as file:
            filedata = file.read()
    elif workload_name == "triad":
        with open(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/stream/triad.py', 'r') as file:
            filedata = file.read()
    elif workload_name == "stream":
        with open(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/synthetic/stream.py', 'r') as file:
            filedata = file.read()
    elif workload_name == "random":
        with open(f'/home/user/Desktop/oss-arch-gym/sims/DRAM/env_spec/synthetic/random.py', 'r') as file:
            filedata = file.read()
    else:
        raise "NO THIS WORKLOAD"
    
    with open('/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py', 'w') as file:
        file.write(filedata)
    