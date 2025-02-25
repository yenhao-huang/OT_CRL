import argparse
import subprocess
import os

HOME_DIR = "/home/user/ramulator-pim/common/DRAMPower/"
DATA_DIR = "/mnt/nvme3n1/benchmark/drampower/"
SIMSPEC_PATH = HOME_DIR + "memspecs/ours/MICRON_4Gb_DDR4-2400_8bit_A.xml"
STATS_DIR = (HOME_DIR+f'stats/')

def modify_cfg_filemodify_cfg_file(config):
    simspec_sample_path = HOME_DIR + "memspecs/ours/MICRON_4Gb_DDR4-2400_8bit_A_sample.xml"
    # 读取 cfg 文件内容
    with open(simspec_sample_path, "r") as file:
        filedata = file.read()

    filedata = filedata.replace('$RCD_VALUE$', str(config[0]))
    filedata = filedata.replace('$RP_VALUE$', str(config[1]))
    filedata = filedata.replace('$RAS_VALUE$', str(config[2]))
    filedata = filedata.replace('$RRDL_VALUE$', str(config[3]))
    filedata = filedata.replace('$REFI_VALUE$', str(config[4]))

    with open(SIMSPEC_PATH, 'w') as file:
        file.write(filedata)


def execute_simualtor(workload):
    # Command components
    #workload = '/root/ramulator-pim/zsim-ramulator/shell/canny/drampower_mem_trace/n_groups=2/dramsys_input_all.stl'
    trace_path = DATA_DIR + f"{workload}/cmd-trace-chan-0-rank-0.cmdtrace"
    sim_path = HOME_DIR + "drampower"
    command = [
        sim_path,
        '-m', SIMSPEC_PATH,
        '-c', trace_path
    ]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # Check if there was an error
    if result.returncode != 0:
        print("Error in command execution!")
        print("Error:", result.stderr)
        return None

    # Process the output to find the total trace energy and average power
    for line in result.stdout.split('\n'):
        if "Total Trace Energy:" in line:
            energy_value = float(line.split(':')[1].strip().split(' ')[0])
        elif "Average Power:" in line:
            average_power = float(line.split(':')[1].strip().split(' ')[0])
    
    return energy_value, average_power

def write_stat(energy, average_power):
    o_path = os.path.join(STATS_DIR, "sim.stat")
    with open(o_path, "w") as f:
        f.writelines(f"{energy},{average_power}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--workload', type=str, required=True, help='pagepolicy')
    args = parser.parse_args()
    STATS_DIR = os.path.join(STATS_DIR, args.workload)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    # RCD、RP、RAS、RRDL、REFI
    default_config = [16, 16, 39, 6, 4680]
    modify_cfg_file(default_config)
    energy, average_power = execute_simualtor(args.workload)
    #print("Default config, Total Trace Energy: {} pJ".format(energy))
    write_stat(energy, average_power)