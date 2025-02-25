import argparse
from pathlib import Path

#4G
DRAM_CAPACITY = 4294967295
BASE_ADDR = 536870912

def convert_to_dramsysformat_total(data):
    
    new_lines = []
    prev_addr = 0
    for j, line in enumerate(data):
        parts = line.split()
        operation = parts[1]
        ori_addr = int(parts[2], 16)
        offset = ori_addr - prev_addr
        prev_addr = ori_addr
        if j == 0:
            continue

        if abs(offset) > BASE_ADDR:
            addr = hex(BASE_ADDR * 2)
        else:
            addr = hex(BASE_ADDR + offset)
        
        new_line = "{}:\t{}\t{}\n".format(j, operation, addr)
        new_lines.append(new_line)
    
    return new_lines

if __name__ == "__main__":
    
    '''
    for example, python reformat_to_dramsys_form.py mem_trace/canny_0
    '''
    parser = argparse.ArgumentParser(description="Convert fifth column of input data to hexadecimal.")
    parser.add_argument("--memtrace_path", type=str, help="Input data string, separated by newlines.")
    args = parser.parse_args()
    
    with Path(args.memtrace_path).open("r") as f:
        data = f.readlines()
    
    # record all data
    new_data = convert_to_dramsysformat_total(data)
    o_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/traces/ours/object_detection/object_detection_full.stl"
    with open(o_path, "w") as f:
        f.writelines(new_data)