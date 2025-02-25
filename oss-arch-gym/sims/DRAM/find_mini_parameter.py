import json
import re
import subprocess
import sys
import numpy as np  
import os

def get_observation(outstream):
    '''
    converts the std out from DRAMSys to observation of energy, power, latency
    [Energy (PJ), Power (mW), Latency (ns)]
    '''
    obs = []
    
    keywords = ["Total Energy", "Average Power", "Total Time"]

    energy = re.findall(keywords[0],outstream)
    all_lines = outstream.splitlines()
    for each_idx in range(len(all_lines)):
        
        if keywords[0] in all_lines[each_idx]:
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)
        if keywords[1] in all_lines[each_idx]:
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e3)
        if keywords[2] in all_lines[each_idx]:
            obs.append(float(all_lines[each_idx].split(":")[1].split()[0])/1e9)

    obs = np.asarray(obs)
    print('[Environment] Observation:', obs)

    return obs

def tune_config(config, default_value):

    # Write minimum setting
    for i in range(0, default_value, 2):
        print("Tuning", config, "to", str(i))
        with open (mem_spec_file, "r") as JsonFile:
            data = json.load(JsonFile)
        
        data['memspec']['memtimingspec'][config] = i

        with open (mem_spec_file, "w") as JsonFile:
            json.dump(data, JsonFile)

        exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
        config_name = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny/ddr4-0.json"
        try:
            process = subprocess.Popen([exe_path, config_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for the process to complete or timeout after 3 seconds
            process.wait(timeout=3)
            
            out, err = process.communicate()
            if err.decode() == "":
                outstream = out.decode()
            else:
                print(err.decode())
                sys.exit()
            
            obs = get_observation(outstream)
        except subprocess.TimeoutExpired:
            # If the process exceeds the timeout, terminate it
            process.terminate()
            print("Process terminated due to timeout.")
            raise "1"

def tune_default_setting(default_arr):
    with open (mem_spec_file, "r") as JsonFile:
        data = json.load(JsonFile)
    
    data['memspec']['memtimingspec']['CL'] = default_arr[0]
    data['memspec']['memtimingspec']['WL'] = default_arr[1]
    data['memspec']['memtimingspec']['RCD'] = default_arr[2]
    data['memspec']['memtimingspec']['RP'] = default_arr[3]
    data['memspec']['memtimingspec']['RAS'] = default_arr[4]
    data['memspec']['memtimingspec']['RRD_L'] = default_arr[5]
    data['memspec']['memtimingspec']['FAW'] = default_arr[6]
    data['memspec']['memtimingspec']['RFC'] = default_arr[7]    

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(data, JsonFile)

    exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
    config_name = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny/ddr4-0.json"
    try:
        process = subprocess.Popen([exe_path, config_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to complete or timeout after 3 seconds
        process.wait(timeout=3)
        
        out, err = process.communicate()
        if err.decode() == "":
            outstream = out.decode()
        else:
            print(err.decode())
            sys.exit()
        
        obs = get_observation(outstream)
    except subprocess.TimeoutExpired:
        # If the process exceeds the timeout, terminate it
        process.terminate()
        raise "Process terminated due to timeout."

if __name__ == "__main__":

    mem_spec_file = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/resources/configs/memspecs/MICRON_4Gb_DDR4-2400_8bit_A.json"
    #default_arr = [16, 16, 16, 16, 39, 6, 26, 313]
    default_arr = [16, 16, 16, 16, 39, 6, 26, 10]
    config_arr = ["CL", "WL", "RCD", "RP", "RAS", "RRD_L", "FAW", "RFC"]

    '''
    tune_default_setting(default_arr)

    for config, default in zip(config_arr, default_arr):
        tune_config(config, default)      
    '''
  

    '''
    goal: 檢查 error config
    將 RCD 設為 9 以下是會 dump error


    with open (mem_spec_file, "r") as JsonFile:
        data = json.load(JsonFile)

    data['memspec']['memtimingspec']['CL'] = 2
    data['memspec']['memtimingspec']['WL'] = 2
    data['memspec']['memtimingspec']['RCD'] = 2
    data['memspec']['memtimingspec']['RP'] = 2
    data['memspec']['memtimingspec']['RAS'] = 9
    data['memspec']['memtimingspec']['RRD_L'] = 15
    data['memspec']['memtimingspec']['FAW'] = 63
    data['memspec']['memtimingspec']['RFC'] = 10   

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(data, JsonFile)

    exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
    config_name = "/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny/ddr4-0.json"
    try:
        process = subprocess.Popen([exe_path, config_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to complete or timeout after 3 seconds
        process.wait(timeout=3)
        
        out, err = process.communicate()
        if err.decode() == "":
            outstream = out.decode()
        else:
            print(err.decode())
            sys.exit()
        
        obs = get_observation(outstream)
    except subprocess.TimeoutExpired:
        # If the process exceeds the timeout, terminate it
        process.terminate()
        raise "Process terminated due to timeout."
    '''  
    '''tuning
    '''
    with open (mem_spec_file, "r") as JsonFile:
        data = json.load(JsonFile)

    data['memspec']['memtimingspec']['CL'] = 3
    data['memspec']['memtimingspec']['WL'] = 15
    data['memspec']['memtimingspec']['RCD'] = 3
    data['memspec']['memtimingspec']['RP'] = 2
    data['memspec']['memtimingspec']['RAS'] = 32
    data['memspec']['memtimingspec']['RRD_L'] = 7
    data['memspec']['memtimingspec']['FAW'] = 42
    data['memspec']['memtimingspec']['RFC'] = 318 

    with open (mem_spec_file, "w") as JsonFile:
        json.dump(data, JsonFile)
    
    n_groups = 10
    for i in range(n_groups):
        exe_path = "/home/user/Desktop/oss-arch-gym/sims/DRAM/binary/DRAMSys/DRAMSys"
        config_name = f"/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny/ddr4-{i}.json"
        try:
            env = os.environ.copy()
            working_dir = '/home/user/Desktop/oss-arch-gym/sims/DRAM'
            # Run the command and capture the output
            result = subprocess.run(
                [exe_path, config_name],
                cwd=working_dir,
                env=env,
                timeout=2,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
                
            outstream = result.stdout.decode()

            obs = get_observation(outstream)
            obs = obs.reshape(1,3)
        except subprocess.TimeoutExpired:
            raise "Process terminated due to timeout."

