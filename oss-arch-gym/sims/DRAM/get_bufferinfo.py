import pickle
from pathlib import Path

def get_bufferinfo(path, workload):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    #print(data[list(data.keys())[0]])
    
    datainfo = {
        "workload" : workload,
        "length" : len(data)
    }
    return datainfo

if __name__ == "__main__":
    workload_all = ["copy", "hmmer", "swaptions", "ferret", "freqmine", "fotonik3d", "GemsFDTD", "namd", "xz", "sphinx3", "cactusADM", "triad", "zeusmp", "lbm", "parest"]
    BUFFER_DIR = "buffer_v6"
    print("For RL")
    for workload in workload_all:
        path = f"{BUFFER_DIR}/merge_for_rl/{workload}.pkl"
        if not Path(path).exists():
            print("No this workload")
            continue
        datainfo = get_bufferinfo(path, workload)
        print(datainfo)

    print("For Random")
    for workload in workload_all:
        path = f"{BUFFER_DIR}/merge_for_rd/{workload}.pkl"
        if not Path(path).exists():
            print("No this workload")
            continue
        datainfo = get_bufferinfo(path, workload)
        print(datainfo)