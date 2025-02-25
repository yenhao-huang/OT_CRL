import argparse
from pathlib import Path
import pandas as pd

# default * 0.8 * 0.8
def update_target(workload):
    if workload == "swaptions":
        target_latency, target_energy = 5.1, 0.7
    elif workload == "ferret":
        target_latency, target_energy = 118.2, 15.6
    elif workload == "freqmine":
        target_latency, target_energy = 205.0, 27.5
    elif workload == "hmmer":
        target_latency, target_energy = 22765.5, 1961.7
    elif workload == "copy":
        target_latency, target_energy = 3518.3, 594.3
    elif workload == "sphinx3":
        target_latency, target_energy = 32743.3, 3294.7
    elif workload == "namd":
        target_latency, target_energy = 27743.9, 2512.4
    elif workload == "xz":
        target_latency, target_energy = 30487.7, 4022.6
    elif workload == "GemsFDTD":
        target_latency, target_energy = 69334.6, 9777.9
    elif workload == "fotonik3d":
        target_latency, target_energy = 69345.7, 9206.1
    elif workload == "zeusmp":
        target_latency, target_energy = 90955.5, 14942.0
    elif workload == "lbm":
        target_latency, target_energy = 64996.2, 12162.8
    elif workload == "parest":
        target_latency, target_energy = 42506.4, 5674.6
    elif workload == "triad":
        target_latency, target_energy = 4405.8, 823.9
    elif workload == "cactusADM":
        target_latency, target_energy = 33773.1, 3329.5
    else:
        raise "No target value for this workload"
    
    target_edp = target_latency * target_energy
    return target_edp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process expert data")
    parser.add_argument('--threadidx', type=int, required=True, help="Index of the thread")
    args = parser.parse_args()
    with open(f"inference_logs/{args.threadidx}.txt", "r") as f:
        workload, log_dir = f.readlines()[0].split(",")
        
    dir_path = Path(f"{log_dir}/evaluator/")
    if dir_path.exists() and dir_path.is_dir():
        # Get the list of files in the directory
        subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
        path = subdirs[0] / "logs/evaluator/logs.csv"
    else:
        raise "No this log"
    
    # get current data
    df = pd.read_csv(path)
    df["edp"] = df["latency"] * df["energy"]
    select_df = df[df["err_rate"] < 1e-4]
    if select_df["edp"].min() < update_target(workload) * 1.55:
        data = "Good inference"
    else:
        data = "Bad inference"
    
    with open(f"inference_logs/{args.threadidx}_results.txt", "w") as f:
        f.writelines(data)