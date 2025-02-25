import argparse
from expdata import DRAM_ExpBuffer
from pathlib import Path
from train_multiagent import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process expert data")
    parser.add_argument('--threadidx', type=int, required=True, help="Index of the thread")
    args = parser.parse_args()
    
    with open(f"expert_data/metadata/{args.threadidx}.txt", "r") as f:
        expertdata_path, workload, log_dir = f.readlines()[0].split(",")
    
    exp_buffer = DRAM_ExpBuffer(expertdata_path, workload, log_dir)
    exp_buffer.save_expertdata()
    