#!/bin/bash
workloadset=("swaptions")
threadidx=1308
num_steps=2000

# Hyperparameters
seed=64 # default:1

# Save shell
cp single_agent.sh shell_log/rl/thread$threadidx.sh

for workload in "${workloadset[@]}"; do
    python3 train_single_agent.py --workload "$workload" \
    --threadidx $threadidx --num_steps $num_steps --seed $seed
done
