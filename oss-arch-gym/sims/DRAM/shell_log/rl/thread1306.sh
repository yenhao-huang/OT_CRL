#!/bin/bash

#workloadset=("ferret")
#workloadset=("freqmine")
#workloadset=("swaptions")
#workloadset=("copy")
#workloadset=("triad")
#workloadset=("namd")
#workloadset=("parest")
#workloadset=("fotonik3d")
workloadset=("xz")
#workloadset=("zeusmp")
#workloadset=("cactusADM")
#workloadset=("hmmer")
#workloadset=("GemsFDTD")
#workloadset=("lbm")
#workloadset=("sphinx3")
threadidx=1306
num_steps=2000

# Hyperparameters
seed=57 # default:1

# Save shell
cp single_agent.sh shell_log/rl/thread$threadidx.sh

for workload in "${workloadset[@]}"; do
    python3 train_single_agent.py --workload "$workload" \
    --threadidx $threadidx --num_steps $num_steps --seed $seed
done
