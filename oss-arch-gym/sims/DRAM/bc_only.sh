#!/bin/bash

# Required variables
threadidx=804
only_bc=True #default: False
is_start_expbuffer=False #default: False
exp_workload="None"
#exp_workload="lbm,zeusmp,fotonik3d,namd"
#exp_workload="GemsFDTD,namd,cactusADM,swaptions"
#exp_workload="hmmer,xz,lbm,swaptions,ferret,zeusmp,namd,freqmine"
#exp_workload="ferret,GemsFDTD,lbm,zeusmp,fotonik3d,namd,hmmer,sphinx3"
base_ckpt_dir="/home/user/acme/checkpoint/bc/online_bc/threadidx${threadidx}"
num_steps=100

# Hyperparameters
batch_size=10 # defualt:128
sgd_epoch=5000 # default:32
rp_bufsize=10 # default:100000
is_norm_adv=True
value_cost=1
bc_cost=1e-2 #default:1
end_explore_updateidx=0

# Save shell
cp bc_only.sh shell_log/rl/thread$threadidx.sh

python3 train_multiagent.py --is_update_buffer \
    --num_steps $num_steps --workload "swaptions" \
    --threadidx $threadidx --sgd_epoch $sgd_epoch \
    --batch_size $batch_size --is_bc --exp_workloads $exp_workload --rp_buf_size $rp_bufsize --is_start_exp $is_start_expbuffer \
    --summarydir logs/Thread_${threadidx}_train/bc_${workload_idx} \
    --only_bc $only_bc --is_norm_adv $is_norm_adv --value_cost $value_cost \
    --bc_cost $bc_cost --end_explore_updateidx $end_explore_updateidx || exit 1
