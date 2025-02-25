#!/bin/bash

# Required variables
#workloadset=("swaptions")
#workloadset=("sphinx3" "xz")
#workloadset=("freqmine" "fotonik3d")
#workloadset=("swaptions" "freqmine")
#workloadset=("freqmine" "ferret")
#workloadset=("parest" "GemsFDTD")
#workloadset=("namd" "cactusADM")
#workloadset=("ferret" "GemsFDTD")
#workloadset=("lbm" "zeusmp")
#workloadset=("namd" "sphinx3")
#workloadset=("fotonik3d" "xz")
#workloadset=("lbm" "triad")
workloadset=("copy" "ferret")
#workloadset=("sphinx3" "xz" "fotonik3d" "parest")
#workloadset=("GemsFDTD" "namd" "cactusADM" "swaptions")
#workloadset=("lbm" "zeusmp" "fotonik3d" "namd")
#workloadset=("cactusADM" "hmmer" "parest" "zeusmp")
#workloadset=("sphinx3" "cactusADM" "GemsFDTD" "freqmine")
#workloadset=("hmmer" "xz" "lbm" "swaptions")
#workloadset=("lbm" "copy" "freqmine" "parest")
#workloadset=("ferret" "xz" "namd" "cactusADM")
#workloadset=("swaptions" "namd" "triad" "parest")
#workloadset=("lbm" "parest" "copy" "cactusADM")
#workloadset=("sphinx3" "xz" "fotonik3d" "parest" "GemsFDTD" "namd" "hmmer" "zeusmp")
#workloadset=("ferret" "triad" "copy" "zeusmp" "fotonik3d" "namd" "hmmer" "sphinx3")
#workloadset=("hmmer" "parest" "zeusmp" "sphinx3" "cactusADM" "GemsFDTD" "fotonik3d" "ferret")
#workloadset=("hmmer" "xz" "lbm" "swaptions" "ferret" "zeusmp" "namd" "freqmine")
#workloadset=("namd" "copy" "xz" "fotonik3d" "ferret" "cactusADM" "triad" "freqmine")
#workloadset=("zeusmp" "parest" "sphinx3" "namd" "ferret" "copy" "triad" "lbm")
#workloadset=("fotonik3d" "copy" "zeusmp" "triad" "freqmine" "cactusADM" "sphinx3" "swaptions")
#workloadset=("copy" "fotonik3d" "sphinx3" "swaptions" "hmmer" "zeusmp" "freqmine" "triad")
threadidx=1204
is_bc=True #default: False
only_bc=True #default: Falsex
is_start_expbuffer=True #default: False
base_ckpt_dir="/home/user/acme/checkpoint/bc/online_bc/threadidx${threadidx}"
num_steps=2000

# Hyperparameters
batch_size=10 # defualt:128
sgd_epoch=32 # default:32
rp_bufsize=10 # default:10
is_norm_adv=True
value_cost=1
bc_cost=5e-1 #default:1
end_explore_updateidx=1000000

# Save shell
cp incre_marl_mult.sh shell_log/rl/thread$threadidx.sh

if [ -f "expert_data/thread/Thread=${threadidx}.pkl" ]; then
    echo "File expert_data/thread/Thread=${threadidx}.pkl exists. Exiting..."
    exit 0
fi

for workload_idx in "${!workloadset[@]}"; do
    workload=${workloadset[$workload_idx]}
    
    if [ "$workload_idx" -eq 0 ]; then
        python3 train_multiagent.py --is_update_buffer \
            --num_steps "$num_steps" --workload "$workload" \
            --threadidx "$threadidx" --sgd_epoch "$sgd_epoch" \
            --checkpoint_dir "$base_ckpt_dir" \
            --batch_size "$batch_size" --rp_buf_size "$rp_bufsize" --is_start_exp "$is_start_expbuffer" \
            --is_bc --summarydir "logs/Thread_${threadidx}_train" \
            --is_norm_adv "$is_norm_adv" --value_cost "$value_cost" \
            --bc_cost "$bc_cost" --end_explore_updateidx "$end_explore_updateidx" || exit 1
    else
        python3 train_multiagent.py --is_update_buffer \
            --num_steps "$num_steps" --workload "$workload" \
            --threadidx "$threadidx" --sgd_epoch "$sgd_epoch" \
            --load_checkpoint --checkpoint_dir "$base_ckpt_dir/checkpoints/learner" \
            --batch_size "$batch_size" --rp_buf_size "$rp_bufsize" --is_start_exp "$is_start_expbuffer" \
            --is_bc --summarydir "logs/Thread_${threadidx}_train" \
            --is_norm_adv "$is_norm_adv" --value_cost "$value_cost" \
            --bc_cost "$bc_cost" --end_explore_updateidx "$end_explore_updateidx" || exit 1
    fi

    # Replay
    python3 save_expdata.py --threadidx $threadidx || exit 1 
    python3 train_multiagent.py --is_update_buffer \
    --num_steps 100 --workload "swaptions" \
    --threadidx $threadidx --sgd_epoch 1000 \
    --load_checkpoint --checkpoint_dir "$base_ckpt_dir/checkpoints/learner" \
    --batch_size 10 --rp_buf_size $rp_bufsize --is_start_exp $is_start_expbuffer \
    --is_bc --summarydir logs/Thread_${threadidx}_train/bc_${workload_idx} \
    --only_bc $only_bc --is_norm_adv $is_norm_adv --value_cost $value_cost \
    --bc_cost $bc_cost --end_explore_updateidx 0 || exit 1

done
