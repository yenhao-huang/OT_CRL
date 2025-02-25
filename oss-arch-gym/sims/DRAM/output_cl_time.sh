#!/bin/bash

# Required variables
#workloadset=("triad" "freqmine" "zeusmp" "hmmer" "swaptions" "sphinx3" "fotonik3d" "copy")
workloadset=("freqmine" "freqmine" "freqmine" "freqmine" "freqmine" "freqmine" "freqmine" "freqmine" "freqmine" "freqmine" "freqmine")
threadidx=1467
is_bc=True #default: False
only_bc=True #default: Falsex
is_start_expbuffer=True #default: False
base_ckpt_dir="/home/user/acme/checkpoint/bc/online_bc/threadidx${threadidx}"
num_steps=100

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
    start_time=$(date +%s)
    python3 save_expdata.py --threadidx $threadidx || exit 1 
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Time for Recording: ${elapsed} seconds"

    start_time=$(date +%s)
    python3 train_multiagent.py --is_update_buffer \
    --num_steps 100 --workload "swaptions" \
    --threadidx $threadidx --sgd_epoch 1000 \
    --load_checkpoint --checkpoint_dir "$base_ckpt_dir/checkpoints/learner" \
    --batch_size 10 --rp_buf_size $rp_bufsize --is_start_exp $is_start_expbuffer \
    --is_bc --summarydir logs/Thread_${threadidx}_train/bc_${workload_idx} \
    --only_bc $only_bc --is_norm_adv $is_norm_adv --value_cost $value_cost \
    --bc_cost $bc_cost --end_explore_updateidx 0 || exit 1
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Time for cloning: ${elapsed} seconds"
    
done
