#!/bin/bash

# Required variables
#workloadset=("swaptions")
#workloadset=("sphinx3" "xz")
#workloadset=("xz" "sphinx3")
#workloadset=("freqmine" "fotonik3d")
#workloadset=("fotonik3d" "freqmine")
#workloadset=("swaptions" "freqmine")
#workloadset=("freqmine" "swaptions")
#workloadset=("freqmine" "ferret")
#workloadset=("ferret" "freqmine")
#workloadset=("parest" "GemsFDTD")
#workloadset=("GemsFDTD" "parest")
#workloadset=("namd" "cactusADM")
#workloadset=("cactusADM" "namd")
#workloadset=("ferret" "GemsFDTD")
#workloadset=("GemsFDTD" "ferret")
#workloadset=("lbm" "zeusmp")
#workloadset=("zeusmp" "lbm")
#workloadset=("namd" "sphinx3")
#workloadset=("sphinx3" "namd")
#workloadset=("fotonik3d" "xz")
#workloadset=("xz" "fotonik3d")
#workloadset=("lbm" "triad")
#workloadset=("triad" "lbm")
#workloadset=("copy" "ferret")
#workloadset=("ferret" "copy")
#workloadset=("sphinx3" "xz" "fotonik3d" "parest")
workloadset=("parest" "fotonik3d" "xz" "sphinx3")
#workloadset=("GemsFDTD" "namd" "cactusADM" "swaptions")
workloadset=("swaptions" "cactusADM" "namd" "GemsFDTD")
#workloadset=("lbm" "zeusmp" "fotonik3d" "namd")
workloadset=("namd" "fotonik3d" "zeusmp" "lbm")
#workloadset=("cactusADM" "hmmer" "parest" "zeusmp")
workloadset=("zeusmp" "parest" "hmmer" "cactusADM")
#workloadset=("sphinx3" "cactusADM" "GemsFDTD" "freqmine")
workloadset=("freqmine" "GemsFDTD" "cactusADM" "sphinx3")
#workloadset=("hmmer" "xz" "lbm" "swaptions")
workloadset=("swaptions" "lbm" "xz" "hmmer")
#workloadset=("lbm" "copy" "freqmine" "parest")
workloadset=("parest" "freqmine" "copy" "lbm")
#workloadset=("ferret" "xz" "namd" "cactusADM")
workloadset=("cactusADM" "namd" "xz" "ferret")
#workloadset=("swaptions" "namd" "triad" "parest")
workloadset=("parest" "triad" "namd" "swaptions")
#workloadset=("lbm" "parest" "copy" "cactusADM")
workloadset=("cactusADM" "copy" "parest" "lbm")
#workloadset=("sphinx3" "xz" "fotonik3d" "parest" "GemsFDTD" "namd" "hmmer" "zeusmp")
#workloadset=("zeusmp" "hmmer" "namd" "GemsFDTD" "parest" "fotonik3d" "xz" "sphinx3")
workloadset=("ferret" "triad" "copy" "zeusmp" "fotonik3d" "namd" "hmmer" "sphinx3")
#workloadset=("sphinx3" "hmmer" "namd" "fotonik3d" "zeusmp" "copy" "triad" "ferret")
workloadset=("hmmer" "parest" "zeusmp" "sphinx3" "cactusADM" "GemsFDTD" "fotonik3d" "ferret")
#workloadset=("ferret" "fotonik3d" "GemsFDTD" "cactusADM" "sphinx3" "zeusmp" "parest" "hmmer")
#workloadset=("hmmer" "xz" "lbm" "swaptions" "ferret" "zeusmp" "namd" "freqmine")
#workloadset=("freqmine" "namd" "zeusmp" "ferret" "swaptions" "lbm" "xz" "hmmer")
#workloadset=("namd" "copy" "xz" "fotonik3d" "ferret" "cactusADM" "triad" "freqmine")
#workloadset=("freqmine" "triad" "cactusADM" "ferret" "fotonik3d" "xz" "copy" "namd")
#workloadset=("zeusmp" "parest" "sphinx3" "namd" "ferret" "copy" "triad" "lbm")
#workloadset=("lbm" "triad" "copy" "ferret" "namd" "sphinx3" "parest" "zeusmp")
#workloadset=("fotonik3d" "copy" "zeusmp" "triad" "freqmine" "cactusADM" "sphinx3" "swaptions")
#workloadset=("swaptions" "sphinx3" "cactusADM" "freqmine" "triad" "zeusmp" "copy" "fotonik3d")
#workloadset=("copy" "fotonik3d" "sphinx3" "swaptions" "hmmer" "zeusmp" "freqmine" "triad")
#workloadset=("triad" "freqmine" "zeusmp" "hmmer" "swaptions" "sphinx3" "fotonik3d" "copy")
threadidx=1451
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
