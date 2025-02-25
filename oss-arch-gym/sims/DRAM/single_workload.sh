# Shell Parameters
# workload="swaptions"
workload="triad"
threadidx=1483
is_bc=True #default: False
only_bc=True #default: Falsex
is_start_expbuffer=True #default: False
base_ckpt_dir="/home/user/acme/checkpoint/bc/online_bc/threadidx${threadidx}"
num_steps=2000

# Model Hyperparameters
batch_size=10 # defualt:128
sgd_epoch=32 # default:32
rp_bufsize=10 # default:10
is_norm_adv=True
value_cost=1
bc_cost=5e-1 #default:1
end_explore_updateidx=1000000

python3 train_multiagent.py --is_update_buffer \
    --num_steps "$num_steps" --workload "$workload" \
    --threadidx "$threadidx" --sgd_epoch "$sgd_epoch" \
    --checkpoint_dir "$base_ckpt_dir" \
    --batch_size "$batch_size" --rp_buf_size "$rp_bufsize" --is_start_exp "$is_start_expbuffer" \
    --is_bc --summarydir "logs/Thread_${threadidx}_train" \
    --is_norm_adv "$is_norm_adv" --value_cost "$value_cost" \
    --bc_cost "$bc_cost" --end_explore_updateidx "$end_explore_updateidx" || exit 1