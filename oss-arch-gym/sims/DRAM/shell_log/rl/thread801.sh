#workload=parest
CHECKPOINT_DIR="/home/user/acme/20240923-191721"
ckpt_threadidx=800
#CHECKPOINT_DIR="/home/user/acme/checkpoint/bc/online_bc/threadidx${ckpt_threadidx}"

#workloadset=("sphinx3" "xz")
#workloadset=("freqmine" "fotonik3d")
#workloadset=("swaptions" "freqmine")
workloadset=("freqmine" "ferret")
#workloadset=("parest" "GemsFDTD")
#workloadset=("namd" "cactusADM")
#workloadset=("ferret" "GemsFDTD")
#workloadset=("lbm" "zeusmp")
#workloadset=("sphinx3" "xz" "fotonik3d" "parest")
#workloadset=("GemsFDTD" "namd" "cactusADM" "swaptions")
#workloadset=("lbm" "zeusmp" "fotonik3d" "namd")
#workloadset=("cactusADM" "hmmer" "parest" "zeusmp")
#workloadset=("sphinx3" "cactusADM" "GemsFDTD" "freqmine")
#workloadset=("hmmer" "xz" "lbm" "swaptions")
#workloadset=("hmmer" "xz" "lbm" "swaptions" "ferret" "zeusmp" "namd" "freqmine")
#workloadset=("ferret" "GemsFDTD" "lbm" "zeusmp" "fotonik3d" "namd" "hmmer" "sphinx3")
#workloadset=("hmmer" "xz" "lbm" "swaptions" "ferret" "zeusmp" "namd" "freqmine" "sphinx3")
#workloadset=("parest" "fotonik3d" "GemsFDTD" "sphinx3" "triad" "copy" "cactusADM")
#workloadset=("parest" "triad" "cactusADM" "copy" "xz" "swaptions" "freqmine")

#workloadset=("freqmine" "ferret" "hmmer" "zeusmp" "xz" "lbm" "swaptions" "namd" "parest" "fotonik3d" "GemsFDTD" "sphinx3" "triad" "copy" "cactusADM")

#workloadset=("parest")
threadidx=801

cp inference_marl.sh shell_log/rl/thread$threadidx.sh

mkdir -p logs/Thread_${threadidx}

for workload in "${workloadset[@]}"; do
    # w/o learning
    #python3 train_multiagent.py --only_acting --eval_episodes 1 --workload $workload --threadidx $threadidx
    # w learning
    python3 train_multiagent.py --only_acting --eval_episodes 1 \
    --workload $workload --threadidx $threadidx \
    --checkpoint_dir $CHECKPOINT_DIR/checkpoints/learner --load_checkpoint \
    --expbuf_threadidx $ckpt_threadidx \
    --summarydir logs/Thread_${threadidx}/${workload}/ || exit 1
done