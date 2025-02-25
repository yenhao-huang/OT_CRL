#workload=parest

#workloadset=("zeusmp")
#workloadset=("cactusADM")
#workloadset=("hmmer")
#workloadset=("GemsFDTD")
#workloadset=("lbm")
#workloadset=("sphinx3")
#workloadset=("namd")
#workloadset=("parest")
#workloadset=("fotonik3d")
#workloadset=("xz")
#workloadset=("ferret")
#workloadset=("freqmine")
#workloadset=("swaptions")
#workloadset=("copy")
#workloadset=("triad")
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
#workloadset=("copy" "ferret")
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
workloadset=("namd" "copy" "xz" "fotonik3d" "ferret" "cactusADM" "triad" "freqmine")
#workloadset=("zeusmp" "parest" "sphinx3" "namd" "ferret" "copy" "triad" "lbm")
#workloadset=("fotonik3d" "copy" "zeusmp" "triad" "freqmine" "cactusADM" "sphinx3" "swaptions")
#workloadset=("copy" "fotonik3d" "sphinx3" "swaptions" "hmmer" "zeusmp" "freqmine" "triad")
#workloadset=("freqmine" "ferret" "hmmer" "zeusmp" "xz" "lbm" "swaptions" "namd" "parest" "fotonik3d" "GemsFDTD" "sphinx3" "triad" "copy" "cactusADM")
#last_workload=${workloadset[-1]}
#CHECKPOINT_DIR="/home/user/acme/checkpoint/singleworkload/$last_workload"
ckpt_threadidx=1222
CHECKPOINT_DIR="/home/user/acme/checkpoint/bc/online_bc/threadidx${ckpt_threadidx}"
threadidx=1223

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