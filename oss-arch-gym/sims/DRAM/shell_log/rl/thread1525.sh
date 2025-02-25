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
#workloadset=("parest" "fotonik3d" "xz" "sphinx3")
#workloadset=("GemsFDTD" "namd" "cactusADM" "swaptions")
#workloadset=("swaptions" "cactusADM" "namd" "GemsFDTD")
#workloadset=("lbm" "zeusmp" "fotonik3d" "namd")
#workloadset=("namd" "fotonik3d" "zeusmp" "lbm")
#workloadset=("cactusADM" "hmmer" "parest" "zeusmp")
#workloadset=("zeusmp" "parest" "hmmer" "cactusADM")
#workloadset=("sphinx3" "cactusADM" "GemsFDTD" "freqmine")
#workloadset=("freqmine" "GemsFDTD" "cactusADM" "sphinx3")
#workloadset=("hmmer" "xz" "lbm" "swaptions")
#workloadset=("swaptions" "lbm" "xz" "hmmer")
#workloadset=("lbm" "copy" "freqmine" "parest")
#workloadset=("parest" "freqmine" "copy" "lbm")
#workloadset=("ferret" "xz" "namd" "cactusADM")
#workloadset=("cactusADM" "namd" "xz" "ferret")
#workloadset=("swaptions" "namd" "triad" "parest")
#workloadset=("parest" "triad" "namd" "swaptions")
#workloadset=("lbm" "parest" "copy" "cactusADM")
#workloadset=("cactusADM" "copy" "parest" "lbm")
# workloadset=("cactusADM" "hmmer" "parest" "zeusmp")
#workloadset=("sphinx3" "xz" "fotonik3d" "parest" "GemsFDTD" "namd" "hmmer" "zeusmp")
#workloadset=("zeusmp" "hmmer" "namd" "GemsFDTD" "parest" "fotonik3d" "xz" "sphinx3")
#workloadset=("ferret" "triad" "copy" "zeusmp" "fotonik3d" "namd" "hmmer" "sphinx3")
#workloadset=("sphinx3" "hmmer" "namd" "fotonik3d" "zeusmp" "copy" "triad" "ferret")
#workloadset=("hmmer" "parest" "zeusmp" "sphinx3" "cactusADM" "GemsFDTD" "fotonik3d" "ferret")
#workloadset=("ferret" "fotonik3d" "GemsFDTD" "cactusADM" "sphinx3" "zeusmp" "parest" "hmmer")
#workloadset=("hmmer" "xz" "lbm" "swaptions" "ferret" "zeusmp" "namd" "freqmine")
#workloadset=("freqmine" "namd" "zeusmp" "ferret" "swaptions" "lbm" "xz" "hmmer")
workloadset=("namd" "copy" "xz" "fotonik3d" "ferret" "cactusADM" "triad" "freqmine")
#workloadset=("freqmine" "triad" "cactusADM" "ferret" "fotonik3d" "xz" "copy" "namd")
#workloadset=("zeusmp" "parest" "sphinx3" "namd" "ferret" "copy" "triad" "lbm")
#workloadset=("lbm" "triad" "copy" "ferret" "namd" "sphinx3" "parest" "zeusmp")
#workloadset=("fotonik3d" "copy" "zeusmp" "triad" "freqmine" "cactusADM" "sphinx3" "swaptions")
#workloadset=("swaptions" "sphinx3" "cactusADM" "freqmine" "triad" "zeusmp" "copy" "fotonik3d")
#workloadset=("copy" "fotonik3d" "sphinx3" "swaptions" "hmmer" "zeusmp" "freqmine" "triad")
#workloadset=("triad" "freqmine" "zeusmp" "hmmer" "swaptions" "sphinx3" "fotonik3d" "copy")
#workloadset=("freqmine" "ferret" "hmmer" "zeusmp" "xz" "lbm" "swaptions" "namd" "parest" "fotonik3d" "GemsFDTD" "sphinx3" "triad" "copy" "cactusADM")
#last_workload=${workloadset[-1]}
#CHECKPOINT_DIR="/home/user/acme/checkpoint/singleworkload/$last_workload"
ckpt_threadidx=1517
CHECKPOINT_DIR="/home/user/acme/checkpoint/bc/online_bc/threadidx${ckpt_threadidx}"
threadidx=1525

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