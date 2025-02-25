# TODO
python3 train_randomsearch_DRAMSys.py --workload GemsFDTD --cost_model buffer --num_steps 10000 --threadidx 1 > /dev/null &
python3 train_randomsearch_DRAMSys.py --workload fotonik3d --cost_model buffer --num_steps 10000 --threadidx 2 > /dev/null &
python3 train_randomsearch_DRAMSys.py --workload hmmer --cost_model buffer --num_steps 10000 --threadidx 3 > /dev/null &
python3 train_randomsearch_DRAMSys.py --workload sphinx3 --cost_model buffer --num_steps 10000 --threadidx 4 > /dev/null &
python3 train_randomsearch_DRAMSys.py --workload ferret --cost_model buffer --num_steps 10000 --threadidx 5 > /dev/null &