#!/bin/bash

runtime=3600 #seconds
device="/dev/nvme0n1"
profiling_home="./profiling_tools/"
tuning_home="./tuning_tools/"
exit_status=$?

# Tuning
sudo bash "${tuning_home}default_tuned_config.sh"

if [ $exit_status -eq 124 ]; then
  exit_status=0
elif [ $exit_status -ne 0 ]; then
  echo "Error: Tuning failed."
  exit 1
fi

echo "Successfully tuning"

# Profiling
sudo timeout $runtime bash "${profiling_home}profiling.sh" $device > "${profiling_home}results/profiling"

if [ $exit_status -eq 124 ]; then
  exit_status=0
elif [ $exit_status -ne 0 ]; then
  echo "Error: Profiling failed."
  exit 1
fi

echo "Successfully profiling"

# Reformat the output
in_path="${profiling_home}results/profiling"
sudo bash "${profiling_home}reformat.sh" $in_path > "${profiling_home}results/performance_log.csv"

if [ $? -ne 0 ]; then
  echo "Error: Reformating failed."
  exit 1
fi

echo "Successfully reformating"

# Save performance log
in_path="${profiling_home}results/performance_log.csv"
o_path="${profiling_home}log/performance_log$(date +'%Y-%m-%d_%H-%M-%S').csv"

cp "$in_path" "$o_path"
echo "Successfully save"