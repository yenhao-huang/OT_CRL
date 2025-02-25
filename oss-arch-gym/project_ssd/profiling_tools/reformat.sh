#!/bin/bash

# Print CSV header
echo "Device,r/s,rkB/s,rrqm/s,%rrqm,r_await,rareq-sz,w/s,wkB/s,wrqm/s,%wrqm,w_await,wareq-sz,d/s,dkB/s,drqm/s,%drqm,d_await,dareq-sz,aqu-sz,%util"

# Process each line in the input
awk '/Device/ { getline; printf "%s", $1; for (i=2; i<=NF; i++) printf ",%s", $i; print "" }' < $1

