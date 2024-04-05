#!/bin/bash

AGENT=$1
N=$2

for i in $(seq 1 $N); do
  # start wandb agent
  sbatch run_sweep.job $AGENT
done
