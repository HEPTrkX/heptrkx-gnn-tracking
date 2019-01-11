#!/bin/bash
#SBATCH -J segclf-big
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

mkdir -p logs
. scripts/setup.sh

srun -l python ./train.py -v -d configs/segclf_big.yaml
