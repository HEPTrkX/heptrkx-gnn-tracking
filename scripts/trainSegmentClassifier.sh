#!/bin/bash
#SBATCH -J segclf-med
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

mkdir -p logs
. scripts/setup.sh

srun -l python ./train.py -d configs/segclf_med.yaml
