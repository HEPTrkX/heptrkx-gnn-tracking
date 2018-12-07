#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30

. scripts/setup.sh
srun -l python ./train.py -d configs/segclf.yaml
