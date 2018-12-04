#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30

. scripts/setup.sh
module list
which python
srun -l python ./train.py configs/segclf.yaml
#srun -l python ./train.py -d $@
