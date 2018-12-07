#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30

. scripts/setup.sh
config=configs/prep_med.yaml

# Loop over tasks (1 per node) and submit
i=0
while [ $i -lt $SLURM_JOB_NUM_NODES ]; do
    echo "Launching task $i"
    srun -N 1 python prepare.py \
        --n-workers 32 --task $i --n-tasks $SLURM_JOB_NUM_NODES $config &
    let i=i+1
done
wait
