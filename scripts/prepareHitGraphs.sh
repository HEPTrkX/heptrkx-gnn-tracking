#!/bin/bash
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30

. scripts/setup.sh
python prepare.py --n-workers 16 configs/prepare_trackml.yaml
