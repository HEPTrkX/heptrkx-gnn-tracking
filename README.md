# Graph Neural Networks for particle track reconstruction

This repository contains the PyTorch implementation of the GNNs for particle
track reconstruction from CTD 2018: https://arxiv.org/abs/1810.06111.

## Contents

The main python scripts for running:
- *[prepare.py](prepare.py)*: the data preparation script which reads
TrackML data files, cleans and reduces the data, and writes hit graphs to
the filesystem.
- *[train.py](train.py)*: the main training script which is steered by
configuration file and loads the data, model, and trainer, and invokes
the trainer to train the model.

Other stuff:
- In the scripts directory are SLURM batch scripts for running the jobs
on Cori at NERSC.
- The GNN model code lives in [models/gnn.py](models/gnn.py).
- The dataset code for reading the prepared hit graphs lives in
[datasets/hitgraphs.py](datasets/hitgraphs.py).
- The main trainer code for the GNN segment classifier lives in
[trainers/gnn.py](trainers/gnn.py).
