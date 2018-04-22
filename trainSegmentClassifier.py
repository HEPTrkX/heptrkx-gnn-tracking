#!/usr/bin/env python

"""
This script is for training the message passing graph neural network
segment classification model.
"""

# System imports
from __future__ import print_function
from __future__ import division
import os
import argparse
import logging
import multiprocessing as mp

# Externals
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split

# Torch imports
import torch
from torch.autograd import Variable
import torch.nn as nn

# Local imports
from graph import load_graphs, SparseGraph, feature_scale, graph_from_sparse
from model import SegmentClassifier
from estimator import Estimator

torch_to_np = lambda x: x.cpu().data.numpy()
np_to_torch = None

def set_cuda(cuda):
    global np_to_torch
    if cuda:
        np_to_torch = lambda x, volatile=False: (
            Variable(torch.from_numpy(x.astype(np.float32)),
                     volatile=volatile).cuda())
    else:
        np_to_torch = lambda x, volatile=False: (
            Variable(torch.from_numpy(x.astype(np.float32)),
                     volatile=volatile))

def parse_args():
    parser = argparse.ArgumentParser('trainTrackFilter.py')
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/global/cscratch1/sd/sfarrell/heptrkx/hit_graphs_mu10_003/data')
            #default='/global/cscratch1/sd/sfarrell/heptrkx/hit_graphs_mu200_000/data')
    add_arg('--output-dir')
    add_arg('--n-samples', type=int, default=1024)
    add_arg('--valid-frac', type=float, default=0.2)
    add_arg('--test-frac', type=float, default=0.2)
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--batch-size', type=int, default=1)
    add_arg('--hidden-dim', type=int, default=8)
    add_arg('--n-iters', type=int, default=1)
    add_arg('--cuda', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def merge_graphs(graphs):
    batch_size = len(graphs)
    
    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension
        return g.X[None], g.Ri[None], g.Ro[None], g.y[None]
    
    # Get the maximum sizes in this batch
    n_features = graphs[0].X.shape[1]
    n_nodes = np.array([g.X.shape[0] for g in graphs])
    n_edges = np.array([g.y.shape[0] for g in graphs])
    max_nodes = n_nodes.max()
    max_edges = n_edges.max()

    # Allocate the tensors for this batch
    batch_X = np.zeros((batch_size, max_nodes, n_features), dtype=np.float32)
    batch_Ri = np.zeros((batch_size, max_nodes, max_edges), dtype=np.uint8)
    batch_Ro = np.zeros((batch_size, max_nodes, max_edges), dtype=np.uint8)
    batch_y = np.zeros((batch_size, max_edges), dtype=np.uint8)

    # Loop over samples and fill the tensors
    for i, g in enumerate(graphs):
        batch_X[i, :n_nodes[i]] = g.X
        batch_Ri[i, :n_nodes[i], :n_edges[i]] = g.Ri
        batch_Ro[i, :n_nodes[i], :n_edges[i]] = g.Ro
        batch_y[i, :n_edges[i]] = g.y
    
    return batch_X, batch_Ri, batch_Ro, batch_y

def batch_generator(graphs, n_samples=1, batch_size=1, train=True):
    volatile = not train
    batch_idxs = np.arange(0, n_samples, batch_size)
    # Loop over epochs
    while True:
        # Loop over batches
        for j in batch_idxs:
            batch_graphs = [graph_from_sparse(g) for g in graphs[j:j+batch_size]]
            batch_X, batch_Ri, batch_Ro, batch_y = merge_graphs(batch_graphs)
            batch_inputs = [
                np_to_torch(batch_X, volatile=volatile),
                np_to_torch(batch_Ri, volatile=volatile),
                np_to_torch(batch_Ro, volatile=volatile)]
            batch_target = np_to_torch(batch_y, volatile=volatile)
            yield batch_inputs, batch_target

def main():
    """Main program execution function"""
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # CUDA option
    set_cuda(args.cuda)

    # Load the data
    logging.info('Loading input graphs')
    filenames = [os.path.join(args.input_dir, 'event%06i.npz' % i)
                 for i in range(args.n_samples)]
    graphs = load_graphs(filenames, SparseGraph)

    # We round by batch_size to avoid partial batches
    logging.info('Partitioning the data')
    n_test = int(args.n_samples * args.test_frac) // args.batch_size * args.batch_size
    n_valid = int(args.n_samples * args.valid_frac) // args.batch_size * args.batch_size
    n_train = (args.n_samples - n_valid - n_test) // args.batch_size * args.batch_size
    n_train_batches = n_train // args.batch_size
    n_valid_batches = n_valid // args.batch_size
    n_test_batches = n_test #// args.batch_size

    # Partition the dataset
    train_graphs, test_graphs = train_test_split(graphs, test_size=n_test)
    train_graphs, valid_graphs = train_test_split(train_graphs, test_size=n_valid)

    logging.info('Train set size: %i' % len(train_graphs))
    logging.info('Valid set size: %i' % len(valid_graphs))
    logging.info('Test set size:  %i' % len(test_graphs))

    # Prepare the batch generators
    train_batcher = batch_generator(train_graphs, n_samples=n_train,
                                    batch_size=args.batch_size)
    valid_batcher = batch_generator(valid_graphs, n_samples=n_valid,
                                    batch_size=args.batch_size, train=False)
    test_batcher = batch_generator(test_graphs, n_samples=n_test,
                                   batch_size=1, train=False)

    # Construct the model
    logging.info('Building the model')
    n_features = feature_scale.shape[0]
    model = SegmentClassifier(input_dim=n_features,
                              hidden_dim=args.hidden_dim,
                              n_iters=args.n_iters)
    loss_func = nn.BCELoss()
    estim = Estimator(model, loss_func=loss_func, cuda=args.cuda)

    # Train the model
    estim.fit_gen(train_batcher, n_batches=n_train_batches,
                  valid_generator=valid_batcher, n_valid_batches=n_valid_batches,
                  n_epochs=args.n_epochs, verbose=0)

    # Save outputs
    if args.output_dir is not None:
        logging.info('Writing outputs to %s' % args.output_dir)
        make_path = lambda s: os.path.join(args.output_dir, s)
        # Serialize the model
        torch.save(estim.model.state_dict(), make_path('model'))
        # Save the losses for plotting
        np.savez(os.path.join(args.output_dir, 'losses'),
                 train_losses=estim.train_losses,
                 valid_losses=estim.valid_losses)

    # Optional interactive session
    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
