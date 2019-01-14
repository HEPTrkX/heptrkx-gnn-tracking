"""
PyTorch specification for the hit graph dataset.
"""

# System imports
import os
import logging

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# Local imports
from datasets.graph import load_graph

class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('event') and f.endswith('.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def get_datasets(input_dir, n_train, n_valid):
    data = HitGraphDataset(input_dir, n_train + n_valid)
    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    return train_data, valid_data

def collate_fn(graphs):
    """
    Collate function for building mini-batches from a list of hit-graphs.
    This function should be passed to the pytorch DataLoader.
    It will stack the hit graph matrices sized according to the maximum
    sizes in the batch and padded with zeros.

    This implementation could probably be optimized further.
    """
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        # Prepend singleton batch dimension, convert inputs and target to torch
        batch_inputs = [torch.from_numpy(m[None]).float() for m in [g.X, g.Ri, g.Ro]]
        batch_target = torch.from_numpy(g.y[None]).float()
        return batch_inputs, batch_target

    # Get the matrix sizes in this batch
    n_features = graphs[0].X.shape[1]
    n_nodes = np.array([g.X.shape[0] for g in graphs])
    n_edges = np.array([g.y.shape[0] for g in graphs])
    max_nodes = n_nodes.max()
    max_edges = n_edges.max()

    # Allocate the tensors for this batch
    batch_X = np.zeros((batch_size, max_nodes, n_features), dtype=np.float32)
    batch_Ri = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    batch_Ro = np.zeros((batch_size, max_nodes, max_edges), dtype=np.float32)
    batch_y = np.zeros((batch_size, max_edges), dtype=np.float32)

    # Loop over samples and fill the tensors
    for i, g in enumerate(graphs):
        batch_X[i, :n_nodes[i]] = g.X
        batch_Ri[i, :n_nodes[i], :n_edges[i]] = g.Ri
        batch_Ro[i, :n_nodes[i], :n_edges[i]] = g.Ro
        batch_y[i, :n_edges[i]] = g.y

    batch_inputs = [torch.from_numpy(bm) for bm in [batch_X, batch_Ri, batch_Ro]]
    batch_target = torch.from_numpy(batch_y)
    return batch_inputs, batch_target
