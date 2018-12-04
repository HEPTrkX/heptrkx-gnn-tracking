"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

def get_datasets(name, **data_args):
    if name == 'dummy':
        from .dummy import get_datasets
        return get_datasets(**data_args)
    elif name == 'hitgraphs':
        from .hitgraphs import get_datasets
        return get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

def get_data_loaders(name, batch_size, distributed=False, **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'dummy':
        from .dummy import get_datasets
        train_dataset, valid_dataset = get_datasets(**data_args)
    elif name == 'hitgraphs':
        from . import hitgraphs
        train_dataset, valid_dataset = hitgraphs.get_datasets(**data_args)
        collate_fn = hitgraphs.collate_fn
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   sampler=train_sampler, collate_fn=collate_fn)
    valid_data_loader = (DataLoader(valid_dataset, batch_size=batch_size,
                                    collate_fn=collate_fn)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader
