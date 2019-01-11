"""
Main training script for NERSC PyTorch examples
"""

# System
import os
import sys
import argparse
import logging

# Externals
import yaml
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Locals
from datasets import get_data_loaders
from trainers import get_trainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--device', default='cpu')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def config_logging(verbose, output_dir):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'out.log'), mode='w')
    file_handler.setLevel(log_level)
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[stream_handler, file_handler])

def init_workers(distributed=False):
    """Initialize worker process group"""
    rank, n_ranks = 0, 1
    if distributed:
        dist.init_process_group(backend='mpi')
        rank = dist.get_rank()
        n_ranks = dist.get_world_size()
    return rank, n_ranks

def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)

def main():
    """Main function"""

    # Parse the command line
    args = parse_args()
    # Initialize MPI
    rank, n_ranks = init_workers(args.distributed)

    # Load configuration
    config = load_config(args.config)
    data_config = config['data']
    model_config = config.get('model', {})
    train_config = config['training']
    experiment_config = config['experiment']
    output_dir = os.path.expandvars(experiment_config.pop('output_dir', None))
    if rank != 0:
        output_dir = None

    # Setup logging
    config_logging(verbose=args.verbose, output_dir=output_dir)
    logging.info('Initialized rank %i out of %i', rank, n_ranks)
    if args.show_config and (rank == 0):
        logging.info('Command line config: %s' % args)
    if rank == 0:
        logging.info('Configuration: %s', config)
        logging.info('Saving job outputs to %s', output_dir)

    # Load the datasets
    train_data_loader, valid_data_loader = get_data_loaders(
        distributed=args.distributed, **data_config)
    logging.info('Loaded %g training samples', len(train_data_loader.dataset))
    if valid_data_loader is not None:
        logging.info('Loaded %g validation samples', len(valid_data_loader.dataset))

    # Load the trainer
    trainer = get_trainer(distributed=args.distributed, output_dir=output_dir,
                          device=args.device, **experiment_config)
    # Build the model
    trainer.build_model(**model_config)
    if rank == 0:
        trainer.print_model_summary()

    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **train_config)
    if rank == 0:
        trainer.write_summaries()

    # Print some conclusions
    n_train_samples = len(train_data_loader.sampler)
    logging.info('Finished training')
    train_time = np.mean(summary['train_time'])
    logging.info('Train samples %g time %gs rate %g samples/s',
                 n_train_samples, train_time, n_train_samples / train_time)
    if valid_data_loader is not None:
        n_valid_samples = len(valid_data_loader.sampler)
        valid_time = np.mean(summary['valid_time'])
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     n_valid_samples, valid_time, n_valid_samples / valid_time)

    # Drop to IPython interactive shell
    if args.interactive and (rank == 0):
        logging.info('Starting IPython interactive session')
        import IPython
        IPython.embed()

    if rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
