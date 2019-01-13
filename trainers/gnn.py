"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time

# Externals
import torch
from torch import nn

# Locals
from .base_trainer import BaseTrainer
from models import get_model

class GNNTrainer(BaseTrainer):
    """Trainer code for basic classification problems."""

    def __init__(self, real_weight=1, fake_weight=1, **kwargs):
        super(GNNTrainer, self).__init__(**kwargs)
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def build_model(self, name='gnn_segment_classifier',
                    optimizer='Adam', learning_rate=0.001,
                    loss_func='binary_cross_entropy', **model_args):
        """Instantiate our model"""

        # Construct the model
        self.model = get_model(name=name, **model_args).to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallelCPU(self.model)
        # TODO: LR scaling
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=learning_rate)
        # Functional loss functions
        self.loss_func = getattr(nn.functional, loss_func)
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0
        start_time = time.time()
        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            # Compute target weights on-the-fly for loss function
            batch_weights_real = batch_target * self.real_weight
            batch_weights_fake = (1 - batch_target) * self.fake_weight
            batch_weights = batch_weights_real + batch_weights_fake
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target, weight=batch_weights)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            self.logger.debug('  batch %i, loss %f', i, batch_loss.item())

        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches' % (i + 1))
        self.logger.info('  Training loss: %.3f' % summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        start_time = time.time()
        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            self.logger.debug(' batch %i', i)
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            sum_loss += self.loss_func(batch_output, batch_target).item()
            # Count number of correct predictions
            matches = ((batch_output > 0.5) == (batch_target > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

def _test():
    t = GNNTrainer(output_dir='./')
    t.build_model()
