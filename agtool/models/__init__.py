from . import ae, vae, gnn


__all__ = ['ae', 'vae', 'gnn']


######################################################
# Base models
######################################################

import torch
from torch import nn

from torch_geometric import nn as gnn


class PytorchModelBase(nn.Module):
    def __init__(self):
        super(PytorchModelBase, self).__init__()

    def from_pretrained(self, save_fname, verbose=True):
        if verbose:
            print(f"Load weights from {save_fname}")
        self.load_state_dict(torch.load(save_fname))

    def save(self, save_fname, verbose=True):
        if verbose:
            print(f"Save weights to {save_fname}")
        torch.save(self.state_dict(), save_fname)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, gnn.Linear):
            gnn.inits.glorot(m.weight)

    def weights_init(self):
        self.apply(self._weights_init)
