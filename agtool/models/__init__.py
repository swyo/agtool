from . import ae, vae, gnn


__all__ = ['ae', 'vae', 'gnn']


######################################################
# Base models
######################################################

import torch


class PytorchModelBase(torch.nn.Module):
    def __init__(self):
        super(PytorchModelBase, self).__init__()

    def from_pretrained(self, save_fname):
        print(f"Load weights from {save_fname}")
        self.load_state_dict(torch.load(save_fname))

    def save(self, save_fname):
        print(f"Save weights to {save_fname}")
        torch.save(self.state_dict(), save_fname)
