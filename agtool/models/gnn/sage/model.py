from os import cpu_count
from itertools import islice

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import SGDClassifier
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

from agtool.models import PytorchModelBase
from agtool.cm.sampling import negative_sampling


def _chunk(it, size):
    it = iter(it)
    for _it in iter(lambda: list(islice(it, size)), ()):
        if not _it:
            break
        yield _it


class NeighborSampler(RawNeighborSampler):
    def __init__(self, edge_index, *args, num_negatives=None, **kwargs):
        super(NeighborSampler, self).__init__(edge_index, *args, **kwargs)
        if num_negatives:
            self.num_negatives = num_negatives
            self.reset_negatives(edge_index)

    def sample(self, batch):
        batch = torch.tensor(batch)
        # For each node in `batch`, we sample a direct neighbor (as positive example) and a random node (as negative example):
        row, col, _ = self.adj_t[batch].coo()
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)[:, 1]
        neg_batch = self.neg_samples[batch].view(-1)
        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        # batch_size + batch_size(positive) + batch_size * num_negatives(negative) = out[0] in super().sample
        _batch_size, n_id, adjs = super(NeighborSampler, self).sample(batch)
        _batch_size = _batch_size // (self.num_negatives + 2)
        return _batch_size, n_id, adjs

    def reset_negatives(self, edge_index):
        indptr, indices, _ = self.adj_t.csr()
        indptr, indices = indptr.numpy().astype(np.int32), indices.numpy().astype(np.int32)
        num_workers = cpu_count() * 3 // 4 if self.num_workers == 0 else self.num_workers
        _, cols = negative_sampling(
            indptr,
            indices,
            self.num_negatives,
            self.adj_t.size(-1) - 1,
            min(num_workers, 16)
        )
        self.neg_samples = torch.tensor(cols, dtype=torch.int64).view(-1, self.num_negatives)


class GraphSAGE(PytorchModelBase):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.supervised = False
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def to_supervised(self, num_classes):
        self.supervised = True
        self.num_classes = num_classes
        self.convs[-1] = SAGEConv(self.hidden_channels, num_classes)

    def forward(self, x, adjs):
        """Forward propagation for minibatch."""
        assert len(adjs) == self.num_layers
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        if not self.supervised:
            return x
        return x.log_softmax(dim=-1)

    def full_forward(self, x, edge_index):
        """Forward propagation for the entire batch."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.supervised or (i != self.num_layers - 1):
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def fit(self, data, epochs=100, batch_size=128, lr=5e-3, weight_decay=5e-4, device='cpu', num_workers=0, save_fname='grapsage.pt', num_negatives=5):
        from tqdm import tqdm
        from torch import optim, device as torch_device
        device = device if isinstance(device, str) else torch_device(device)
        best_acc = 0
        self.best_clf = None
        self = self.to(device)
        data = data.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        sampler_kwargs = dict(
            edge_index=data.edge_index, batch_size=batch_size,
            sizes=[25, 10], shuffle=True, return_e_id=False,
            num_nodes=data.num_nodes, node_idx=data.train_mask,
            num_workers=cpu_count() * 3 // 4 if num_workers == 0 else num_workers
        )
        if self.supervised:
            sampler = RawNeighborSampler(**sampler_kwargs)
        else:
            sampler = NeighborSampler(num_negatives=num_negatives, **sampler_kwargs)
        pbar = tqdm(range(epochs), 'Train', position=0, ncols=150)
        for epoch in pbar:
            if self.supervised:
                hit = total = 0
            else:
                labels = []
                embeddings = []
            self.train()
            for _batch_size, n_id, adjs in tqdm(sampler, 'Minibatch', position=1, ncols=150, leave=False):
                optimizer.zero_grad()
                adjs = [adj.to(device) for adj in adjs]
                y_true = data.y[n_id][:_batch_size].to(device)
                out = self(data.x[n_id].to(device), adjs)
                if self.supervised:
                    logit = out
                    loss = F.nll_loss(logit, y_true)
                else:
                    nums_splited_nodes = [_batch_size, _batch_size, _batch_size * num_negatives]
                    assert out.size(0) == sum(nums_splited_nodes)
                    emb, pos_emb, neg_emb = out.split(nums_splited_nodes, dim=0)
                    pos_loss = F.logsigmoid((emb * pos_emb).sum(-1)).mean()
                    emb_repeat = emb.repeat(num_negatives, 1)
                    neg_loss = F.logsigmoid(-(emb_repeat * neg_emb).sum(-1)).mean()
                    loss = -pos_loss - neg_loss
                loss.backward()
                optimizer.step()
                if self.supervised:
                    hit += sum(logit.argmax(dim=-1) == y_true).item()
                    total += _batch_size
                else:
                    labels.append(y_true.cpu())
                    embeddings.append(emb.detach().cpu())
            if self.supervised:
                acc = hit / float(total)
            else:
                labels = torch.cat(labels, dim=0)
                embeddings = torch.cat(embeddings, dim=0)
                self.clf = self.fit_classifier(embeddings, labels)
                acc = self.clf.score(embeddings, labels)
                if epoch > 0:
                    sampler.reset_negatives(data.edge_index)
            val_acc = self.test(data, (batch_size + 1) // 2, valid=True, device=device, verbose=False, num_workers=num_workers)
            if val_acc > best_acc:
                if not self.supervised:
                    self.best_clf = self.clf
                best_acc = val_acc
                self.save(save_fname, verbose=False)
            pbar.set_postfix({'Acc(val)': f'{val_acc:.4f}[<={best_acc:.4f}]', 'Acc(train)': acc, 'Loss': loss.item()})

    def test(self, data, batch_size=128, device='cpu', num_workers=0, valid=False, verbose=True):
        from tqdm import tqdm
        from torch import device as torch_device
        device = device if isinstance(device, str) else torch_device(device)
        self = self.to(device)
        mask = data.val_mask if valid else data.test_mask
        sampler_kwargs = dict(
            edge_index=data.edge_index, batch_size=batch_size,
            sizes=[50, 25], shuffle=False, return_e_id=False,
            num_nodes=data.num_nodes, node_idx=mask,
            num_workers=cpu_count() * 3 // 4 if num_workers == 0 else num_workers
        )
        sampler = RawNeighborSampler(**sampler_kwargs)
        self.eval()
        if self.supervised:
            hit = total = 0
        else:
            labels = []
            embeddings = []
        for _batch_size, n_id, adjs in tqdm(sampler, 'Test', position=1, ncols=150, leave=False):
            adjs = [adj.to(device) for adj in adjs]
            y_true = data.y[n_id][:_batch_size].to(device)
            out = self(data.x[n_id].to(device), adjs)
            if self.supervised:
                logit = out
                y_pred = logit.argmax(dim=-1)
                hit += sum(y_pred == y_true).item()
                total += _batch_size
            else:
                emb = out
                labels.append(y_true.cpu())
                embeddings.append(emb.detach().cpu())
        if self.supervised:
            acc = hit / float(total)
        else:
            labels = torch.cat(labels, dim=0)
            embeddings = torch.cat(embeddings, dim=0)
            clf = self.best_clf if not valid and self.best_clf else self.clf
            acc = clf.score(embeddings, labels)
        if verbose:
            print(f'\rTest Accuracy: {acc}')
        return acc

    def test_full(self, data, valid=False, verbose=True):
        self.eval()
        out = self.full_forward(data.x, data.edge_index).detach().cpu()
        clf = self.fit_classifier(out, data.y[data.train_mask])
        if valid:
            mask = data.val_mask
        else:
            mask = data.test_mask
        y_true = data.y[mask].detach().cpu().numpy()
        acc = clf.score(out[mask], y_true)
        if not verbose:
            return acc
        print(f'\rTest Accuracy (full): {acc}')

    def fit_classifier(self, x, y, batch_size=64, epochs=400):
        clf = SGDClassifier()
        classes = np.unique(y)
        for _ in range(epochs):
            _range = np.arange(len(y))
            np.random.shuffle(_range)
            for indices in _chunk(_range, batch_size):
                clf.partial_fit(x[indices], y[indices], classes)
        return clf


def main(dataset='Cora', epochs=100, batch_size=128, lr=5e-3, weight_decay=5e-4, device='cpu', num_workers=0, supervised=True, num_negatives=5):
    from os import path as osp
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Reddit, Planetoid
    assert dataset in ['Cora', 'Reddit', 'CiteSeer', 'Pubmed']
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    if dataset == 'Reddit':
        dataset = Reddit(path)
    else:
        dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    _model = GraphSAGE(dataset.num_features, 256 if supervised else 128)
    if supervised:
        _model.to_supervised(dataset.num_classes)
    print(_model)
    _model.fit(data, epochs, batch_size, lr, weight_decay, device=device, num_workers=num_workers, save_fname='graphsage.pt', num_negatives=num_negatives)
    _model.from_pretrained('graphsage.pt')
    _model.test(data, batch_size=(batch_size + 1) // 2, device=device, num_workers=num_workers)
    _model.test_full(data)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
