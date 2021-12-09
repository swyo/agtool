import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborSampler

from agtool.models import PytorchModelBase


class GCN(PytorchModelBase):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16, cached=False)
        self.conv2 = GCNConv(16, num_classes, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, data, epochs=100, batch_size=256, lr=5e-3, save_fname='gcn.pt', device='cpu'):
        from os import cpu_count
        from tqdm import tqdm
        from torch import optim, device as torch_device
        from torch_geometric.loader import NeighborSampler
        assert len(set(data.to_dict().keys()).intersection(set(['x', 'y', 'edge_index', 'train_mask']))) == 4
        device = device if isinstance(device, str) else torch_device(device)
        subgraph_loader = NeighborSampler(
            data.edge_index, node_idx=data.train_mask, sizes=[-1],
            batch_size=batch_size, shuffle=True, num_workers=cpu_count() * 3 // 4
        )
        best_acc = 0
        self = self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        pbar = tqdm(range(epochs), desc='Train', position=0, ncols=150)
        for _ in pbar:
            hit = cnt = 0
            self.train()
            for _batch_size, n_id, adj in tqdm(subgraph_loader, position=1, ncols=100, leave=False):
                # adj has edeges for  all hops
                edge_index, _, shape = adj.to(device)
                features = data.x[n_id].to(device)
                optimizer.zero_grad()
                out = self(features, edge_index)
                assert shape[1] == _batch_size
                y_pred = out[:_batch_size]
                y_true = data.y[n_id[:_batch_size]]
                loss = F.nll_loss(y_pred, y_true)
                loss.backward()
                optimizer.step()
                hit += sum(y_pred.argmax(dim=-1) == y_true).item()
                cnt += _batch_size
            acc = hit / cnt
            if epochs % 5 == 0:
                val_acc = self.test(data, batch_size, valid=True, device=device, verbose=False)
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save(save_fname, verbose=False)
            acc = hit / cnt
            pbar.set_postfix({'Acc(val)': f'{val_acc:.4f}[<={best_acc:.4f}]', 'Acc(train)': acc})

    def test(self, data, batch_size, device='cpu', valid=False, verbose=True):
        from os import cpu_count
        from torch import device as torch_device
        from torch_geometric.loader import NeighborSampler
        assert len(set(data.to_dict().keys()).intersection(set(['x', 'y', 'edge_index', 'test_mask', 'val_mask']))) == 5
        device = device if isinstance(device, str) else torch_device(device)
        mask = data.val_mask if valid else data.test_mask
        test_subgraph_loader = NeighborSampler(
            data.edge_index, node_idx=mask, sizes=[-1],
            batch_size=batch_size, shuffle=False, num_workers=cpu_count() * 3 // 4
        )
        self = self.to(device)
        self.eval()
        hit = cnt = 0
        for _batch_size, n_id, adj in test_subgraph_loader:
            edge_index = adj.to(device)[0]
            features = data.x[n_id].to(device)
            y_pred = self(features, edge_index)[:_batch_size]
            y_true = data.y[n_id[:_batch_size]]
            hit += sum(y_pred.argmax(dim=-1) == y_true).item()
            cnt += _batch_size
        acc = hit / cnt
        if verbose:
            print(f'Test Accuracy: {acc}')
        return acc


def main(dataset='Cora', epochs=100, batch_size=128, lr=5e-3, device='cpu'):
    from os import path as osp
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Reddit, Planetoid
    assert dataset in ['Cora', 'Reddit']
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    if dataset == 'Reddit':
        dataset = Reddit(path)
    else:
        dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    _model = GCN(dataset.num_features, dataset.num_classes)
    #_model.fit(data, epochs, batch_size, lr, device=device)
    _model.from_pretrained('gcn.pt')
    _model.test(data, batch_size, device=device)
    _model.eval()
    y_pred = _model(data.x, data.edge_index).argmax(dim=1)
    acc = (sum(y_pred[data.test_mask] == data.y[data.test_mask]) / sum(data.test_mask)).item()
    print(f'Test Accuracy (full used): {acc}')


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
