import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborSampler

from agtool.models import PytorchModelBase


class GCN(PytorchModelBase):
    def __init__(self, num_features, num_classes, drop=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16, cached=False)
        self.conv2 = GCNConv(16, num_classes, cached=False)
        self.num_features = num_features
        self.num_classes = num_classes
        self.drop = drop

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, data, epochs=100, lr=5e-3, weight_decay=5e-4, save_fname='gcn.pt', device='cpu'):
        from os import cpu_count
        from tqdm import tqdm
        from torch import optim, device as torch_device
        assert len(set(data.to_dict().keys()).intersection(set(['x', 'y', 'edge_index', 'train_mask']))) == 4
        device = device if isinstance(device, str) else torch_device(device)
        best_acc = 0
        self = self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        pbar = tqdm(range(epochs), desc='Train', position=0, ncols=150)
        x = data.x.to(device)
        y = data.y.to(device)
        edge_index = data.edge_index.to(device)
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()
            y_pred = self(x, edge_index)[data.train_mask]
            y_true = y[data.train_mask]
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            hit = sum(y_pred.argmax(dim=-1) == y_true).item()
            acc = hit / len(y_true)
            if epoch % 5 == 0:
                val_acc = self.test(data, valid=True, device=device, verbose=False)
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save(save_fname, verbose=False)
            pbar.set_postfix({'Acc(val)': f'{val_acc:.4f}[<={best_acc:.4f}]', 'Acc(train)': acc})

    def test(self, data, device='cpu', valid=False, verbose=True):
        from os import cpu_count
        from torch import device as torch_device
        assert len(set(data.to_dict().keys()).intersection(set(['x', 'y', 'edge_index', 'test_mask', 'val_mask']))) == 5
        device = device if isinstance(device, str) else torch_device(device)
        mask = data.val_mask if valid else data.test_mask
        self = self.to(device)
        self.eval()
        x = data.x.to(device)
        y = data.y.to(device)
        edge_index = data.edge_index.to(device)
        y_pred = self(x, edge_index)[mask]
        y_true = data.y[mask]
        hit = sum(y_pred.argmax(dim=-1) == y_true).item()
        acc = hit / len(y_true)
        if verbose:
            print(f'Test Accuracy: {acc}')
        return acc


def main(dataset='Cora', epochs=100, lr=5e-3, weight_decay=5e-4, device='cpu'):
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
    _model.fit(data, epochs, lr, weight_decay, device=device)
    _model.from_pretrained('gcn.pt')
    _model.test(data, device=device)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
