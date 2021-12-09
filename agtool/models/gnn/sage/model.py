from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler

from agtool.models import PytorchModelBase


class GraphSAGE(PytorchModelBase):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.supervised = False
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else 2 * hidden_channels
            self.convs.append(
                SAGEConv(
                    in_channels, hidden_channels
                    if i != 0 else 2 * hidden_channels
                )
            )

    def to_supervised(self, num_classes):
        self.supervised = True
        self.num_classes = num_classes
        self.clf = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, adjs):
        """Forward propagation for minibatch."""
        assert len(adjs) == self.num_layers
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if self.supervised or (i != self.num_layers - 1):
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        if not self.supervised:
            return x
        return self.clf(x).log_softmax(dim=-1)

    def fit(self, data, epochs=100, batch_size=128, lr=5e-3, weight_decay=5e-4, device='cpu', save_fname='grapsage.pt'):
        from os import cpu_count
        from tqdm import tqdm
        from torch import optim, device as torch_device
        device = device if isinstance(device, str) else torch_device(device)
        best_acc = 0
        self = self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        sampler = NeighborSampler(
            data.edge_index, batch_size=batch_size,
            sizes=[10, 25], shuffle=True, return_e_id=False,
            num_nodes=data.num_nodes, node_idx=data.train_mask,
            num_workers=cpu_count() * 3 // 4
        )
        pbar = tqdm(range(epochs), 'Train', position=0, ncols=150)
        for epoch in pbar:
            hit = total = 0
            self.train()
            for _batch_size, n_id, adjs in tqdm(sampler, 'Minibatch', position=1, ncols=150, leave=False):
                optimizer.zero_grad()
                adjs = [adj.to(device) for adj in adjs]
                logit = self(data.x[n_id].to(device), adjs)
                y_true = data.y[n_id][:_batch_size].to(device)
                loss = F.nll_loss(logit, y_true)
                loss.backward()
                optimizer.step()
                hit += sum(logit.argmax(dim=-1) == y_true).item()
                total += _batch_size
            acc = hit / float(total)
            if epoch % 5 == 0:
                val_acc = self.test(data, batch_size, valid=True, device=device, verbose=False)
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save(save_fname, verbose=False)
            pbar.set_postfix({'Acc(val)': f'{val_acc:.4f}[<={best_acc:.4f}]', 'Acc(train)': acc})

    def test(self, data, batch_size=128, device='cpu', valid=False, verbose=True):
        from os import cpu_count
        from tqdm import tqdm
        from torch import optim, device as torch_device
        device = device if isinstance(device, str) else torch_device(device)
        self = self.to(device)
        mask = data.val_mask if valid else data.test_mask
        sampler = NeighborSampler(
            data.edge_index, batch_size=batch_size,
            sizes=[-1, -1], shuffle=True, return_e_id=False,
            num_nodes=data.num_nodes, node_idx=mask,
            num_workers=cpu_count() * 3 // 4
        )
        hit = total = 0
        for _batch_size, n_id, adjs in tqdm(sampler, 'Test', position=1, ncols=150, leave=False):
            adjs = [adj.to(device) for adj in adjs]
            logit = self(data.x[n_id].to(device), adjs)
            y_true = data.y[n_id][:_batch_size].to(device)
            hit += sum(logit.argmax(dim=-1) == y_true).item()
            total += _batch_size
        acc = hit / float(total)
        if verbose:
            print(f'\rTest Accuracy: {acc}')
        return acc


def main(dataset='Cora', epochs=100, batch_size=128, lr=5e-3, weight_decay=5e-4, device='cpu'):
    from os import path as osp
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Reddit, Planetoid
    assert dataset in ['Cora', 'Reddit', ]
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    if dataset == 'Reddit':
        dataset = Reddit(path)
    else:
        dataset = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    _model = GraphSAGE(dataset.num_features, 256)
    _model.to_supervised(dataset.num_classes)
    _model.weights_init()
    print(_model)
    _model.fit(data, epochs, batch_size, lr, weight_decay, device=device, save_fname='graphsage.pt')
    _model.from_pretrained('graphsage.pt')
    _model.test(data, batch_size, device, False, True)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
