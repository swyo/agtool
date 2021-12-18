from torch_geometric.nn import GATConv

from agtool.models.gnn.gcn.vanilla import GCN


class GAT(GCN):
    def __init__(self, num_features, num_classes, drop=0.6, heads=8):
        super(GAT, self).__init__(num_classes, num_classes, drop)
        self.conv1 = GATConv(num_features, 8, heads=heads)
        self.conv2 = GATConv(8 * heads, num_classes)
        self.weights_init()

    def forward(self, x, edge_index):
        return super().forward(x, edge_index)

    def fit(self, data, epochs=100, lr=5e-3, weight_decay=5e-4, save_fname='gat.pt', device='cpu'):
        return super().fit(data, epochs, lr, weight_decay, save_fname, device)


def main(dataset='Cora', epochs=100, lr=5e-3, weight_decay=5e-4, heads=8, device='cpu'):
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
    _model = GAT(dataset.num_features, dataset.num_classes, heads=heads)
    _model.fit(data, epochs, lr, weight_decay, save_fname='gat.pt', device=device)
    _model.from_pretrained('gat.pt')
    _model.test(data, device=device)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
