import unittest

from agtool.models.gnn.gat import GAT


class TestGAT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from os import path as osp
        import torch_geometric.transforms as T
        from torch_geometric.datasets import Reddit, Planetoid
        cls.path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
        dataset = Planetoid(cls.path, 'Cora', transform=T.NormalizeFeatures())
        cls.data = dataset[0]
        cls.num_features = dataset.num_features
        cls.num_classes = dataset.num_classes

    def test01_init(self):
        model = GAT(self.num_features, self.num_classes)
        self.assertIsInstance(model, GAT)

    def test02_fit(self):
        model = GAT(self.num_features, self.num_classes)
        model.fit(self.data, epochs=10, lr=0.05)

    def test03_inference(self):
        model = GAT(self.num_features, self.num_classes)
        model.from_pretrained('gat.pt')
        acc = model.test(self.data)
        self.assertTrue(acc > 0.5)
