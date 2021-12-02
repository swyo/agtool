import unittest

from agtool.models.ae.vanilla import DeepAutoEncoder


class TestAE(unittest.TestCase):
    def test01_init(self):
        model = DeepAutoEncoder()

    def test02_train(self):
        model = DeepAutoEncoder()
        model.fit(epochs=10)

    def test03_analysis(self):
        model = DeepAutoEncoder()
        model.analysis()
