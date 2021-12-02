import unittest

from agtool.models.ae.vanilla import DeepAutoEncoder


class TestAE(unittest.TestCase):
    def test01_init(self):
        model = DeepAutoEncoder()
