import shutil
import unittest
from os.path import exists

from agtool.download.movielen import ml_100k, ml_10m


class MovieLen(unittest.TestCase):
    def test01_ml_100k(self):
        DIR = ml_100k()
        self.assertTrue(exists(DIR))
        shutil.rmtree(DIR)

    def test01_ml_10m(self):
        DIR = ml_10m()
        self.assertTrue(exists(DIR))
        shutil.rmtree(DIR)
