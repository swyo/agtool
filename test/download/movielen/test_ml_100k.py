import shutil
import unittest
from os.path import exists

from agtool.download.movielen import ml_100k


class MovieLen(unittest.TestCase):
    def test01_ml_100k(self):
        DIR = ml_100k()
        self.assertTrue(exists(DIR))
        shutil.rmtree(DIR)
