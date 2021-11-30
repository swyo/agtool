import time
import shutil
import unittest
from os import cpu_count

import numpy as np
import pandas as pd
from numba import set_num_threads
from scipy.sparse import csr_matrix

# from agtool.download.movielen import ml_10m
from agtool.download.movielen import ml_100k
from agtool.cm.sampling import negative_sampling
from agtool.sampling import negative_sampling as numba_negative_sampling


class TestNegativeSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.DIR = ml_10m()
        cls.DIR = ml_100k()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.DIR)
        print(f"Remove {cls.DIR}")

    def _load_data(self):
        df_indptr = pd.read_csv(f'{self.DIR}/processed/indptr', header=None)
        indptr = df_indptr.to_numpy().squeeze().astype(np.int32)
        df_indices = pd.read_csv(f'{self.DIR}/processed/indices', header=None)
        indices = df_indices.to_numpy().squeeze().astype(np.int32)
        num_items = max(indices) + 1
        return indptr, indices, num_items

    def test01_negative_sampling(self):
        indptr, indices, num_items = self._load_data()
        print("num_items: ", num_items)
        for num_threads in range(1, min(7, cpu_count())):
            start = time.time()
            uids, negatives = negative_sampling(
                indptr, indices, 5, num_items, num_threads
            )
            print(f"[negative_sampling] takes {time.time() - start:.6f} seconds for [{num_threads}] threads")
        print("10 samples: ", uids[:10])
        print("10 negatives: ", negatives[:10])
        negative_matrix = csr_matrix((np.ones_like(uids), (uids, negatives)), shape=(len(indptr) - 1, num_items))
        for i in range(len(indptr) - 1):
            positives = indices[indptr[i]: indptr[i + 1]]
            negatives = negative_matrix[i].nonzero()[1]
            intersect = np.intersect1d(positives, negatives)
            is_empty = intersect.size == 0
            msg = f"Negative sampling [Invalid] intersect[user {i}]: {intersect.tolist()}"
            self.assertTrue(is_empty, msg=msg)

    def test02_speed_experiments(self):
        indptr, indices, num_items = self._load_data()
        start = time.time()
        numba_negative_sampling(indptr, indices, 5, num_items)
        print(f"[numba_negative_sampling] build time takes {time.time() - start:.6f} seconds")
        for num_threads in range(1, min(cpu_count(), 7)):
            set_num_threads(num_threads)
            start = time.time()
            numba_negative_sampling(indptr, indices, 5, num_items)
            print(f"[numba_negative_sampling] takes {time.time() - start:.6f} seconds for [{num_threads}] threads")
