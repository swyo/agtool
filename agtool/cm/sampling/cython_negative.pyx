from libcpp.pair cimport pair
from libcpp.vector cimport vector

from typing import Tuple, List

import numpy as np
cimport numpy as np
import pandas as pd


cdef extern from "sampling/negative.hpp":
    pair[vector[int], vector[int]] _negative_sampling(vector[int]&, vector[int]&, int, int, int) nogil except +


def negative_sampling(
    np.ndarray[np.int32_t, ndim=1] indptr,
    np.ndarray[np.int32_t, ndim=1] indices,
    int num_negatives, int num_items, int num_threads
) -> Tuple[List[int], List[int]]:
    ret = _negative_sampling(indptr, indices, num_negatives, num_items, num_threads)
    assert isinstance(ret, Tuple)
    return ret

# def test_negative_sampling():
#     load_dataset()
#     indptr = pd.read_csv('ml-100k/processed/indptr', header=None).to_numpy().squeeze().astype(np.int32)
#     indices = pd.read_csv('ml-100k/processed/indices', header=None).to_numpy().squeeze().astype(np.int32)
#     num_items = max(indices) + 1
#     print("num_items: ", num_items)
#     for num_threads in range(1, 7):
#         uids, negatives = cython_negative_sampling(indptr, indices, 5, num_items, num_threads)
#     print("10 samples: ", uids[:10])
#     print("10 negatives: ", negatives[:10])
