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
