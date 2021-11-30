import numpy as np

from numba import njit


@njit
def negative_sampling(indptr, indices, n_negatives, n_items):
    """Negative sampling.
    Note:
        numba makes 3 times faster."""
    uids = []
    iids = []
    items = np.arange(n_items)
    mask = np.ones_like(items)
    for i in range(len(indptr) - 1):
        positive = indices[indptr[i]: indptr[i + 1]]
        mask[positive] = 0
        negatives = np.random.choice(
            items[mask != 0],
            min(n_items - len(positive), n_negatives), replace=False
        )
        mask[positive] = 1
        uids.extend([i] * len(negatives))
        iids.extend(negatives)
    return uids, iids
