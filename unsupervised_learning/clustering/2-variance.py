#!/usr/bin/env python3

"""a Model that calculates the total intra-cluster variance for a data set"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        C: numpy.ndarray of shape (k, d), the centroid means for each cluster

    Returns:
        var: total intra-cluster variance
        None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None
        if not isinstance(C, np.ndarray) or C.ndim != 2:
            return None
        if X.shape[1] != C.shape[1]:
            return None
        if C.shape[0] == 0:
            return None

        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        min_sq_dist = np.min(sq_dist, axis=1)
        var = np.sum(min_sq_dist)

        return var
    except Exception:
        return None
