#!/usr/bin/env python3

"""a Model that initializes cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate
    uniform distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None
        if not isinstance(k, int) or k <= 0:
            return None

        n, d = X.shape
        if n == 0 or d == 0:
            return None

        low = np.min(X, axis=0)
        high = np.max(X, axis=0)

        return np.random.uniform(low, high, size=(k, d))
    except Exception:
        return None
