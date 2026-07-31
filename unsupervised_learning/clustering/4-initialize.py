#!/usr/bin/env python3

"""a Model that initializes variables for a Gaussian Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) containing priors for each cluster
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices
        Returns (None, None, None) on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None
        if not isinstance(k, int) or k <= 0:
            return None, None, None

        n, d = X.shape
        if n == 0 or d == 0:
            return None, None, None

        pi = np.full((k,), 1 / k)

        m, clss = kmeans(X, k)
        if m is None:
            return None, None, None

        S = np.tile(np.eye(d), (k, 1, 1))

        return pi, m, S
    except Exception:
        return None, None, None
