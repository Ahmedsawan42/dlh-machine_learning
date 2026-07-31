#!/usr/bin/env python3

"""a Model that tests for the optimum number of clusters by variance"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by analyzing variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive int containing the minimum number of clusters to check
        kmax: positive int containing the maximum number of clusters to check
        iterations: positive int containing the maximum iterations for K-means

    Returns:
        results: list containing the outputs of K-means for each cluster size
        d_vars: list containing the dif. in variance from the smallest clst s.
        Returns (None, None) on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None
        n, d = X.shape
        if n == 0 or d == 0:
            return None, None
        if not isinstance(kmin, int) or kmin < 1:
            return None, None
        if kmax is None:
            kmax = n
        if not isinstance(kmax, int) or kmax < 1:
            return None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None
        if kmax <= kmin:
            return None, None

        results = []
        d_vars = []
        first_var = None

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if C is None:
                return None, None
            var = variance(X, C)
            if var is None:
                return None, None
            results.append((C, clss))
            if first_var is None:
                first_var = var
            d_vars.append(first_var - var)

        return results, d_vars
    except Exception:
        return None, None
