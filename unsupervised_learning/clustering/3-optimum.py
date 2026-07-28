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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Set kmax if not provided
    if kmax is None:
        kmax = X.shape[0]  # Max. possible clusters is number of data points
    elif not isinstance(kmax, int) or kmax <= 0:
        return None, None

    # Ensure kmin <= kmax
    if kmin > kmax:
        return None, None

    # Ensure at least 2 different cluster sizes are analyzed
    if kmax - kmin + 1 < 2:
        return None, None

    n, d = X.shape

    # Check if kmax is valid (can't have more clusters than data points)
    if kmax > n:
        kmax = n
        if kmax - kmin + 1 < 2:
            return None, None

    results = []
    variances = []

    # Loop through each cluster size
    for k in range(kmin, kmax + 1):
        # Run K-means for this cluster size
        C, clss = kmeans(X, k, iterations)

        # Check for failure
        if C is None or clss is None:
            return None, None

        # Calculate variance for this clustering
        var = variance(X, C)
        if var is None:
            return None, None

        # Store results
        results.append((C, clss))
        variances.append(var)

    # Calculate differences in variance from the smallest cluster size
    # The smallest cluster size is kmin (first element in the list)
    min_variance = variances[0]
    # Use absolute difference or variance(kmin) - variance(k)
    # Since variance decreases as k increases, this gives positive values
    d_vars = [min_variance - var for var in variances]

    return results, d_vars
