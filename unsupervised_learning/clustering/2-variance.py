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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:  # Check if dimensions match
        return None
    if C.shape[0] == 0:  # At least one cluster
        return None
    if X.shape[0] == 0:  # At least one data point
        return None

    n, d = X.shape

    # Compute distance from each point to each centroid
    # X[:, np.newaxis, :] shape: (n, 1, d)
    # C[np.newaxis, :, :] shape: (1, k, d)
    # Result shape: (n, k, d)
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]

    # Compute squared distances
    # Sum along the last dimension (d) to get squared Euclidean distance
    # Result shape: (n, k)
    squared_distances = np.sum(diff ** 2, axis=2)

    # Find the minimum distance for each point (nearest centroid)
    # Result shape: (n,)
    min_distances = np.min(squared_distances, axis=1)

    # Sum all minimum squared distances to get total variance
    var = np.sum(min_distances)

    return var
