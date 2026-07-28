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
    # Check for invalid inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if k > X.shape[0]:  # More clusters than data points
        return None

    # Get dimensions
    n, d = X.shape

    # Find min and max along each dimension
    min_vals = np.min(X, axis=0)  # shape (d,)
    max_vals = np.max(X, axis=0)  # shape (d,)

    # Initialize centroids with uniform distribution
    # Use numpy.random.uniform exactly once with broadcasting
    centroids = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, d)
    )

    return centroids
