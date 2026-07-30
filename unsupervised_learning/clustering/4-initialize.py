#!/usr/bin/env python3

"""a Model that initializes variables for a Gaussian Mixture Model"""

import numpy as np


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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    if k > X.shape[0]:
        return None, None, None

    n, d = X.shape

    # Import kmeans function
    kmeans = __import__('1-kmeans').kmeans

    # Initialize centroids using K-means
    C, clss = kmeans(X, k)
    if C is None or clss is None:
        return None, None, None

    # Initialize priors evenly
    # Each cluster gets equal probability: 1/k
    pi = np.full(k, 1.0 / k)

    # Centroids are already initialized by K-means
    m = C

    # Initialize covariance matrices as identity matrices for each cluster
    # Shape: (k, d, d)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
