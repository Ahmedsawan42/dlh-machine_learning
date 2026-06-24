#!/usr/bin/env python3

"""a Model Calculates the mean and covariance of a data set"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args: X: numpy.ndarray of shape (n, d) containing the data set.

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data.
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix.
    """
    # Check if X is a 2D numpy.ndarray
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    # Get dimensions
    n, d = X.shape

    # Check if there are at least 2 data points
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate mean (axis=0 gives mean along each column/dimension)
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate covariance matrix
    # Center the data by subtracting the mean
    X_centered = X - mean

    # Calculate covariance using the formula:
    cov = (1 / (n - 1)) * X_centered.T @ X_centered

    return mean, cov
