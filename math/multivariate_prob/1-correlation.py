#!/usr/bin/env python3

"""a Model Calculates the correlation matrix from a covariance matrix"""

import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Args: C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns: numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    # Check if C is a numpy.ndarray
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check if C is 2D and square
    if C.ndim != 2:
        raise ValueError("C must be a 2D square matrix")

    d1, d2 = C.shape
    if d1 != d2:
        raise ValueError("C must be a 2D square matrix")

    d = d1

    # Extract the diagonal (standard deviations)
    # Note: variance is on the diagonal of the covariance matrix
    variances = np.diag(C)

    # Calculate standard deviations (sqrt of variance)
    std_devs = np.sqrt(variances)

    # Create correlation matrix
    # Method 1: Using broadcasting
    # corr = C / np.outer(std_devs, std_devs)

    # Method 2: More explicit with error handling for zero variance
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if std_devs[i] == 0 or std_devs[j] == 0:
                # If variance zero, correlation is undefined will set it to 0
                corr[i, j] = 0
            else:
                corr[i, j] = C[i, j] / (std_devs[i] * std_devs[j])

    return corr
