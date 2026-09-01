#!/usr/bin/env python3

"""This module perform batch normalization on neural network outputs."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a NN using batch normalization.

    Args:
        Z (numpy.ndarray): Array of shape (m, n) that should be normalized,
            where m is the number of data points and n the number of features.
        gamma (numpy.ndarray): Array of shape (1, n) containing the scales
            used for batch normalization.
        beta (numpy.ndarray): Array of shape (1, n) containing the offsets
            used for batch normalization.
        epsilon (float): Small number used to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix of shape (m, n).
    """
    # Calculate the mean of each feature (column-wise)
    mean = np.mean(Z, axis=0, keepdims=True)

    # Calculate the variance of each feature (column-wise)
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize: subtract mean and divide by standard deviation
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift: apply gamma and beta
    Z_norm = gamma * Z_normalized + beta

    return Z_norm
