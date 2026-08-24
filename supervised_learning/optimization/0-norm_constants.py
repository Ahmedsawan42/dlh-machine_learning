#!/usr/bin/env python3

"""This module that calculate normalization constants for a matrix."""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): Array of shape (m, nx) to normalize, where
                           m is the number of data points and nx is the
                           number of features.

    Returns:
        tuple: (mean, std) containing the mean and standard deviation of
               each feature, respectively. Both are numpy.ndarrays of shape
               (nx,).
    """
    # Calculate the mean of each feature (column-wise)
    mean = np.mean(X, axis=0)

    # Calculate the standard deviation of each feature (column-wise)
    # Using ddof=0 for population standard deviation.
    std = np.std(X, axis=0)

    return mean, std
