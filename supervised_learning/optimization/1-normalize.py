#!/usr/bin/env python3

"""This module is a function to normalize a matrix using given constants."""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix using provided mean and std deviation.

    Args:
        X (numpy.ndarray): Array of shape (d, nx) to normalize, where
                           d is the number of data points and nx is the
                           number of features.
        m (numpy.ndarray): Array of shape (nx,) containing the mean of
                           each feature.
        s (numpy.ndarray): Array of shape (nx,) containing the standard
                           deviation of each feature.

    Returns:
        numpy.ndarray: The normalized X matrix of shape (d, nx).
    """
    # Subtract the mean and divide by standard deviation
    # Create a copy of s to avoid modifying the original
    s_copy = s.copy()
    # Adding a small epsilon to avoid division by zero if s is 0
    # Add epsilon only where s is zero
    s_copy[s_copy == 0] = 1e-12
    normalized_X = (X - m) / s_copy

    return normalized_X
