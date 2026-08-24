#!/usr/bin/env python3

"""This module provides a function to shuffle two matrices in the same way."""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): First array of shape (m, nx) to shuffle, where
                           m is the number of data points and nx is the
                           number of features.
        Y (numpy.ndarray): Second array of shape (m, ny) to shuffle, where
                           m is the same number of data points as in X and
                           ny is the number of features.

    Returns:
        tuple: (shuffled_X, shuffled_Y) containing shuffled X and Y matrices.
    """
    # Get the number of data points
    m = X.shape[0]

    # Generate a random permutation of indices
    permutation = np.random.permutation(m)

    # Shuffle both matrices using the same permutation
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y
