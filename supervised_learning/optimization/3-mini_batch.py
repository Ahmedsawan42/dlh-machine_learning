#!/usr/bin/env python3

"""This a module to create mini-batches for training neural networks."""

import numpy as np

# Import shuffle_data function
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network using
    mini-batch gradient descent.

    Args:
        X (numpy.ndarray): Array of shape (m, nx) representing input data,
                           where m is the number of data points and nx is
                           the number of features.
        Y (numpy.ndarray): Array of shape (m, ny) representing the labels,
                           where m is the same number of data points as in X
                           and ny is the number of classes.
        batch_size (int): The number of data points in a batch.

    Returns:
        list: List of mini-batches containing tuples (X_batch, Y_batch).
           Each tuple contains a batch of input data and corresponding labels.
    """
    # Shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # Get the number of data points
    m = X_shuffled.shape[0]

    # Initialize list to store mini-batches
    mini_batches = []

    # Create full batches
    for i in range(0, m, batch_size):
        # Get the end index for the batch
        end = min(i + batch_size, m)

        # Create batch
        X_batch = X_shuffled[i:end]
        Y_batch = Y_shuffled[i:end]

        # Add batch to list
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
