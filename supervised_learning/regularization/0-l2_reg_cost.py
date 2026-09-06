#!/usr/bin/env python3

"""Function to calculate the cost of neural network with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float): Cost of the network without L2 regularization
        lambtha (float): Regularization parameter
        weights (dict): Dictionary of weights and biases (numpy.ndarrays)
        L (int): Number of layers in the neural network
        m (int): Number of data points used

    Returns:
        float: Cost of the network accounting for L2 regularization
    """
    # Initialize L2 regularization term
    l2_reg = 0.0

    # Sum the squared Frobenius norms of all weight matrices
    for layer in range(1, L + 1):
        # Get the weight matrix for this layer
        W = weights.get(f'W{layer}')
        if W is not None:
            # Add squared Frobenius norm (sum of all squared elements)
            l2_reg += np.sum(W ** 2)

    # Calculate the total cost with L2 regularization
    # L2 regularization term = (lambtha / (2 * m)) * sum of squared weights
    l2_cost = cost + (lambtha / (2 * m)) * l2_reg

    return l2_cost
