#!/usr/bin/env python3
"""
Function to update weights of a neural network with Dropout regularization
using gradient descent.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y (numpy.ndarray): One-hot encoded labels of shape (classes, m)
        weights (dict): Dictionary of weights and biases of the neural network
        cache (dict): Dictionary of outputs and dropout masks of each layer
        alpha (float): Learning rate
        keep_prob (float): Probability that a node will be kept
        L (int): Number of layers of the network

    Returns:
        None (updates weights and biases in place)
    """
    m = Y.shape[1]

    # Output activation
    A = cache[f'A{L}']

    # Gradient at output layer
    dZ = A - Y

    for layer in range(L, 0, -1):
        # Current layer weights
        W = weights[f'W{layer}']

        # Activation from previous layer
        A_prev = cache[f'A{layer - 1}']

        # Gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Calculate dZ for previous layer BEFORE updating W
        if layer > 1:
            A_prev_layer = cache[f'A{layer - 1}']
            D_prev = cache[f'D{layer - 1}']

            dZ = np.matmul(W.T, dZ)
            dZ *= (1 - A_prev_layer ** 2)
            dZ *= D_prev / keep_prob

        # Update parameters
        weights[f'W{layer}'] = W - alpha * dW
        weights[f'b{layer}'] -= alpha * db
