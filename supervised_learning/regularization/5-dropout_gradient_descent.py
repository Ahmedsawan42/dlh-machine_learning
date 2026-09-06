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
    # Get number of data points
    m = Y.shape[1]

    # Get the output of the last layer (softmax output)
    A_prev = cache[f'A{L}']

    # Calculate derivative of loss with respect to output (softmax)
    dZ = A_prev - Y

    # Backpropagate through the network
    for layer in range(L, 0, -1):
        # Get current layer's weights and bias
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        # Get current layer's activation from cache
        A_prev_layer = cache[f'A{layer - 1}'] if layer > 1 else cache['A0']

        # Calculate gradients
        dW = (1 / m) * np.matmul(dZ, A_prev_layer.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases in place
        weights[f'W{layer}'] = W - alpha * dW
        weights[f'b{layer}'] = b - alpha * db

        # If not the first layer, compute dZ for the previous layer
        if layer > 1:
            # Get previous layer's weights
            W_prev = weights[f'W{layer - 1}']

            # Get previous layer's activation and dropout mask
            A_prev2 = cache[f'A{layer - 1}']
            D_prev = cache[f'D{layer - 1}']

            # Calculate dZ for previous layer using tanh derivative
            # First compute dZ with respect to the pre-activation
            dZ = np.matmul(W_prev.T, dZ) * (1 - A_prev2 ** 2)

            # Apply dropout mask to dZ for the previous layer
            # Scale by keep_prob to maintain expected gradient magnitude
            dZ = dZ * D_prev / keep_prob
