#!/usr/bin/env python3

"""
Function to update weights and biases of a neural network using gradient
descentwith L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y (numpy.ndarray): One-hot encoded labels of shape (classes, m)
        weights (dict): Dictionary of weights and biases of the neural network
        cache (dict): Dictionary of outputs of each layer of the neuralnetwork
        alpha (float): Learning rate
        lambtha (float): L2 regularization parameter
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

        # Calculate gradients with L2 regularization
        # dW = (1/m) * dZ * A_prev.T + (lambtha/m) * W
        dW = (1 / m) * np.matmul(dZ, A_prev_layer.T) + (lambtha / m) * W

        # db = (1/m) * sum(dZ, axis=1, keepdims=True)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases in place
        weights[f'W{layer}'] = W - alpha * dW
        weights[f'b{layer}'] = b - alpha * db

        # If not the first layer, compute dZ for the previous layer
        if layer > 1:
            # Get previous layer's weights
            W_prev = weights[f'W{layer - 1}']

            # Get previous layer's activation
            A_prev2 = cache[f'A{layer - 1}']

            # Calculate dZ for previous layer using tanh derivative
            # dZ_prev = (W.T * dZ) * (1 - A_prev2^2)
            dZ = np.matmul(W.T, dZ) * (1 - A_prev2 ** 2)
