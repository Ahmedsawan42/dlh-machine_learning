#!/usr/bin/env python3

"""Function to conduct forward propagation using Dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X (numpy.ndarray): Input data of shape (nx, m)
        weights (dict): Dictionary of weights and biases of the neural network
        L (int): Number of layers in the network
        keep_prob (float): Probability that a node will be kept

    Returns:
        dict: Dictionary containing:
            - outputs of each layer (A0, A1, ..., AL)
            - dropout masks used on each layer (D1, D2, ..., D(L-1))
    """
    # Initialize cache dictionary
    cache = {}

    # Store input as A0
    cache['A0'] = X

    # Forward propagation through all layers
    for layer in range(1, L + 1):
        # Get weights and bias for current layer
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        # Get previous layer's output
        A_prev = cache[f'A{layer - 1}']

        # Linear transformation
        Z = np.matmul(W, A_prev) + b

        # Apply activation function
        if layer == L:
            # Last layer: softmax activation
            # Softmax for numerical stability
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Hidden layers: tanh activation
            A = np.tanh(Z)

            # Apply dropout to the activations
            # Create dropout mask D for this layer as integers (0/1)
            D = (
                np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            ).astype(int)
            # Scale the activations to maintain expected value
            A = A * D / keep_prob

            # Store dropout mask in cache
            cache[f'D{layer}'] = D

        # Store output of current layer
        cache[f'A{layer}'] = A

    return cache
