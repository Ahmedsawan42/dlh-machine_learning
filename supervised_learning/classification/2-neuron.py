#!/usr/bin/env python3

"""a Model that defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """This class defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initialize a new neuron performing binary classification"""
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")

        # Check if nx is positive
        if nx < 1:
            raise ValueError("nx must be positive")

        # Initialize private weights with random normal distribution (2D array)
        self.__W = np.random.randn(1, nx)

        # Initialize private bias to 0
        self.__b = 0

        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights vector of the neuron."""
        return self.__W

    @property
    def b(self):
        """Get the bias of the neuron."""
        return self.__b

    @property
    def A(self):
        """Get the activated output of the neuron (prediction)."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx = number of input features
               m = number of examples

        Returns:
            The private attribute __A (activated output)
        """
        # Calculate the weighted sum: z = W·X + b
        # W has shape (1, nx), X has shape (nx, m)
        # Result of dot product: (1, nx) · (nx, m) = (1, m)
        z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function: A = 1 / (1 + e^(-z))
        # This avoids overflow by using np.exp with clipping
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A
