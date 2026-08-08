#!/usr/bin/env python3

"""Neural Network module for binary classification with one hidden layer"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer for binary classification

    Attributes:
        nx (int): Number of input features
        nodes (int): Number of nodes in the hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Class constructor for NeuralNetwork

        Args:
            nx (int): Number of input features
            nodes (int): Number of nodes in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1

        All exceptions are raised in the order listed above
        """
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        # Check if nx is positive
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check if nodes is an integer
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        # Check if nodes is positive
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights for hidden layer with random normal distribution
        # Shape: (nodes, nx) - nodes neurons, each with nx weights
        self.W1 = np.random.randn(nodes, nx)

        # Initialize bias for hidden layer with zeros
        # Shape: (nodes, 1) - one bias per neuron in hidden layer
        self.b1 = np.zeros((nodes, 1))

        # Initialize activated output for hidden layer to 0
        self.A1 = 0

        # Initialize weights for output neuron with random normal distribution
        # Shape: (1, nodes) - 1 output neuron, with nodes weights
        # (one for each hidden neuron)
        self.W2 = np.random.randn(1, nodes)

        # Initialize bias for output neuron to 0
        self.b2 = 0

        # Initialize activated output for output neuron to 0
        self.A2 = 0
