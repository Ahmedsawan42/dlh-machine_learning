#!/usr/bin/env python3

"""Neural Network module of binary classification with one hidden layer"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer of binary classification

    Attributes:
        nx (int): Number of input features
        nodes (int): Number of nodes in the hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Class constructor of NeuralNetwork

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

        # Initialize private weights of hidden layer with random normal
        # Shape: (nodes, nx) - nodes neurons, each with nx weights
        self.__W1 = np.random.randn(nodes, nx)

        # Initialize private bias of hidden layer with zeros
        # Shape: (nodes, 1) - one bias per neuron in hidden layer
        self.__b1 = np.zeros((nodes, 1))

        # Initialize private activated output of hidden layer to 0
        self.__A1 = 0

        # Initialize private weights of output neuron with random normal
        # Shape: (1, nodes) - 1 output neuron, with nodes weights
        self.__W2 = np.random.randn(1, nodes)

        # Initialize private bias of output neuron to 0
        self.__b2 = 0

        # Initialize private activated output of output neuron to 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter of private attribute __W1

        Returns:
            numpy.ndarray: The weights matrix of the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter of private attribute __b1

        Returns:
            numpy.ndarray: The bias vector of the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter of private attribute __A1

        Returns:
            numpy.ndarray: The activated output of the hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter of private attribute __W2

        Returns:
            numpy.ndarray: The weights vector of the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter of private attribute __b2

        Returns:
            float: The bias of the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter of private attribute __A2

        Returns:
            numpy.ndarray: The activated output of the output neuron
        """
        return self.__A2
