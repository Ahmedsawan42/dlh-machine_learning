#!/usr/bin/env python3

"""a Model that defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    def __init__(self, nx):
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

    # Getter for W
    @property
    def W(self):
        return self.__W

    # Getter for b
    @property
    def b(self):
        return self.__b

    # Getter for A
    @property
    def A(self):
        return self.__A
