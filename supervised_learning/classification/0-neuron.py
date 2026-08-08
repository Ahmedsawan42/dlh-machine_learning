#!/usr/bin/env python3

"""a Model that defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """This class defines a single neuron performing binary classification"""
    def __init__(self, nx):
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        # Check if nx is positive
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights with random normal distribution  as a row vector
        self.W = np.random.randn(1, nx)

        # Initialize bias to 0
        self.b = 0

        # Initialize activated output to 0
        self.A = 0
