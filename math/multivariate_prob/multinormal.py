#!/usr/bin/env python3

"""a Model represents a Multivariate Normal distribution."""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Class constructor for MultiNormal.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
                n: number of data points
                d: number of dimensions in each data point
        """
        # Check if data is a numpy.ndarray
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")

        # Check if data is 2D
        if data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Get dimensions
        d, n = data.shape

        # Check if there are at least 2 data points
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean (shape: (d, 1))
        # Mean along axis 1 (columns) since each column is a data point
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data by subtracting the mean
        data_centered = data - self.mean

        # Calculate covariance matrix using the formula:
        #           (1/(n-1)) * data_centered @ data_centered.T
        # Since data_centered is (d, n), data_centered X transpos gives (d, d)
        self.cov = (1 / (n - 1)) * data_centered @ data_centered.T
