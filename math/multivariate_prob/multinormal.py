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

    def pdf(self, x):
        """
        Calculates the probability density function at a given point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point whose
            PDF should be calculated

        Returns: float: The PDF value at point x
        """
        # Check if x is a numpy.ndarray
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Get the number of dimensions
        d = self.mean.shape[0]

        # Check if x has the correct shape
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Center the data point
        x_centered = x - self.mean

        # Calculate the inverse of the covariance matrix
        # Using Cholesky decomposition for better numerical stability
        """
        try:
            # Using Cholesky decomposition: L @ L.T = cov
            # Then cov_inv = (L^-1).T @ L^-1
            L = np.linalg.cholesky(self.cov)
            L_inv = np.linalg.inv(L)
            cov_inv = L_inv.T @ L_inv
        except np.linalg.LinAlgError:"""
            # If Cholesky fails (non-positive definite), use general inverse
        cov_inv = np.linalg.inv(self.cov)

        # Calculate the Mahalanobis distance squared
        mahalanobis_sq = (x_centered.T @ cov_inv @ x_centered).item()

        # Calculate the determ. of the covariance matrix using Cholesky
        
        try:
            L = np.linalg.cholesky(self.cov)
            # det(cov) = det(L) * det(L.T) = det(L)^2
            det_cov = np.prod(np.diag(L)) ** 2
        except np.linalg.LinAlgError:
            # Fall back to regular determinant
            det_cov = np.linalg.det(self.cov)

        # If determ. is negative or zero (shouldn't be for valid covariance)
        if det_cov <= 0:
            det_cov = 1e-10

        # Calculate the normalization constant
        norm_const = 1.0 / ((2 * np.pi) ** (d / 2.0) * np.sqrt(det_cov))

        # Calculate the PDF
        pdf_value = norm_const * np.exp(-0.5 * mahalanobis_sq)

        return float(pdf_value)
