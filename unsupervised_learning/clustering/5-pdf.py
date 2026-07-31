#!/usr/bin/env python3

"""a Model that calculates the probability density function of a Gaussian D"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data points
        m: numpy.ndarray of shape (d,) containing the mean of the distribution
        S: numpy.ndarray of shape (d, d) containing the distribution covarianc
    Returns:
        P: numpy.ndarray of shape (n,) containing the PDF values
        Returns None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None
        if not isinstance(m, np.ndarray) or m.ndim != 1:
            return None
        if not isinstance(S, np.ndarray) or S.ndim != 2:
            return None

        n, d = X.shape
        if m.shape[0] != d or S.shape != (d, d):
            return None

        det_S = np.linalg.det(S)
        if det_S <= 0:
            return None

        inv_S = np.linalg.inv(S)

        diff = X - m
        mahal = np.sum((diff @ inv_S) * diff, axis=1)

        norm = 1 / np.sqrt((2 * np.pi) ** d * det_S)
        P = norm * np.exp(-0.5 * mahal)
        P = np.maximum(P, 1e-300)

        return P
    except Exception:
        return None
