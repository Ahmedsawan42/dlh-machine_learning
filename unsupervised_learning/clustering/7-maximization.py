#!/usr/bin/env python3

"""a Model that calculates maximization step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        pi: numpy.ndarray of shape (k,) containing updated priors
        m: numpy.ndarray of shape (k, d) containing updated centroid means
        S: numpy.ndarray of shape (k, d, d) cont.updated covariance matrices
        Returns (None, None, None) on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None
        if not isinstance(g, np.ndarray) or g.ndim != 2:
            return None, None, None

        n, d = X.shape
        k = g.shape[0]

        if g.shape[1] != n or n == 0 or d == 0 or k == 0:
            return None, None, None

        if not np.issubdtype(g.dtype, np.number):
            return None, None, None

        if np.any(np.isnan(g)) or np.any(np.isinf(g)) or np.any(g < 0):
            return None, None, None

        N = np.sum(g, axis=1)
        if np.any(N == 0):
            return None, None, None

        pi = N / n

        m = (np.sum(g[:, :, np.newaxis] * X[np.newaxis, :, :], axis=1)
             / N[:, np.newaxis])

        diff = X[np.newaxis, :, :] - m[:, np.newaxis, :]
        S = (np.einsum('kn,knd,kne->kde', g, diff, diff)
             / N[:, np.newaxis, np.newaxis])

        return pi, m, S
    except Exception:
        return None, None, None
