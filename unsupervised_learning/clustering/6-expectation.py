#!/usr/bin/env python3

"""a Model that calculates the expectation step in the EM algorithm for GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        pi: numpy.ndarray of shape (k,) containing the priors for each cluster
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices

    Returns:
        g: numpy.ndarray of shape (k, n) containing posterior probabilities
        l: total log likelihood
        Returns (None, None) on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None
        if not isinstance(pi, np.ndarray) or pi.ndim != 1:
            return None, None
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            return None, None
        if not isinstance(S, np.ndarray) or S.ndim != 3:
            return None, None

        n, d = X.shape
        k = pi.shape[0]

        if n == 0 or d == 0 or k == 0:
            return None, None
        if m.shape != (k, d):
            return None, None
        if S.shape != (k, d, d):
            return None, None

        joint = np.zeros((k, n))

        for j in range(k):
            P = pdf(X, m[j], S[j])
            if P is None:
                return None, None
            joint[j] = pi[j] * P

        marginal = np.sum(joint, axis=0)

        if np.any(marginal == 0):
            return None, None

        g = joint / marginal
        lh_sum = np.sum(np.log(marginal))

        return g, lh_sum
    except Exception:
        return None, None
