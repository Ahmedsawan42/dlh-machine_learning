#!/usr/bin/env python3

"""a Model that calculates the expectation step in the EM algorithm for GMM"""

import numpy as np


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
    # Import pdf function
    pdf = __import__('5-pdf').pdf

    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Check dimensions match
    if m.shape != (k, d):
        return None, None
    if S.shape != (k, d, d):
        return None, None
    if pi.shape[0] != k:
        return None, None

    # Check that priors sum to 1 (approximately)
    if not np.isclose(np.sum(pi), 1.0):
        return None, None

    # Initialize g matrix of shape (k, n)
    g = np.zeros((k, n))

    # Initialize weighted_pdf matrix of shape (k, n)
    weighted_pdf = np.zeros((k, n))

    # Calculate PDF for each cluster (at most 1 loop over k clusters)
    for j in range(k):
        # Calculate PDF for cluster j
        pdf_vals = pdf(X, m[j], S[j])
        if pdf_vals is None:
            return None, None
        # Weight by prior
        weighted_pdf[j] = pi[j] * pdf_vals

    # Calculate denominator (sum over clusters for each data point)
    # Shape: (n,)
    denominator = np.sum(weighted_pdf, axis=0)

    # Check for near-zero denominator (numerical stability)
    # If denominator is 0, set posterior to 0 and use -inf for log likelihood
    zero_mask = (denominator <= 1e-300)

    # Calculate posterior probabilities g = (pi * pdf) / sum(pi * pdf)
    # g shape: (k, n)
    # Use broadcasting: weighted_pdf (k, n) / denominator (n,)
    # Need to add a new axis to denominator to broadcast correctly
    # OR: weighted_pdf[:, zero_mask] = 0
    denominator[zero_mask] = 1.0  # Avoid division by zero, will mask later

    g = weighted_pdf / denominator[np.newaxis, :]

    # Set posterior probabilities to 0 for points where denominator was 0
    g[:, zero_mask] = 0.0

    # Calculate total log likelihood
    # l = sum(log(sum(pi * pdf)))
    # For numerical stability, use log of sum
    # Only sum over non-zero denominator
    # Add small epsilon to avoid -inf
    log_likelihoods = np.log(denominator + 1e-300)
    lh_sum = np.sum(log_likelihoods)

    return g, lh_sum
