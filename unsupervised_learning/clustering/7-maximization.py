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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Check that g has correct shape (k, n)
    if g.shape[1] != n:
        return None, None, None

    # Check that posterior probabilities sum to 1 for each data point
    # (approximately)
    col_sums = np.sum(g, axis=0)
    if not np.allclose(col_sums, 1.0, rtol=1e-5, atol=1e-5):
        return None, None, None

    # Check for negative values in g
    if np.any(g < 0):
        return None, None, None

    # Initialize outputs
    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Calculate N_k (sum of responsibilities for each cluster)
    # N_k = sum_i g[k][i]
    N_k = np.sum(g, axis=1)  # shape: (k,)

    # Check for empty clusters (N_k should be > 0)
    if np.any(N_k <= 0):
        return None, None, None

    # Update priors: pi_k = N_k / n
    pi = N_k / n

    # Update means: m_k = (sum_i g[k][i] * X[i]) / N_k
    # Use at most 1 loop over clusters
    for j in range(k):
        # Get responsibilities for cluster j
        g_j = g[j]  # shape: (n,)

        # Calculate weighted sum of data points
        # Element-wise multiplication of g_j (n,) with X (n, d)
        # using broadcasting: g_j[:, np.newaxis] * X
        weighted_sum = np.sum(g_j[:, np.newaxis] * X, axis=0)  # shape: (d,)

        # Update mean
        m[j] = weighted_sum / N_k[j]

        # Update covariance matrix
        # S_j = (sum_i g[j][i] * (X[i] - m_j) @ (X[i] - m_j).T) / N_k[j]
        # Center the data
        X_centered = X - m[j]  # shape: (n, d)

        # Calculate weighted covariance
        # For each point, compute outer product and multiply by weight
        # Using broadcasting and vectorized operations
        # g_j[:, np.newaxis, np.newaxis] *
        # (X_centered[:, :, np.newaxis] @ X_centered[:, np.newaxis, :])
        # This would create a large (n, d, d) array, so we do it efficiently
        weighted_cov = np.zeros((d, d))
        for i in range(n):
            if g_j[i] > 1e-300:  # Only compute if weight is significant
                x_i = X_centered[i]  # shape: (d,)
                weighted_cov += g_j[i] * np.outer(x_i, x_i)

        # Alternative more efficient vectorized approach (no inner loop):
        # Using einsum for efficiency
        # weighted_cov = np.einsum('i,ij,ik->jk', g_j, X_centered, X_centered)
        # This computes sum_i g_j[i] * (X_centered[i] * X_centered[i].T)
        # Let's use this vectorized approach instead
        weighted_cov = np.einsum('i,ij,ik->jk', g_j, X_centered, X_centered)

        # Update covariance
        S[j] = weighted_cov / N_k[j]

        # Add small regularization to ensure positive definiteness
        # This helps with numerical stability
        S[j] += 1e-6 * np.eye(d)

    return pi, m, S
