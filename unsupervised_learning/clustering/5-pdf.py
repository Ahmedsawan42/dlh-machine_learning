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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    # Check dimensions match
    if m.shape[0] != d:
        return None
    if S.shape != (d, d):
        return None

    # Check if covariance matrix is symmetric (for safety)
    if not np.allclose(S, S.T):
        return None

    # Check if covariance matrix is positive definite (for safety)
    # Using eigvalsh which is more stable than checking eigenvalues manually
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return None

    # Center the data points by subtracting the mean
    # X_centered shape: (n, d)
    X_centered = X - m

    # Calculate the inverse of the covariance matrix
    # Using Cholesky decomposition for numerical stability
    try:
        # Get Cholesky factor L such that S = L @ L.T
        L = np.linalg.cholesky(S)

        # Solve L @ Y = X_centered.T for Y
        Y = np.linalg.solve(L, X_centered.T)

        # Solve L.T @ Z = Y for Z
        # This gives us S^{-1} @ X_centered.T
        Z = np.linalg.solve(L.T, Y)

        # Transpose to get (n, d)
        S_inv_X = Z.T
    except np.linalg.LinAlgError:
        return None

    # Calculate Mahalanobis distance squared for each point
    # Element-wise multiplication of (X_centered) and (S_inv_X)
    # Then sum along axis 1
    mahalanobis_sq = np.sum(X_centered * S_inv_X, axis=1)

    # Calculate the constant term: 1 / ((2*pi)^(d/2) * sqrt(det(S)))
    # Using Cholesky factor: det(S) = (product of diagonal of L)^2
    # Since we can't use np.diag or diagonal, we use a workaround
    # Get diagonal elements using advanced indexing
    diag_indices = np.arange(d)
    diag_L = L[diag_indices, diag_indices]

    # Product of diagonal elements (using numpy.product without loops)
    det_S = np.prod(diag_L) ** 2
    const = 1.0 / ((2.0 * np.pi) ** (d / 2.0) * np.sqrt(det_S))

    # Calculate PDF: P = const * exp(-0.5 * mahalanobis_sq)
    P = const * np.exp(-0.5 * mahalanobis_sq)

    # Ensure minimum value of 1e-300
    P = np.maximum(P, 1e-300)

    return P
