#!/usr/bin/env python3
"""Model that finds the best number of clusters for a GMM using BIC"""

import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM
    using the Bayesian Information Criterion.
    """
    expectation_maximization = __import__('8-EM').expectation_maximization

    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    elif not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    if kmax > n:
        kmax = n

    num_ks = kmax - kmin + 1
    if num_ks < 1:
        return None, None, None, None

    log_likelihoods = np.full(num_ks, -np.inf)
    bic_values = np.full(num_ks, np.inf)
    results = [None] * num_ks

    best_k = None
    best_bic = np.inf
    best_result = None

    # At most 1 loop
    for i, k in enumerate(range(kmin, kmax + 1)):
        try:
            pi, m, S, g, l = expectation_maximization(
                X, k, iterations, tol, verbose
            )
        except Exception:
            continue

        if pi is None or m is None or S is None or g is None or l is None:
            continue

        log_likelihoods[i] = l

        # Parameter count for GMM with full covariance matrices
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # BIC = p * ln(n) - 2 * l
        bic = p * np.log(n) - 2 * l
        bic_values[i] = bic
        results[i] = (pi, m, S)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    if best_k is None:
        return None, None, None, None

    return best_k, best_result, log_likelihoods, bic_values
