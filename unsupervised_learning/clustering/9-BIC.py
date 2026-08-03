#!/usr/bin/env python3

"""a Model that Finds the best number of clusters for a GMM using the BIC"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM
    using the Bayesian Information Criterion.
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None
        n, d = X.shape
        if n == 0 or d == 0:
            return None, None, None, None
        if not isinstance(kmin, int) or kmin < 1:
            return None, None, None, None
        if kmax is None:
            kmax = n
        if not isinstance(kmax, int) or kmax < 1:
            return None, None, None, None
        if kmax <= kmin:
            return None, None, None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None, None, None
        if not isinstance(tol, (int, float)) or tol < 0:
            return None, None, None, None
        if not isinstance(verbose, bool):
            return None, None, None, None

        k_range = kmax - kmin + 1
        lh_sum = np.zeros(k_range)
        b = np.zeros(k_range)

        best_k = None
        best_result = None
        best_bic = np.inf

        for i, k in enumerate(range(kmin, kmax + 1)):
            pi, m, S, g, l_val = expectation_maximization(
                X, k, iterations=iterations, tol=tol, verbose=verbose
            )
            if pi is None:
                return None, None, None, None

            p = k - 1 + k * d + k * d * (d + 1) // 2
            bic_val = p * np.log(n) - 2 * l_val

            lh_sum[i] = l_val
            b[i] = bic_val

            if bic_val < best_bic:
                best_bic = bic_val
                best_k = k
                best_result = (pi, m, S)

        return best_k, best_result, lh_sum, b
    except Exception:
        return None, None, None, None
