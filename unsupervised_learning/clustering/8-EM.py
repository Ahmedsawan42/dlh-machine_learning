#!/usr/bin/env python3

"""a Model that calculates maximization step in the EM algorithm for a GMM"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing maximum iterations
        tol: non-negative float containing tolerance for log likelihood
        verbose: boolean determining if information should be printed

    Returns:
        pi: numpy.ndarray of shape (k,) containing priors
        m: numpy.ndarray of shape (k, d) containing centroid means
        S: numpy.ndarray of shape (k, d, d) containing covariance matrices
        g: numpy.ndarray of shape (k, n) containing posterior probabilities
        l: log likelihood of the model
        Returns (None, None, None, None, None) on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None, None
        if not isinstance(k, int) or k <= 0:
            return None, None, None, None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None, None, None, None
        if not isinstance(tol, (int, float)) or tol < 0:
            return None, None, None, None, None
        if not isinstance(verbose, bool):
            return None, None, None, None, None

        n, d = X.shape
        if n == 0 or d == 0:
            return None, None, None, None, None

        pi, m, S = initialize(X, k)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, lh_sum = expectation(X, pi, m, S)
        if g is None or lh_sum is None:
            return None, None, None, None, None

        if verbose:
            print(f"Log Likelihood after 0 iterations: {lh_sum:.5f}")

        for i in range(1, iterations + 1):
            pi, m, S = maximization(X, g)
            if pi is None:
                return None, None, None, None, None

            g, l_new = expectation(X, pi, m, S)
            if g is None or l_new is None:
                return None, None, None, None, None

            if verbose and i % 10 == 0:
                print(f"Log Likelihood after {i} iterations: {l_new:.5f}")

            if abs(l_new - lh_sum) <= tol:
                if verbose and i % 10 != 0:
                    print(f"Log Likelihood after {i} iterations: {l_new:.5f}")
                lh_sum = l_new
                break

            lh_sum = l_new
        else:
            if verbose and iterations % 10 != 0:
                print(
                    f"Log Likelihood after {iterations} iterations: "
                    f"{lh_sum:.5f}"
                )

        return pi, m, S, g, lh_sum
    except Exception:
        return None, None, None, None, None
