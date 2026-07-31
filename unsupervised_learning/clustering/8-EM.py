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
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    if k > X.shape[0]:
        return None, None, None, None, None

    n, d = X.shape

    # Initialize parameters using initialize function
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Initialize log likelihood
    l_prev = -np.inf
    l = -np.inf

    # EM algorithm - at most 1 loop
    for i in range(iterations + 1):  # Include iteration 0
        # E-step: Calculate posterior probabilities and log likelihood
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Print log likelihood if verbose
        if verbose and (i % 10 == 0 or i == iterations):
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        # Check for convergence
        if i > 0 and abs(l - l_prev) <= tol:
            break

        # M-step: Update parameters
        pi_new, m_new, S_new = maximization(X, g)
        if pi_new is None or m_new is None or S_new is None:
            return None, None, None, None, None

        # Update parameters
        pi, m, S = pi_new, m_new, S_new
        l_prev = l

    return pi, m, S, g, l
