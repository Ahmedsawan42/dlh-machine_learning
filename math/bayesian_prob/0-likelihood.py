#!/usr/bin/env python3

"""a Model Calculates the likelihood of obtaining observed data"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining observed data given various
    hypothetical probabilities of developing severe side effects.

    Parameters:
    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
    numpy.ndarray: 1D array containing likelihood for each probability in P
    """
    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    # Validate x <= n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")

    if P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate all values in P are in [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate binomial coefficient C(n, x)
    def factorial(num):
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result

    comb_val = factorial(n) // (factorial(x) * factorial(n - x))

    # Calculate likelihood for each probability in P
    # L(p) = C(n, x) * p^x * (1-p)^(n-x)
    likelihood_values = comb_val * (P ** x) * ((1 - P) ** (n - x))

    return likelihood_values
