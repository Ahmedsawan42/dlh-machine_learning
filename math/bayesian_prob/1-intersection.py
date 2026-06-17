#!/usr/bin/env python3

"""a Model Calculates the intersectionof obtaining observed data"""

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
    comb_val = float(comb_val)  # Convert to float for NumPy

    # Calculate likelihood for each probability in P
    # Handle edge cases to avoid 0**0 issues
    p_power = np.where(P == 0,
                       np.where(x == 0, 1.0, 0.0),
                       np.power(P, x))

    q_power = np.where(P == 1,
                       np.where(n - x == 0, 1.0, 0.0),
                       np.power((1 - P), (n - x)))

    likelihood_v = comb_val * p_power * q_power

    # Clean up very small values
    likelihood_v = np.where(np.abs(likelihood_v) < 1e-15, 0.0, likelihood_v)

    return likelihood_v


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the data with the various
    hypothetical probabilities (joint probability).

    Intersection = P(data and p) = P(data | p) * P(p)
    where P(data | p) is the likelihood and P(p) is the prior probability.

    Parameters:
    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    P (numpy.ndarray): 1D array of hypothetical probabilities
    Pr (numpy.ndarray): 1D array of prior beliefs of P

    Returns:
    numpy.ndarray: 1D array containing intersection for each probability in P
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

    # Validate Pr
    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Validate all values in P are in [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Validate all values in Pr are in [0, 1]
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Validate Pr sums to 1
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    # Calculate likelihood
    lik = likelihood(x, n, P)

    # Calculate intersection: P(data and p) = P(data | p) * P(p)
    intersection_values = lik * Pr

    return intersection_values
