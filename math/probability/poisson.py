#!/usr/bin/env python3

"""
Class representing a Poisson distribution Module.
"""


class Poisson:
    """Class representing a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the Poisson distribution.

        Args:
            data: List of data to estimate the distribution (default: None)
            lambtha: Expected number of occurrences (default: 1.0)
        """

        self.e = 2.7182818285

        if data is None:
            # Use the given lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Calculate lambtha from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate the mean of the data (which is lambtha for Poisson)
            self.lambtha = float(sum(data) / len(data))

            # Validate that calculated lambtha is positive
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def pmf(self, k):
        """Calculate the Probability Mass Function (PMF) for a given k.

        Args:
            k: Number of "successes" (non-negative integer)

        Returns:
            PMF value for k (probability of exactly k occurrences)
        """
        # Convert k to integer if it's not already
        k = int(k)

        # If k is out of range (negative), return 0
        if k < 0:
            return 0

        # Calculate factorial iteratively to match expected precision
        def factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result

        # Calculate PMF: P(X = k) = (e^(-λ) * λ^k) / k!
        e_to_negative_lambtha = self.e ** (-self.lambtha)
        lambtha_pow_k = self.lambtha ** k
        k_factorial = factorial(k)

        result = (e_to_negative_lambtha * lambtha_pow_k) / k_factorial
        return result

    def cdf(self, k):
        """Calculate the Cumulative Distribution Function (CDF) for a given k.

        Args:
            k: Number of "successes" (non-negative integer)

        Returns:
            CDF value for k (probability of k or fewer occurrences)
        """
        # Convert k to integer if it's not already
        k = int(k)

        # If k is out of range (negative), return 0
        if k < 0:
            return 0

        # Calculate CDF: P(X ≤ k) = sum_{i=0}^{k} P(X = i)
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
