#!/usr/bin/env python3

"""
Class representing Binomial distribution Module.
"""


class Binomial:
    """Class representing a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize the binomial distribution.

        Args:
            data: List of data to estimate the distribution (default: None)
            n: Number of Bernoulli trials (default: 1)
            p: Probability of a "success" (default: 0.5)
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            # Calculate n and p from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from data
            mean = sum(data) / len(data)

            # Calculate variance: sum((x - mean)^2) / n
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # For binomial distribution:
            # mean = n * p
            # variance = n * p * (1 - p)

            # Calculate p first: p = 1 - (variance / mean)
            # Because variance = n*p*(1-p) = mean*(1-p)
            # So 1-p = variance / mean
            # Therefore p = 1 - (variance / mean)
            p_estimate = 1 - (variance / mean)

            # Then calculate n: n = mean / p
            n_estimate = mean / p_estimate

            # Round n to the nearest integer
            self.n = int(round(n_estimate))

            # Recalculate p using the rounded n
            self.p = float(mean / self.n)

            # Validate that p is a valid probability
            if self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
