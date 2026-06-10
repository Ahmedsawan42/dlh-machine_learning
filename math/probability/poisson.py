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
