#!/usr/bin/env python3

"""
Class representing Exponential distribution Module.
"""


class Exponential:
    """Class representing an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize the exponential distribution.

        Args:
            data: List of data to estimate the distribution (default: None)
            lambtha: Expected number of occurrences in a given time frame.
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

            # For exponential distribution, lambtha = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

            # Validate that calculated lambtha is positive
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
