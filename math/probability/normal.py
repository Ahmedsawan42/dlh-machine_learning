#!/usr/bin/env python3

"""
Class representing Normal distribution Module.
"""


class Normal:
    """Class representing a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the normal distribution.

        Args:
            data: List of data to estimate the distribution (default: None)
            mean: Mean of the distribution (default: 0.0)
            stddev: Standard deviation of the distribution (default: 1.0)
        """
        if data is None:
            # Use the given mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # Calculate mean and stddev from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            # Calculate the mean
            self.mean = float(sum(data) / len(data))

            # Calculate the standard deviation (population standard deviation)
            # Using the formula: sqrt( sum((x - mean)^2) / n )
            sum_squared_diff = sum((x - self.mean) ** 2 for x in data)
            self.stddev = float((sum_squared_diff / len(data)) ** 0.5)

            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
