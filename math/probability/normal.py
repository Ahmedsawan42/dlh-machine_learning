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
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate the mean
            self.mean = float(sum(data) / len(data))

            # Calculate the standard deviation (population standard deviation)
            # Using the formula: sqrt( sum((x - mean)^2) / n )
            sum_squared_diff = sum((x - self.mean) ** 2 for x in data)
            self.stddev = float((sum_squared_diff / len(data)) ** 0.5)

            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")

    def z_score(self, x):
        """Calculate the z-score of a given x-value.

        Args:
            x: The x-value to convert to a z-score

        Returns:
            The z-score of x
        """
        # Formula: z = (x - μ) / σ
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """Calculate the x-value of a given z-score.

        Args:
            z: The z-score to convert to an x-value

        Returns:
            The x-value corresponding to the z-score
        """
        # Formula: x = μ + z * σ
        x = self.mean + (z * self.stddev)
        return x
