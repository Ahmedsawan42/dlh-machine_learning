#!/usr/bin/env python3

"""
Sum of squares module.

This module provides a function to calculate the sum of squares from 1 to n
without using loops, using the mathematical closed-form formula.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1^2 to n^2.

    Parameters: n : int or float.

    Returns: The integer sum of squares or None if n is not a valid number.
    """
    if n is None:
        return None

    try:
        n_int = int(n)
    except (ValueError, TypeError):
        return None

#    if n_int < 1:
#        return 0  # Return 0 for n < 1 (empty sum)

    # Mathematical closed-form formula: n(n+1)(2n+1)/6
    result = n_int * (n_int + 1) * (2 * n_int + 1) // 6

    return result
