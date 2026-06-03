#!/usr/bin/env python3

"""
This module provides a function to calculate the derivative of a polynomial
represented as a list of coefficients.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Parameters:  poly : list or tuple of coefficients.

    Returns: New list of coefficients representing the derivative polynomial.
    """
    if poly is None:
        return None
    if not isinstance(poly, (list, tuple)):
        return None
    if len(poly) == 0:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    # Derivative term for x^n (at index n) becomes coefficient * n at index n-1
    derivative = []

    for power in range(1, len(poly)):
        coeff = poly[power]
        new_coeff = coeff * power
        derivative.append(new_coeff)

    # If all coefficients became zero (derivative is zero polynomial)
    if len(derivative) == 0 or all(coeff == 0 for coeff in derivative):
        return [0]

    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
