"""
Polynomial Integral Module.

This module provides a function to calculate the indefinite integral of a
polynomial represented as a list of coefficients.
"""


def poly_integral(poly, C=0):
    """
    Calculate the indefinite integral of a polynomial.

    Parameters
    ----------
    poly : List of coefficients where poly[i] is coefficient for x^i.
    C : int, Integration constant. Default is 0.

    Returns: New list of coefficients representing the integral polynomial.
    """
    # Check the parameters
    if poly is None:
        return None
    if not isinstance(poly, (list, tuple)):
        return None
    if not isinstance(C, (int, float)):
        return None
    if len(poly) == 0:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    poly_list = list(poly)
    integral = [C]

    for i in range(len(poly_list)):
        new_coeff = poly_list[i] / (i + 1)
        if isinstance(new_coeff, float) and new_coeff.is_integer():
            new_coeff = int(new_coeff)
        integral.append(new_coeff)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
