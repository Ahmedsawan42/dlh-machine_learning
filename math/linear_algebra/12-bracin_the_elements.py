#!/usr/bin/env python3


def np_elementwise(mat1, mat2):
    """
    This function takes two NumPy-compatible arrays and returns a tuple
    containing the element-wise sum, difference, product, and quotient.

    Args:
        mat1 (numpy.ndarray or array-like): The first input array.
        mat2 (numpy.ndarray or array-like): The second input array.

    Returns:
        tuple: A tuple of four numpy.ndarray objects in the following order:
               - Element-wise sum (mat1 + mat2)
               - Element-wise difference (mat1 - mat2)
               - Element-wise product (mat1 * mat2)
               - Element-wise quotient (mat1 / mat2)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
