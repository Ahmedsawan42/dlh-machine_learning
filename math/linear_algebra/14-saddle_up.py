#!/usr/bin/env python3
"""function that performs matrix multiplication"""


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication on two NumPy arrays.

    This function computes the matrix product of two NumPy arrays using the
    `@` operator (Python 3.5+). For 2D arrays, this performs standard matrix
    multiplication. For higher-dimensional arrays, it performs broadcasting
    matrix multiplication as defined by NumPy's matmul behavior.

    Args:
        mat1 (numpy.ndarray): The first NumPy array (left operand).
        mat2 (numpy.ndarray): The second NumPy array (right operand).

    Returns:
        numpy.ndarray: The matrix product of mat1 and mat2.
    """
    return mat1 @ mat2
