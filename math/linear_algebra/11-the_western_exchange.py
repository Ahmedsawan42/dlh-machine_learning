#!/usr/bin/env python3
"""function get The transpose of a matrix by NumPy"""


def np_transpose(matrix):
    """
    Get The transpose of a matrix by NumPy.

    Args:
        matrix (numpy.ndarray): A NumPy array whose shape is to be transposed.

    Returns:
        numpy.ndarray: A new NumPy array representing the transpose of input.
    """
    return matrix.T
