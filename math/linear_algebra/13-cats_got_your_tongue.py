#!/usr/bin/env python3
""" function concatenates two matrices along a specific axis"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    It joins a sequence of arrays, allowing for vertical (axis=0), horizontal
    (axis=1), or other dimensional concatenation.

    Args:
        mat1 (numpy.ndarray): The first NumPy array to concatenate.
        mat2 (numpy.ndarray): The second NumPy array to concatenate.
        axis (int, optional): The axis along which to join the arrays.

    Returns:
        numpy.ndarray: A new NumPy array from concatenating mat1 and mat2
                       along the specified axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
