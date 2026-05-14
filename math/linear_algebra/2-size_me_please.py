#!/usr/bin/env python3


def matrix_shape(matrix):
    """
    Recursively determine the shape of a nested list representing a matrix.
    The function traverses the first element of each sublist recursively to
    determine the dimensionality.

    Args:
        matrix: A nested list structure representing a matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    if not isinstance(matrix, list) or not matrix:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
