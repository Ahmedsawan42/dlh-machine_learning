#!/usr/bin/env python3
"""Function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """
    Compute the transpose of a 2D matrix (list of lists).
    The transpose of a matrix is obtained by swapping its rows and columns.

    Args:
        matrix (list of list): A 2D list, the matrix to be transposed.

    Returns:
        list of list: A new 2D list representing the transposed matrix.
    """
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    transpose = []
    for col_index in range(num_cols):
        new_row = []
        for row_index in range(num_rows):
            new_row.append(matrix[row_index][col_index])
        transpose.append(new_row)

    return transpose
