#!/usr/bin/env python3
"""function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """
    Perform element-wise addition of two 2D matrices (lists of lists).
    This function takes two matrices of equal dimensions and returns a matrix
    where each element is the sum of the corresponding elements from the input
    matrices.

    Args:
        mat1 (list of list of numeric): The first 2D matrix to be added.
        mat2 (list of list of numeric): The second 2D matrix to be added.

    Returns:
        list of list of numeric: A new 2D matrix, the element-wise sums.
    """
    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    result = []

    for i in range(len(mat1)):
        new_row = []

        for j in range(len(mat1[i])):
            sum_of_elements = mat1[i][j] + mat2[i][j]
            new_row.append(sum_of_elements)

        result.append(new_row)

    return result
