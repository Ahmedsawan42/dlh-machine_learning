#!/usr/bin/env python3
""" function concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    This function joins two matrices either vertically (along rows, axis=0) or
    horizontally (along columns, axis=1).

    Args:
        mat1 (list of list): The first 2D matrix to concatenate.
        mat2 (list of list): The second 2D matrix to concatenate.
        Matrices must be a rectangular matrix (all rows have same length).
        axis (int, optional): The axis along which to concatenate the matrices.

    Returns:
        list of list or None: A new 2D matrix resulting from the concatenation
        of mat1 and mat2 along the specified axis.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        result = []

        for row in mat1:
            result.append(row.copy())

        for row in mat2:
            result.append(row.copy())

        return result

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result = []

        for i in range(len(mat1)):
            new_row = []

            for element in mat1[i]:
                new_row.append(element)

            for element in mat2[i]:
                new_row.append(element)

            result.append(new_row)

        return result

    else:
        return None
