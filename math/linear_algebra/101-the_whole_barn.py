#!/usr/bin/env python3

"""Function that adds two matrices"""


def add_matrices(mat1, mat2):
    """
    Recursively add two matrices (nested lists) or numbers element-wise.

    Args:
        mat1 (list or int or float): The first matrix (nested list).
        mat2 (list or int or float): The second matrix (nested list).

    Returns:
        list: A new matrix (nested list) containing the element-wise sums.
    """
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) != len(mat2):
            return None

        result = []

        for i in range(len(mat1)):
            added_element = add_matrices(mat1[i], mat2[i])

            if added_element is None:
                return None

            result.append(added_element)

        return result

    elif isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2

    else:
        return None
