#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    """
    This function multiplies two matrices where the number of columns in the
    first matrix must equal the number of rows in the second matrix.

    Args:
        mat1 (list of list of numeric): The first 2D matrix.
        mat2 (list of list of numeric): The second 2D matrix.

    Returns:
        list of list of numeric or None: A new 2D matrix, the maltiplication.
    """
    rows_mat1 = len(mat1)
    cols_mat1 = len(mat1[0])
    rows_mat2 = len(mat2)
    cols_mat2 = len(mat2[0])

    if cols_mat1 != rows_mat2:
        return None

    result = []

    for i in range(rows_mat1):
        new_row = []

        for j in range(cols_mat2):
            sum_product = 0

            for k in range(cols_mat1):
                sum_product = sum_product + (mat1[i][k] * mat2[k][j])

            new_row.append(sum_product)

        result.append(new_row)

    return result
